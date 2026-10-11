import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import httpx
from openai import APIConnectionError, APITimeoutError, NotFoundError

from deepseek_app.providers import providers


class ProviderTests(unittest.TestCase):
    @patch("deepseek_app.providers.providers.OpenAI")
    def test_local_model_uses_loopback_without_cloud_keys_and_forwards_options(self, openai):
        message = SimpleNamespace(content="Локальный ответ", tool_calls=None)
        openai.return_value.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=message)], usage=None)
        payload = {"model": "qwen2.5:1.5b", "messages": [{"role": "user", "content": "Привет"}], "temperature": 0.7}
        with patch("deepseek_app.providers.providers.os.getenv", side_effect=AssertionError("cloud key read")):
            result = providers.ask_model(payload, max_tokens=512, stop="КОНЕЦ")
        openai.assert_called_once_with(api_key="ollama", base_url="http://127.0.0.1:11434/v1/", timeout=120, max_retries=0)
        openai.return_value.chat.completions.create.assert_called_once_with(
            model="qwen2.5:1.5b", messages=payload["messages"], temperature=0.7,
            max_tokens=512, stop="КОНЕЦ")
        self.assertEqual(result["content"], "Локальный ответ")

    @patch("deepseek_app.providers.providers.OpenAI")
    def test_local_provider_errors_are_classified(self, openai):
        request = httpx.Request("POST", "http://127.0.0.1:11434/v1/chat/completions")
        cases = [
            (APIConnectionError(request=request), "unavailable"),
            (APITimeoutError(request=request), "timeout"),
            (NotFoundError("missing", response=httpx.Response(404, request=request), body={}), "missing_model"),
        ]
        for failure, expected_kind in cases:
            with self.subTest(expected_kind=expected_kind):
                openai.return_value.chat.completions.create.side_effect = failure
                with self.assertRaises(providers.LocalModelError) as caught:
                    providers.ask_model({"model": "qwen2.5:1.5b", "messages": [], "temperature": 1})
                self.assertEqual(caught.exception.kind, expected_kind)

    @patch("deepseek_app.providers.providers.OpenAI")
    def test_local_empty_answer_is_an_error(self, openai):
        openai.return_value.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=None))], usage=None)
        with self.assertRaises(providers.LocalModelError) as caught:
            providers.ask_model({"model": "qwen2.5:1.5b", "messages": [], "temperature": 1})
        self.assertEqual(caught.exception.kind, "empty")

    @patch("deepseek_app.providers.providers.OpenAI")
    def test_local_route_never_enables_sdk_retries(self, openai):
        openai.return_value.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Ок", tool_calls=None))], usage=None)
        providers.ask_model({"model": "qwen2.5:1.5b", "messages": [], "temperature": 1}, max_retries=3)
        self.assertEqual(openai.call_args.kwargs["max_retries"], 0)

    @patch("deepseek_app.providers.providers.OpenAI")
    def test_deepseek_route_still_uses_its_cloud_key(self, openai):
        openai.return_value.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Облако", tool_calls=None))], usage=None)
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "deepseek-key"}, clear=True):
            providers.ask_model({"model": "deepseek-v4-flash", "messages": [], "temperature": 1})
        openai.assert_called_once_with(api_key="deepseek-key", base_url="https://api.deepseek.com")

    @patch("deepseek_app.providers.providers.OpenAI")
    def test_glm_model_uses_zai_endpoint_and_key(self, openai):
        openai.return_value.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Ответ GLM"))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        payload = {
            "model": "glm-4.7-flash",
            "messages": [{"role": "user", "content": "Привет"}],
            "temperature": 1,
        }

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "deepseek-key", "ZAI_API_KEY": "zai-key"}, clear=True):
            result = providers.ask_model(payload)

        openai.assert_called_once_with(api_key="zai-key", base_url="https://api.z.ai/api/paas/v4")
        self.assertEqual(result["content"], "Ответ GLM")


if __name__ == "__main__":
    unittest.main()

class ProviderToolTests(unittest.TestCase):
    @patch('deepseek_app.providers.providers.OpenAI')
    def test_tool_only_response_is_valid(self, openai):
        tool = SimpleNamespace(id='call_1', type='function', function=SimpleNamespace(name='repo', arguments='{}'))
        openai.return_value.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[tool]))], usage=None)
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'fake'}):
            result = providers.ask_model({'model':'deepseek-v4-flash','messages':[], 'temperature':1}, tools=[])
        self.assertEqual(result['tool_calls'][0]['id'], 'call_1')
        self.assertIsNone(result['content'])

    @patch('deepseek_app.providers.providers.OpenAI')
    def test_per_call_retry_override_is_applied_to_client_not_api_body(self, openai):
        openai.return_value.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{}'))], usage=None)
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'fake'}):
            providers.ask_model({'model':'deepseek-v4-flash','messages':[], 'temperature':0}, max_retries=0, timeout=30)
        self.assertEqual(openai.call_args.kwargs.get('max_retries'), 0)
        self.assertNotIn('max_retries', openai.return_value.chat.completions.create.call_args.kwargs)
        self.assertEqual(openai.return_value.chat.completions.create.call_args.kwargs['timeout'], 30)
