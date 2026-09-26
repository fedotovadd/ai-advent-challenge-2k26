import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import providers


class ProviderTests(unittest.TestCase):
    @patch("providers.OpenAI")
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
    @patch('providers.OpenAI')
    def test_tool_only_response_is_valid(self, openai):
        tool = SimpleNamespace(id='call_1', type='function', function=SimpleNamespace(name='repo', arguments='{}'))
        openai.return_value.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[tool]))], usage=None)
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'fake'}):
            result = providers.ask_model({'model':'deepseek-v4-flash','messages':[], 'temperature':1}, tools=[])
        self.assertEqual(result['tool_calls'][0]['id'], 'call_1')
        self.assertIsNone(result['content'])
