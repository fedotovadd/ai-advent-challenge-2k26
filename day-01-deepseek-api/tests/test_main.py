import io
import os
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import MagicMock, patch

import main


class DeepSeekCliTests(unittest.TestCase):
    def run_main(
        self,
        *,
        api_key="test-key",
        prompt="Привет",
        input_error=None,
        client_error=None,
        api_error=None,
        response_content="Тестовый ответ",
        empty_choices=False,
    ):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with (
            patch.dict(os.environ, {"DEEPSEEK_API_KEY": api_key} if api_key else {}, clear=True),
            patch("main.OpenAI") as openai_class,
            patch("builtins.input", return_value=prompt, side_effect=input_error),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            if client_error:
                openai_class.side_effect = client_error
            else:
                response = MagicMock()
                response.choices = [] if empty_choices else [MagicMock()]
                if response.choices:
                    response.choices[0].message.content = response_content
                openai_class.return_value.chat.completions.create.return_value = response
                openai_class.return_value.chat.completions.create.side_effect = api_error

            try:
                exit_code = main.main()
            except Exception as error:
                self.fail(f"main() не должна выбрасывать исключение: {error}")

        return exit_code, stdout.getvalue(), stderr.getvalue(), openai_class

    def test_missing_key_stops_before_creating_client(self):
        exit_code, _, stderr, openai_class = self.run_main(api_key="")

        self.assertEqual(exit_code, 1)
        self.assertIn("DEEPSEEK_API_KEY", stderr)
        openai_class.assert_not_called()

    def test_empty_prompt_stops_before_creating_client(self):
        exit_code, _, stderr, openai_class = self.run_main(prompt="   ")

        self.assertEqual(exit_code, 1)
        self.assertIn("запрос не может быть пустым", stderr)
        openai_class.assert_not_called()

    def test_success_sends_expected_request_and_prints_answer(self):
        exit_code, stdout, stderr, openai_class = self.run_main(prompt="Расскажи об API")

        self.assertEqual(exit_code, 0)
        self.assertIn("Тестовый ответ", stdout)
        self.assertEqual(stderr, "")
        openai_class.assert_called_once_with(
            api_key="test-key",
            base_url="https://api.deepseek.com",
        )
        openai_class.return_value.chat.completions.create.assert_called_once_with(
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": "Расскажи об API"}],
        )

    def test_api_error_is_reported_without_traceback(self):
        exit_code, _, stderr, _ = self.run_main(api_error=RuntimeError("сеть недоступна"))

        self.assertEqual(exit_code, 1)
        self.assertIn("Ошибка при обращении к DeepSeek", stderr)
        self.assertIn("сеть недоступна", stderr)

    def test_client_initialization_error_is_reported_without_traceback(self):
        exit_code, _, stderr, _ = self.run_main(client_error=RuntimeError("ошибка клиента"))

        self.assertEqual(exit_code, 1)
        self.assertIn("Ошибка при обращении к DeepSeek", stderr)
        self.assertIn("ошибка клиента", stderr)

    def test_empty_api_response_is_reported_without_traceback(self):
        exit_code, _, stderr, _ = self.run_main(empty_choices=True)

        self.assertEqual(exit_code, 1)
        self.assertIn("DeepSeek вернул пустой ответ", stderr)

    def test_eof_is_reported_without_traceback(self):
        exit_code, _, stderr, openai_class = self.run_main(input_error=EOFError())

        self.assertEqual(exit_code, 1)
        self.assertIn("ввод отменён", stderr)
        openai_class.assert_not_called()


if __name__ == "__main__":
    unittest.main()
