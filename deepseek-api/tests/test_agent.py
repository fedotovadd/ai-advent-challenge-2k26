import unittest

from agent import Agent, default_settings


class AgentTests(unittest.TestCase):
    def test_respond_builds_request_from_its_settings_and_saves_answer(self):
        calls = []

        def ask_model(payload, **options):
            calls.append({"payload": payload, "options": options})
            return {
                "content": "Готово",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }

        settings = default_settings()
        settings.update({"model": "deepseek-v4-pro", "systemPrompt": "Отвечай кратко."})
        agent = Agent("agent-1", "Первый агент", settings, ask_model)

        result = agent.respond("  Привет  ", 0.4)

        self.assertEqual(calls, [{
            "payload": {
                "model": "deepseek-v4-pro",
                "messages": [
                    {"role": "system", "content": "Отвечай кратко."},
                    {"role": "user", "content": "Привет"},
                ],
                "temperature": 0.4,
            },
            "options": {},
        }])
        self.assertEqual(result["id"], "agent-1")
        self.assertEqual(result["name"], "Первый агент")
        self.assertEqual(result["messages"], [
            {"role": "user", "content": "Привет"},
            {"role": "assistant", "content": "Готово"},
        ])
        self.assertEqual(result["metadata"]["usage"], {
            "promptTokens": 10,
            "completionTokens": 5,
            "totalTokens": 15,
        })
        self.assertEqual(result["metadata"]["cost"], {"kind": "paid", "usd": 0.0000165})

    def test_respond_sends_only_its_own_previous_messages(self):
        calls = []
        answers = iter(["Первый ответ", "Второй ответ", "Продолжение первого ответа"])

        def ask_model(payload, **options):
            calls.append({"payload": payload, "options": options})
            return next(answers)

        first = Agent("agent-1", "Первый агент", default_settings(), ask_model)
        second = Agent("agent-2", "Второй агент", default_settings(), ask_model)

        first.respond("Первый вопрос", 1)
        second.respond("Другой вопрос", 1)
        first.respond("Продолжи", 1)

        self.assertEqual(calls[-1]["payload"]["messages"], [
            {"role": "system", "content": default_settings()["systemPrompt"]},
            {"role": "user", "content": "Первый вопрос"},
            {"role": "assistant", "content": "Первый ответ"},
            {"role": "user", "content": "Продолжи"},
        ])
        self.assertEqual(second.snapshot()["messages"], [
            {"role": "user", "content": "Другой вопрос"},
            {"role": "assistant", "content": "Второй ответ"},
        ])

    def test_respond_applies_json_options_and_saves_exact_metadata(self):
        calls = []

        def ask_model(payload, **options):
            calls.append({"payload": payload, "options": options})
            return "{\"answer\": \"ok\"}"

        settings = default_settings()
        settings.update({"format": "json", "maxTokens": 300, "stop": "<END>"})
        agent = Agent("agent-1", "JSON агент", settings, ask_model)

        result = agent.respond("Верни JSON", 1)

        self.assertEqual(calls[0]["options"], {
            "response_format": {"type": "json_object"},
            "max_tokens": 300,
            "stop": "<END>",
        })
        self.assertEqual(result["metadata"]["systemPrompt"], (
            default_settings()["systemPrompt"] + "\n\nВерни только валидный JSON без Markdown-разметки."
        ))
        self.assertEqual(result["metadata"]["payload"], {
            "model": "deepseek-v4-flash",
            "messages": [
                {"role": "system", "content": result["metadata"]["systemPrompt"]},
                {"role": "user", "content": "Верни JSON"},
            ],
            "temperature": 1,
            "response_format": {"type": "json_object"},
            "max_tokens": 300,
            "stop": "<END>",
        })

    def test_failed_response_keeps_user_message_and_records_error_metadata(self):
        def ask_model(payload, **options):
            raise RuntimeError("сеть недоступна")

        agent = Agent("agent-1", "Первый агент", default_settings(), ask_model)

        with self.assertRaisesRegex(RuntimeError, "сеть недоступна"):
            agent.respond("Привет", 1)

        result = agent.snapshot()
        self.assertEqual(result["messages"], [{"role": "user", "content": "Привет"}])
        self.assertEqual(result["metadata"]["status"], {"kind": "error", "label": "Ошибка API"})


if __name__ == "__main__":
    unittest.main()
