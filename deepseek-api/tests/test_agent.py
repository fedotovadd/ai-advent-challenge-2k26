import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import agent
from agent import Agent, AgentRegistry, MAX_BULK_AGENTS, PersistenceError, default_settings


class AgentTests(unittest.TestCase):
    def test_estimate_payload_counts_system_messages_and_request_options(self):
        request = {
            "model": "deepseek-v4-flash",
            "messages": [
                {"role": "system", "content": "Будь кратким"},
                {"role": "user", "content": "Привет"},
            ],
            "temperature": 1,
            "max_tokens": 300,
        }

        self.assertGreater(agent.estimate_text_tokens("один два"), 0)
        self.assertGreater(agent.estimate_payload_tokens(request), agent.estimate_text_tokens("Привет"))
        self.assertGreater(
            agent.estimate_payload_tokens(request),
            agent.estimate_payload_tokens({**request, "messages": request["messages"][1:]}),
        )

    def test_usage_uses_estimates_when_usage_is_missing_or_partial(self):
        actual, _ = agent.request_usage_and_cost(
            "deepseek-v4-flash", {"prompt_tokens": 10, "completion_tokens": 5}, 12, 4,
        )
        estimated, cost = agent.request_usage_and_cost("deepseek-v4-flash", {"prompt_tokens": 10}, 12, 4)

        self.assertEqual(actual["source"], "actual")
        self.assertEqual(estimated, {
            "promptTokens": 12, "completionTokens": 4, "totalTokens": 16, "source": "estimated",
        })
        self.assertEqual(cost["kind"], "paid")

    def test_model_capabilities_keep_real_context_and_output_limits(self):
        self.assertEqual(agent.MODEL_CAPABILITIES["deepseek-v4-pro"]["contextLimit"], 1_000_000)
        self.assertEqual(agent.MODEL_CAPABILITIES["deepseek-v4-pro"]["maxOutputTokens"], 384_000)
        self.assertEqual(agent.MODEL_CAPABILITIES["glm-4.7-flash"]["contextLimit"], 200_000)
        self.assertEqual(agent.MODEL_CAPABILITIES["glm-4.7-flash"]["maxOutputTokens"], 131_072)

    def test_successful_response_records_isolated_session_metrics(self):
        def ask_model(payload, **options):
            return {"content": "Ответ", "usage": {"prompt_tokens": 100, "completion_tokens": 50}}

        first = Agent("agent-1", "Первый", default_settings(), ask_model)
        second = Agent("agent-2", "Второй", default_settings(), ask_model)
        result = first.respond("Привет")

        record = result["metrics"]["calls"][0]
        self.assertEqual(record["messageNumber"], 1)
        self.assertEqual(record["promptTokens"], 100)
        self.assertEqual(record["completionTokens"], 50)
        self.assertIn("historyBeforeTokens", record)
        self.assertIn("historyAfterTokens", record)
        self.assertEqual(result["metrics"]["totals"]["totalTokens"], 150)
        self.assertEqual(second.snapshot()["metrics"]["calls"], [])

    def test_overflow_blocks_request_without_changing_history(self):
        calls = []
        settings = default_settings()
        settings["trainingContextLimit"] = 10
        agent_instance = Agent("agent-1", "Первый", settings, lambda *args, **kwargs: calls.append(args))

        with self.assertRaisesRegex(agent.ContextOverflowError, "превышают лимит"):
            agent_instance.respond("Длинное сообщение")

        self.assertEqual(calls, [])
        self.assertEqual(agent_instance.snapshot()["messages"], [])
        self.assertEqual(agent_instance.snapshot()["metrics"]["lastContextAttempt"]["mode"], "blocked")

    def test_overflow_probe_returns_provider_message_without_committing_message(self):
        class ProviderFailure(RuntimeError):
            body = {"error": {"message": "maximum context length exceeded"}}

        settings = default_settings()
        settings.update({"trainingContextLimit": 10, "sendOnOverflow": True})
        agent_instance = Agent("agent-1", "Первый", settings, lambda *args, **kwargs: (_ for _ in ()).throw(ProviderFailure()))

        with self.assertRaisesRegex(agent.OverflowProbeError, "maximum context"):
            agent_instance.respond("Длинное сообщение")

        self.assertEqual(agent_instance.snapshot()["messages"], [])

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
            "source": "actual",
        })
        self.assertEqual(result["metadata"]["cost"], {"kind": "paid", "usd": 0.0000165, "source": "actual"})

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


class AgentRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.state_path = Path(self.temporary_directory.name) / "agents.json"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_registry_restores_messages_settings_and_context(self):
        first_registry = AgentRegistry(lambda payload, **options: "Рада познакомиться!", self.state_path)
        settings = default_settings()
        settings.update({"model": "deepseek-v4-pro", "systemPrompt": "Отвечай дружелюбно."})

        first_registry.update_settings("agent-1", settings)
        first_registry.respond("agent-1", "Меня зовут Маша", 1)

        calls = []

        def ask_model(payload, **options):
            calls.append(payload)
            return "Тебя зовут Маша."

        restarted_registry = AgentRegistry(ask_model, self.state_path)
        restored = restarted_registry.get("agent-1").snapshot()
        restarted_registry.respond("agent-1", "Как меня зовут?", 1)

        self.assertEqual(restored["settings"], settings)
        self.assertEqual(restored["messages"], [
            {"role": "user", "content": "Меня зовут Маша"},
            {"role": "assistant", "content": "Рада познакомиться!"},
        ])
        self.assertEqual(restored["metadata"]["status"], {"kind": "success", "label": "200 OK"})
        self.assertEqual(calls[0]["messages"], [
            {"role": "system", "content": "Отвечай дружелюбно."},
            {"role": "user", "content": "Меня зовут Маша"},
            {"role": "assistant", "content": "Рада познакомиться!"},
            {"role": "user", "content": "Как меня зовут?"},
        ])

    def test_registry_ignores_invalid_saved_state(self):
        self.state_path.write_text(json.dumps({
            "version": 1,
            "nextId": 2,
            "agents": [{
                "id": "agent-1",
                "name": "Сохранённый агент",
                "messages": [],
                "settings": default_settings(),
                "metadata": {
                    "userPrompt": 42,
                    "systemPrompt": "Отвечай ясно.",
                    "payload": {},
                    "status": {"kind": "success", "label": "200 OK"},
                    "responseTimeMs": None,
                    "usage": None,
                    "cost": None,
                },
            }],
        }), encoding="utf-8")

        registry = AgentRegistry(lambda payload, **options: "Ответ", self.state_path)

        self.assertEqual(registry.agents(), [{
            "id": "agent-1",
            "name": "Агент 1",
            "messages": [],
            "settings": default_settings(),
            "metadata": None,
            "metrics": agent.default_metrics(),
        }])

    def test_create_many_adds_requested_agents_with_default_configuration(self):
        registry = AgentRegistry(lambda payload, **options: "Ответ", self.state_path)

        created = registry.create_many(3)

        self.assertEqual([agent["id"] for agent in created], ["agent-2", "agent-3", "agent-4"])
        self.assertEqual([agent["name"] for agent in created], ["Агент 2", "Агент 3", "Агент 4"])
        self.assertEqual(len(registry.agents()), 4)
        self.assertEqual(registry.agents()[0]["id"], "agent-1")
        self.assertEqual(registry.agents()[0]["name"], "Агент 1")
        self.assertTrue(all(agent["settings"] == default_settings() for agent in created))

    def test_agents_created_by_registry_keep_messages_and_settings_isolated(self):
        calls = []

        def ask_model(payload, **options):
            calls.append(payload)
            return "Ответ"

        registry = AgentRegistry(ask_model, self.state_path)
        first, second = registry.create_many(2)
        settings = default_settings()
        settings["systemPrompt"] = "Отвечай только одним словом."

        registry.get(first["id"]).update_settings(settings)
        registry.get(first["id"]).respond("Привет", 1)

        self.assertEqual(calls[0]["messages"][0]["content"], "Отвечай только одним словом.")
        self.assertEqual(registry.get(second["id"]).snapshot()["messages"], [])
        self.assertNotEqual(
            registry.get(second["id"]).snapshot()["settings"]["systemPrompt"],
            "Отвечай только одним словом.",
        )

    def test_create_many_rejects_counts_outside_allowed_range(self):
        registry = AgentRegistry(lambda payload, **options: "Ответ", self.state_path)

        for count in (0, -1, MAX_BULK_AGENTS + 1, True, "3"):
            with self.subTest(count=count):
                with self.assertRaises(ValueError):
                    registry.create_many(count)

    def test_delete_removes_agent_history_settings_and_metadata_from_saved_state(self):
        registry = AgentRegistry(lambda payload, **options: "Секретный ответ", self.state_path)
        created = registry.create()
        settings = default_settings()
        settings["systemPrompt"] = "Не сохраняй этот текст."
        registry.update_settings(created["id"], settings)
        registry.respond(created["id"], "Секретный вопрос", 1)

        deleted = registry.delete(created["id"])

        self.assertEqual(deleted["id"], created["id"])
        self.assertIsNone(registry.get(created["id"]))
        self.assertEqual([snapshot["id"] for snapshot in registry.agents()], ["agent-1"])
        saved_state = self.state_path.read_text(encoding="utf-8")
        self.assertNotIn(created["id"], saved_state)
        self.assertNotIn("Секретный вопрос", saved_state)
        self.assertNotIn("Не сохраняй этот текст.", saved_state)
        self.assertNotIn("Секретный ответ", saved_state)

    def test_delete_last_agent_persists_empty_registry_and_keeps_next_id(self):
        registry = AgentRegistry(lambda payload, **options: "Ответ", self.state_path)

        registry.delete("agent-1")

        restarted = AgentRegistry(lambda payload, **options: "Ответ", self.state_path)
        self.assertEqual(restarted.agents(), [])
        self.assertEqual(restarted.create()["id"], "agent-2")

    def test_delete_rolls_back_memory_when_persistence_fails(self):
        registry = AgentRegistry(lambda payload, **options: "Ответ", self.state_path)
        registry.create()
        before = self.state_path.read_text(encoding="utf-8")

        with patch.object(registry, "_save_state", side_effect=PersistenceError("Диск недоступен")):
            with self.assertRaises(PersistenceError):
                registry.delete("agent-1")

        self.assertIsNotNone(registry.get("agent-1"))
        self.assertEqual(self.state_path.read_text(encoding="utf-8"), before)


if __name__ == "__main__":
    unittest.main()
