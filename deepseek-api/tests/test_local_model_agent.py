import tempfile
import unittest
from pathlib import Path

from agent import AgentRegistry, ContextOverflowError, _valid_settings, default_settings


class LocalModelAgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "agents.json"
        self.calls = []

    def answer(self, payload, **options):
        self.calls.append((payload, options))
        return "Локальный ответ"

    def local_registry(self, ask=None, mcp_registry=None):
        registry = AgentRegistry(ask or self.answer, self.path, mcp_registry=mcp_registry)
        registry.update_settings("agent-1", {**default_settings(), "model": "qwen2.5:1.5b"})
        return registry

    def test_default_output_is_sent_reserved_and_saved_with_local_agent(self):
        registry = self.local_registry()
        other_id = registry.create()["id"]
        snapshot = registry.respond("agent-1", "Первый вопрос")
        self.assertEqual(self.calls[0][1]["max_tokens"], 512)
        self.assertEqual(snapshot["metadata"]["payload"]["max_tokens"], 512)
        self.assertEqual(snapshot["metrics"]["lastMessage"]["contextLimit"], 4096)
        self.assertEqual(snapshot["metadata"]["cost"]["usd"], 0)
        self.assertEqual(snapshot["messages"][-1]["content"], "Локальный ответ")
        registry.respond("agent-1", "Второй вопрос")
        self.assertIn({"role": "user", "content": "Первый вопрос"}, self.calls[1][0]["messages"])
        self.assertIn({"role": "assistant", "content": "Локальный ответ"}, self.calls[1][0]["messages"])
        restored = AgentRegistry(self.answer, self.path)
        self.assertEqual(restored.get("agent-1").snapshot()["settings"]["model"], "qwen2.5:1.5b")
        self.assertEqual(restored.get(other_id).snapshot()["settings"]["model"], default_settings()["model"])
        self.assertEqual(restored.get(other_id).snapshot()["messages"], [])

    def test_explicit_limit_is_respected_and_over_1024_is_rejected(self):
        registry = self.local_registry()
        registry.update_settings("agent-1", {**default_settings(), "model": "qwen2.5:1.5b", "maxTokens": 1024})
        snapshot = registry.respond("agent-1", "Вопрос")
        self.assertEqual(self.calls[-1][1]["max_tokens"], 1024)
        self.assertEqual(snapshot["metadata"]["payload"]["max_tokens"], 1024)
        self.assertFalse(_valid_settings({**default_settings(), "model": "qwen2.5:1.5b", "maxTokens": 1025}))

    def test_context_check_reserves_default_512_tokens(self):
        registry = self.local_registry()
        registry.update_settings("agent-1", {**default_settings(), "model": "qwen2.5:1.5b", "trainingContextLimit": 512})
        with self.assertRaises(ContextOverflowError):
            registry.respond("agent-1", "Вопрос")
        attempt = registry.get("agent-1").snapshot()["metrics"]["lastContextAttempt"]
        self.assertEqual(attempt["requestedOutputTokens"], 512)
        self.assertEqual(self.calls, [])

    def test_local_model_does_not_send_mcp_tools(self):
        class FakeMcp:
            def snapshot(self):
                return [{"type": "function", "function": {"name": "example"}}]
        registry = self.local_registry(mcp_registry=FakeMcp())
        registry.respond("agent-1", "Вопрос")
        self.assertNotIn("tools", self.calls[-1][1])

    def test_failure_keeps_unanswered_user_message_and_makes_one_call(self):
        def unavailable(payload, **options):
            self.calls.append((payload, options))
            raise RuntimeError("offline")
        registry = self.local_registry(ask=unavailable)
        with self.assertRaises(RuntimeError):
            registry.respond("agent-1", "Вопрос")
        restored = AgentRegistry(unavailable, self.path).get("agent-1").snapshot()
        self.assertEqual(restored["messages"], [{"role": "user", "content": "Вопрос"}])
        self.assertEqual(restored["metadata"]["status"]["kind"], "error")
        self.assertEqual(len(self.calls), 1)


if __name__ == "__main__":
    unittest.main()
