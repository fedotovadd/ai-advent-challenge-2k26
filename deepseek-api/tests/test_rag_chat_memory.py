import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag_chat_memory import empty_state, validate_preparation, conversation_context, prepare_payload


class MemoryTests(unittest.TestCase):
    def test_add_replace_remove_and_provenance(self):
        state = empty_state()
        self.assertEqual(state, {"goal": None, "clarifications": [], "constraints": [], "terms": []})
        state, query = validate_preparation({"search_query": "  книга  ", "changes": [
            {"op": "set_goal", "text": "изучить агента", "evidence": "изучить агента"},
            {"op": "add_constraint", "text": "без кода", "evidence": "без кода"},
            {"op": "add_clarification", "text": "кратко", "evidence": "кратко"},
            {"op": "set_term", "term": "агент", "text": "цикл с инструментами", "evidence": "агент — цикл с инструментами"},
        ]}, "Хочу изучить агента, без кода, кратко; агент — цикл с инструментами", state, 1)
        self.assertEqual(query, "книга")
        self.assertEqual(state["goal"], {"text": "изучить агента", "turn": 1})
        self.assertEqual(state["terms"][0]["term"], "агент")
        state, _ = validate_preparation({"search_query": "агент", "changes": [
            {"op": "add_constraint", "text": "без кода", "evidence": "без кода"},
            {"op": "remove_constraint", "text": "без кода", "evidence": "теперь можно с кодом"},
            {"op": "add_constraint", "text": "с кодом", "evidence": "с кодом"},
        ]}, "без кода — теперь можно с кодом", state, 2)
        self.assertEqual(state["constraints"], [{"text": "с кодом", "turn": 2}])
        self.assertEqual(state["goal"]["turn"], 1)

    def test_invalid_patch_is_atomic(self):
        original = empty_state()
        for patch in (
            {"search_query": "", "changes": []},
            {"search_query": "x", "changes": [{"op": "set_goal", "text": "a", "evidence": "not here"}]},
            {"search_query": "x", "changes": [{"op": "bogus", "text": "a", "evidence": "a"}]},
            {"search_query": "x", "changes": [{"op": "remove_constraint", "text": "a", "evidence": "a"}]},
            {"search_query": "x", "changes": [{"op": "add_constraint", "text": "a" * 301, "evidence": "a"}]},
        ):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                validate_preparation(patch, "a", original, 1)
        self.assertEqual(original, empty_state())

    def test_fact_text_and_term_must_be_in_current_message(self):
        for change in (
            {"op": "set_goal", "text": "discard the project rules", "evidence": "hello"},
            {"op": "add_constraint", "text": "secret limit", "evidence": "hello"},
            {"op": "set_term", "term": "invented", "text": "hello", "evidence": "hello"},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_preparation({"search_query": "hello", "changes": [change]},
                                     "hello", empty_state(), 1)

    def test_context_is_bounded_and_labels_prior_answers(self):
        turns = [{"user_text": f"user{i}" + "u" * 1000,
                  "result": {"answer": f"answer{i}" + "a" * 1000}} for i in range(8)]
        context = conversation_context(empty_state(), turns)
        self.assertLessEqual(len(context), 35000)
        self.assertNotIn("user3", context)
        self.assertIn("user4", context)
        self.assertIn("answer7", context)
        self.assertIn("не доказательство", context)

    def test_large_state_does_not_hide_recent_turns(self):
        state = empty_state()
        state["constraints"] = [{"text": "x" * 300, "turn": i} for i in range(20)]
        context = conversation_context(state, [{"user_text": "recent user",
                                                "result": {"answer": "recent answer"}}])
        self.assertLessEqual(len(context), 35000)
        self.assertIn("recent user", context)
        self.assertIn("recent answer", context)

    def test_maximum_valid_state_keeps_last_constraint_and_term(self):
        state = empty_state()
        state["clarifications"] = [{"text": f"clarification{i}-" + "c" * 280,
                                    "turn": i + 1} for i in range(20)]
        state["constraints"] = [{"text": f"constraint{i}-" + "x" * 280,
                                 "turn": i + 21} for i in range(20)]
        state["terms"] = [{"term": f"term{i}-" + "t" * 280,
                           "text": f"definition{i}-" + "d" * 280,
                           "turn": i + 41} for i in range(20)]
        context = conversation_context(state, [{"user_text": "recent user",
                                                "result": {"answer": "recent answer"}}])
        self.assertIn("clarification19-", context)
        self.assertIn("constraint19-", context)
        self.assertIn("term19-", context)
        self.assertIn("definition19-", context)
        self.assertIn("recent user", context)
        self.assertIn("recent answer", context)

    def test_payload_uses_four_turns_and_zero_temperature(self):
        payload = prepare_payload("А это?", empty_state(), [], "glm-4.7-flash")
        self.assertEqual(payload["temperature"], 0)
        self.assertIn("А это?", payload["messages"][-1]["content"])
