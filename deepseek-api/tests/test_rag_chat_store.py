import tempfile
from pathlib import Path
import unittest

from deepseek_app.rag.rag_chat_store import ChatStore, StaleSessionError
from deepseek_app.rag.rag_chat_memory import empty_state


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "chat.sqlite3"

    def test_reopen_order_and_isolation(self):
        store = ChatStore(self.path)
        first = store.load_session("one")
        self.assertEqual((first["version"], first["state"], first["turns"]),
                         (0, empty_state(), []))
        result = {"status": "no_evidence", "answer": "Не знаю", "sources": [], "citations": []}
        state = empty_state()
        state["goal"] = {"text": "цель", "turn": 1}
        store.commit_turn("one", 0, "вопрос", result, "запрос", state)
        store.commit_turn("one", 1, "ещё", result, "ещё запрос", state)
        reopened = ChatStore(self.path).load_session("one")
        self.assertEqual(reopened["version"], 2)
        self.assertEqual([t["turn_no"] for t in reopened["turns"]], [1, 2])
        self.assertEqual(reopened["turns"][0]["search_query"], "запрос")
        self.assertEqual(reopened["state"], state)
        self.assertEqual(store.load_session("two")["turns"], [])

    def test_stale_write_preserves_current_turn(self):
        store = ChatStore(self.path)
        store.load_session("one")
        result = {"answer": "Ответ"}
        store.commit_turn("one", 0, "first", result, "query", empty_state())
        with self.assertRaises(StaleSessionError):
            store.commit_turn("one", 0, "stale", result, "query", empty_state())
        self.assertEqual(len(store.load_session("one")["turns"]), 1)

    def test_serialization_error_rolls_back(self):
        store = ChatStore(self.path)
        store.load_session("one")
        with self.assertRaises(TypeError):
            store.commit_turn("one", 0, "broken", {"bad": object()}, "query", empty_state())
        self.assertEqual(store.load_session("one")["version"], 0)
