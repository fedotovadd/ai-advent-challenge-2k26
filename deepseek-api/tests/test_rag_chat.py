import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rag_chat
from rag_chat_store import ChatStore, StaleSessionError


class TurnTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = ChatStore(Path(self.temp.name) / "chat.db")
        self.prepare = Mock(return_value={"content": json.dumps({"search_query": "полный запрос",
            "changes": [{"op": "set_goal", "text": "изучить RAG", "evidence": "изучить RAG"}]})})
        self.answer = Mock(return_value={"status": "no_evidence", "answer": "Не знаю",
                                         "sources": [], "citations": [], "selected": []})

    def test_turn_uses_resolved_query_and_persists_refusal(self):
        result = rag_chat.run_turn(self.store, "one", "Хочу изучить RAG", model="glm-4.7-flash",
                                   prepare=self.prepare, answer=self.answer)
        self.assertEqual(self.answer.call_args.kwargs["search_query"], "полный запрос")
        self.assertIn("изучить RAG", self.answer.call_args.kwargs["conversation_context"])
        self.assertEqual(result["status"], "no_evidence")
        again = ChatStore(self.store.path).load_session("one")
        self.assertEqual(again["turns"][0]["search_query"], "полный запрос")
        self.assertEqual(again["state"]["goal"]["turn"], 1)

    def test_invalid_preparation_retries_once_without_commit(self):
        self.prepare.return_value = {"content": '{"search_query":"","changes":[]}'}
        with self.assertRaises(ValueError):
            rag_chat.run_turn(self.store, "one", "Вопрос", model="glm-4.7-flash",
                              prepare=self.prepare, answer=self.answer)
        self.assertEqual(self.prepare.call_count, 2)
        self.assertIn("Исправь", self.prepare.call_args.args[0]["messages"][-1]["content"])
        self.answer.assert_not_called()
        self.assertEqual(self.store.load_session("one")["version"], 0)

    def test_answer_error_leaves_state_and_history(self):
        self.answer.side_effect = RuntimeError("provider down")
        with self.assertRaisesRegex(RuntimeError, "provider down"):
            rag_chat.run_turn(self.store, "one", "Хочу изучить RAG", model="glm-4.7-flash",
                              prepare=self.prepare, answer=self.answer)
        self.assertEqual(self.store.load_session("one")["version"], 0)

    def test_second_turn_receives_previous_dialogue(self):
        rag_chat.run_turn(self.store, "one", "Хочу изучить RAG", model="glm-4.7-flash",
                          prepare=self.prepare, answer=self.answer)
        self.prepare.return_value = {"content": '{"search_query":"проверка RAG","changes":[]}'}
        rag_chat.run_turn(self.store, "one", "А как это проверить?", model="glm-4.7-flash",
                          prepare=self.prepare, answer=self.answer)
        self.assertIn("Хочу изучить RAG", self.prepare.call_args.args[0]["messages"][-1]["content"])
        self.assertEqual(self.answer.call_args.kwargs["search_query"], "проверка RAG")


class CliTests(unittest.TestCase):
    def test_state_and_history_read_errors_keep_loop(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        inputs = iter(["/state", "/history", "/exit"])
        out = io.StringIO()
        with patch("builtins.input", side_effect=lambda *_: next(inputs)), \
             patch.object(rag_chat.ChatStore, "load_session", side_effect=RuntimeError("db down")), \
             contextlib.redirect_stdout(out):
            self.assertEqual(rag_chat.main(["chat", "--session", "test", "--db",
                                            str(Path(temp.name) / "db")]), 0)
        self.assertEqual(out.getvalue().count("Ошибка: db down"), 2)

    def test_history_renders_persisted_answer_sources_and_citations(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        db = Path(temp.name) / "db"
        store = ChatStore(db)
        store.load_session("test")
        state = {"goal": {"text": "изучить TDD", "turn": 1},
                 "clarifications": [], "constraints": [], "terms": []}
        result = {"status": "answered", "answer": "Ответ [1]", "sources": [
            {"source": "book:one", "section": "Раздел", "chunk_id": "c1"}],
            "citations": [{"ref": 1, "source": "book:one", "section": "Раздел",
                           "chunk_id": "c1", "quote": "Цитата"}]}
        store.commit_turn("test", 0, "вопрос", result, "запрос", state)
        inputs = iter(["/state", "/history", "/exit"])
        out = io.StringIO()
        with patch("builtins.input", side_effect=lambda *_: next(inputs)), \
             contextlib.redirect_stdout(out):
            rag_chat.main(["chat", "--session", "test", "--db", str(db)])
        self.assertIn("изучить TDD (реплика №1)", out.getvalue())
        self.assertIn("book:one / Раздел (c1)", out.getvalue())
        self.assertIn("[1] Цитата", out.getvalue())

    def test_default_chat_model_is_deepseek(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        inputs = iter(["вопрос", "/exit"])
        answer = {"status": "no_evidence", "answer": "Не знаю", "sources": [], "citations": []}
        with patch("builtins.input", side_effect=lambda *_: next(inputs)), \
             patch.object(rag_chat, "run_turn", return_value=answer) as turn, \
             contextlib.redirect_stdout(io.StringIO()):
            rag_chat.main(["chat", "--session", "test", "--db", str(Path(temp.name) / "db")])
        self.assertEqual(turn.call_args.kwargs["model"], "deepseek-v4-flash")

    def test_commands_and_refusal_render(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        db = str(Path(temp.name) / "db.sqlite3")
        inputs = iter(["/state", "/history", "вопрос", "/history", "/exit"])
        answer = {"status": "no_evidence", "answer": "Не знаю", "sources": [], "citations": []}
        out = io.StringIO()
        with patch("builtins.input", side_effect=lambda *_: next(inputs)), \
             patch.object(rag_chat, "run_turn", return_value=answer), contextlib.redirect_stdout(out):
            self.assertEqual(rag_chat.main(["chat", "--session", "test", "--db", db,
                                            "--model", "glm-4.7-flash", "--index", "idx",
                                            "--ollama-url", "http://localhost:11434"]), 0)
        self.assertIn("Источники: нет", out.getvalue())
        self.assertIn("Цитаты: нет", out.getvalue())
        self.assertIn("Цель", out.getvalue())

    def test_invalid_command_and_provider_error_keep_loop(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        inputs = iter(["/unknown", "вопрос", "/exit"])
        out = io.StringIO()
        with patch("builtins.input", side_effect=lambda *_: next(inputs)), \
             patch.object(rag_chat, "run_turn", side_effect=RuntimeError("provider down")), \
             contextlib.redirect_stdout(out):
            rag_chat.main(["chat", "--session", "test", "--db", str(Path(temp.name) / "db")])
        self.assertIn("Неизвестная команда", out.getvalue())
        self.assertIn("provider down", out.getvalue())
