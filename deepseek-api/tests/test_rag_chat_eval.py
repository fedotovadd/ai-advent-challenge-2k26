import json
from pathlib import Path
import tempfile
import unittest

from deepseek_app.rag.rag_chat_eval import run_scenarios, summarize, render_report
from deepseek_app.rag.rag_chat_store import ChatStore


class EvaluationTests(unittest.TestCase):
    def test_report_shows_search_sources_state_and_manual_review(self):
        data = {"summary": {"attempted_count": 1, "retrieval_count": 1,
                            "answered_count": 1, "refusal_count": 0, "error_count": 0},
                "turns": [{"scenario": "A", "turn": 1, "status": "answered",
                           "search_query": "query", "sources": [{"source": "book"}],
                           "citations": [{"quote": "quote"}],
                           "state": {"goal": {"text": "goal", "turn": 1},
                                     "clarifications": [], "constraints": [], "terms": []}}],
                "failed_attempts": [{"scenario": "A", "turn": 1,
                                     "error": "bad quote"}]}
        report = render_report(data, {("A", 1): ("да", "Цитата подтверждает тезис")})
        self.assertIn("1/1", report)
        self.assertIn("book", report)
        self.assertIn("goal (№1)", report)
        self.assertIn("Цитата подтверждает тезис", report)
        self.assertIn("bad quote", report)

    def test_counts_and_restart(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        scenarios = [{"id": "A", "turns": ["one", "two"], "restart_after": 1,
                      "expected_final_state": {"goal": "goal"}},
                     {"id": "B", "turns": ["three"], "restart_after": None,
                      "expected_final_state": {"goal": "goal"}}]
        inp = root / "scenarios.json"
        inp.write_text(json.dumps(scenarios))
        seen = []
        def runner(store, name, message, **kwargs):
            seen.append((name, message, id(store), kwargs["model"]))
            store.load_session(name)
            result = {"status": "no_evidence", "answer": "Не знаю", "sources": [],
                      "citations": [], "selected": [], "search_query": message,
                      "reason": "below_threshold", "index": {"path": "index.db"}}
            store.commit_turn(name, 0 if message in ("one", "three") else 1,
                              message, result, message, {"goal": None,
                              "clarifications": [], "constraints": [], "terms": []})
            return result
        data = run_scenarios(inp, root / "result.json", root / "sessions.db", runner=runner)
        self.assertEqual(len(data["turns"]), 3)
        self.assertEqual(summarize(data)["retrieval_count"], 3)
        self.assertEqual(data["turns"][0]["reason"], "below_threshold")
        self.assertEqual(data["turns"][0]["index_metadata"], {"path": "index.db"})
        self.assertNotEqual(seen[0][2], seen[1][2])
        self.assertEqual(seen[0][3], "deepseek-v4-flash")
        self.assertTrue((root / "result.json").exists())

    def test_existing_result_is_not_overwritten(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        target = root / "result.json"
        target.write_text("old")
        inp = root / "scenarios.json"
        inp.write_text("[]")
        with self.assertRaises(FileExistsError):
            run_scenarios(inp, target, root / "sessions.db")
        self.assertEqual(target.read_text(), "old")

    def test_colliding_paths_rejected_before_creating_database(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        inp = root / "scenarios.json"
        inp.write_text("[]")
        target = root / "same.db"
        with self.assertRaises(ValueError):
            run_scenarios(inp, target, target)
        self.assertFalse(target.exists())
        with self.assertRaises(ValueError):
            run_scenarios(inp, root / "result.json", target, index_path=target)
        self.assertFalse(target.exists())

    def test_resume_reconciles_turn_committed_before_checkpoint(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        inp = root / "scenarios.json"
        inp.write_text(json.dumps([{"id": "A", "turns": ["one", "two"],
                                    "restart_after": None}]))
        target, db = root / "results.json", root / "sessions.db"
        def fail(*args, **kwargs):
            raise RuntimeError("simulated crash")
        run_scenarios(inp, target, db, runner=fail)
        partial = target.with_name(target.name + ".partial")
        checkpoint = json.loads(partial.read_text())
        checkpoint["status"] = "running"
        partial.write_text(json.dumps(checkpoint))
        store = ChatStore(db)
        state = {"goal": None, "clarifications": [], "constraints": [], "terms": []}
        result = {"status": "no_evidence", "answer": "Не знаю", "sources": [],
                  "citations": [], "selected": [], "search_query": "one"}
        store.commit_turn("A", 0, "one", result, "one", state)
        def next_turn(store, name, message, **kwargs):
            self.assertEqual(message, "two")
            store.commit_turn(name, 1, message, {**result, "search_query": "two"}, "two", state)
            return {**result, "search_query": "two"}
        data = run_scenarios(inp, target, db, runner=next_turn)
        self.assertEqual(data["status"], "complete")
        self.assertEqual([t["search_query"] for t in data["turns"]], ["one", "two"])
        self.assertTrue(data["recovery_events"])

    def test_error_preserves_partial_and_does_not_claim_complete(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        inp = root / "scenarios.json"
        inp.write_text(json.dumps([{"id": "A", "turns": ["one", "two"],
                                    "restart_after": None}]))
        target = root / "results.json"
        def fail(*args, **kwargs):
            raise RuntimeError("Ollama down")
        data = run_scenarios(inp, target, root / "sessions.db", runner=fail)
        self.assertEqual(data["status"], "incomplete")
        self.assertEqual(len(data["turns"]), 1)
        self.assertEqual(data["turns"][0]["error"], "RuntimeError: Ollama down")
        self.assertFalse(target.exists())
        self.assertTrue((root / "results.json.partial").exists())

    def test_resume_retries_failed_turn_and_preserves_attempt(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        inp = root / "scenarios.json"
        inp.write_text(json.dumps([{"id": "A", "turns": ["one", "two"],
                                    "restart_after": None}]))
        target, db = root / "results.json", root / "sessions.db"
        def runner(store, name, message, **kwargs):
            if message == "two" and not getattr(runner, "retry", False):
                raise RuntimeError("bad quote")
            version = store.load_session(name)["version"]
            result = {"status": "no_evidence", "answer": "Не знаю", "sources": [],
                      "citations": [], "selected": [], "search_query": message}
            store.commit_turn(name, version, message, result, message,
                              {"goal": None, "clarifications": [], "constraints": [], "terms": []})
            return result
        first = run_scenarios(inp, target, db, runner=runner)
        self.assertEqual(first["status"], "incomplete")
        runner.retry = True
        second = run_scenarios(inp, target, db, runner=runner)
        self.assertEqual(second["status"], "complete")
        self.assertEqual(second["summary"]["retrieval_count"], 2)
        self.assertEqual(second["failed_attempts"][0]["error"], "RuntimeError: bad quote")

    def test_resume_rejects_changed_provider_index_and_ollama_url(self):
        for changed in ({"model": "glm-4.7-flash"},
                        {"index_path": Path("different.db")},
                        {"ollama_url": "http://other:11434"}):
            with self.subTest(changed=changed):
                temp = tempfile.TemporaryDirectory()
                self.addCleanup(temp.cleanup)
                root = Path(temp.name)
                inp = root / "scenarios.json"
                inp.write_text(json.dumps([{"id": "A", "turns": ["one"],
                                            "restart_after": None}]))
                target, db = root / "results.json", root / "sessions.db"
                def fail(*args, **kwargs):
                    raise RuntimeError("bad quote")
                run_scenarios(inp, target, db, runner=fail, model="deepseek-v4-flash",
                              index_path=Path("index.db"), ollama_url="http://localhost:11434")
                settings = {"model": "deepseek-v4-flash", "index_path": Path("index.db"),
                            "ollama_url": "http://localhost:11434", **changed}
                with self.assertRaises(ValueError):
                    run_scenarios(inp, target, db, runner=fail, **settings)
