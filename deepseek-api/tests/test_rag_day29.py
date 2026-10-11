import json
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock
from unittest.mock import patch

from deepseek_app.rag import rag_day29
from deepseek_app.rag.rag_agent import RagError
from deepseek_app.rag.day29_runtime import _is_local_model, _same_model, _ollama_rss_from_ps


class Day29Tests(unittest.TestCase):
    def test_ollama_latest_suffix_matches_allowlisted_variant(self):
        self.assertTrue(_is_local_model("qwen2.5-day29-ctx4096:latest"))
        self.assertTrue(_same_model("qwen2.5-day29-ctx4096:latest", "qwen2.5-day29-ctx4096"))
        self.assertFalse(_is_local_model("qwen2.5-day29-ctx16384:latest"))

    def test_rss_includes_llama_server_but_excludes_shell_commands(self):
        output = """46480 ollama serve
2035376 /opt/homebrew/lib/ollama/llama-server --model /tmp/blob
2464 /bin/zsh -c ps -axo rss=,command= | rg ollama
1664 rg ollama
"""
        self.assertEqual(_ollama_rss_from_ps(output), (46480 + 2035376) * 1024)

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.questions = self.root / "questions.json"
        self.questions.write_text(json.dumps([
            {"id": i, "question": f"Вопрос {i}?", "expected_source": "one.xhtml", "expected_theses": ["Тезис"]}
            for i in range(1, 11)]), encoding="utf-8")
        self.index = self.root / "index.db"
        self.index.write_bytes(b"index")
        self.results = self.root / "results.json"
        self.builder = Mock(side_effect=lambda question, **_: {
            "question": question, "selected": [{"chunk_id": "c1", "source": "book:one.xhtml",
                "section": "Раздел", "text": "Тезис."}], "index": {"path": str(self.index)},
            "settings": {"top_k_before": 20, "top_k_after": 5, "min_rerank_score": .5}})
        self.profiles = [
            {"id": "baseline", "model": "qwen2.5:1.5b", "temperature": 0,
             "max_tokens": 1200, "num_ctx": 32768, "prompt": "day28"},
            {"id": "short", "model": "qwen2.5:1.5b", "temperature": 0,
             "max_tokens": 512, "num_ctx": 32768, "prompt": "day29-short"},
        ]

    def test_parameters_snapshots_tokens_and_resume(self):
        seen = []
        def ask(payload, **options):
            seen.append((payload, options))
            return {"content": "Тезис.", "finish_reason": "stop", "prompt_tokens": 20,
                    "completion_tokens": 5, "total_tokens": 25}
        data = rag_day29.compare(self.questions, self.results, index_path=self.index,
            profiles=self.profiles, context_builder=self.builder, ask=ask,
            passes=3, runtime=rag_day29.NoopRuntime())
        self.assertEqual(self.builder.call_count, 10)
        self.assertEqual(len(seen), 60)
        self.assertEqual(seen[0][1]["max_tokens"], 1200)
        self.assertEqual(seen[3][1]["max_tokens"], 512)
        self.assertEqual(data["questions"][0]["runs"]["baseline"][0]["total_tokens"], 25)
        self.assertEqual(data["questions"][0]["snapshot_sha256"], data["questions"][0]["runs"]["short"][0]["snapshot_sha256"])
        self.assertEqual(data["questions"][0]["runs"]["baseline"][0]["snapshot_sha256"], data["questions"][0]["runs"]["short"][0]["snapshot_sha256"])
        self.assertNotEqual(data["questions"][0]["runs"]["baseline"][0]["messages_sha256"],
                            data["questions"][0]["runs"]["short"][0]["messages_sha256"])
        self.builder.reset_mock()
        rag_day29.compare(self.questions, self.results, index_path=self.index,
            profiles=self.profiles, context_builder=self.builder,
            ask=Mock(side_effect=AssertionError("called")), passes=3,
            runtime=rag_day29.NoopRuntime())
        self.builder.assert_not_called()

    def test_statuses_and_checkpoint_after_interrupt(self):
        responses = iter([
            {"content": "Не знаю: нет сведений.", "finish_reason": "stop"},
            {"content": "Незаконченный", "finish_reason": "length", "completion_tokens": 512},
            {"content": "", "finish_reason": "stop"},
        ])
        def ask(*_, **__):
            try:
                return next(responses)
            except StopIteration:
                raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            rag_day29.compare(self.questions, self.results, index_path=self.index,
                profiles=self.profiles, context_builder=self.builder, ask=ask,
                passes=3, runtime=rag_day29.NoopRuntime())
        saved = json.loads(self.results.read_text())
        runs = saved["questions"][0]["runs"]["baseline"]
        self.assertEqual([run["status"] for run in runs], ["model_refusal", "truncated", "error"])
        self.assertEqual(runs[1]["answer"], "Незаконченный")
        self.assertIsNone(runs[0]["prompt_tokens"])

    def test_resume_rejects_changed_profile_or_snapshot(self):
        with self.assertRaises(KeyboardInterrupt):
            rag_day29.compare(self.questions, self.results, index_path=self.index,
                profiles=self.profiles, context_builder=self.builder,
                ask=Mock(side_effect=KeyboardInterrupt()), passes=3,
                runtime=rag_day29.NoopRuntime())
        changed = [*self.profiles]
        changed[1] = {**changed[1], "temperature": .2}
        with self.assertRaisesRegex(ValueError, "астро"):
            rag_day29.compare(self.questions, self.results, index_path=self.index,
                profiles=changed, context_builder=self.builder,
                runtime=rag_day29.NoopRuntime())
        saved = json.loads(self.results.read_text())
        saved["questions"][0]["context"]["selected"][0]["text"] = "подмена"
        self.results.write_text(json.dumps(saved), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "сним"):
            rag_day29.compare(self.questions, self.results, index_path=self.index,
                profiles=self.profiles, context_builder=self.builder,
                runtime=rag_day29.NoopRuntime())

    def test_resume_rejects_changed_user_prompt_template(self):
        with self.assertRaises(KeyboardInterrupt):
            rag_day29.compare(self.questions, self.results, index_path=self.index,
                profiles=self.profiles, context_builder=self.builder,
                ask=Mock(side_effect=KeyboardInterrupt()), runtime=rag_day29.NoopRuntime())
        original = rag_day29.day28_messages
        with patch.object(rag_day29, "day28_messages", side_effect=lambda context:
                          [{**row, "content": row["content"] + " changed"}
                           if row["role"] == "user" else row for row in original(context)]):
            with self.assertRaisesRegex(ValueError, "Настройки"):
                rag_day29.compare(self.questions, self.results, index_path=self.index,
                    profiles=self.profiles, context_builder=self.builder,
                    runtime=rag_day29.NoopRuntime())

    def test_resume_rejects_changed_messages_for_saved_question(self):
        with self.assertRaises(KeyboardInterrupt):
            rag_day29.compare(self.questions, self.results, index_path=self.index,
                profiles=self.profiles, context_builder=self.builder,
                ask=Mock(side_effect=KeyboardInterrupt()), runtime=rag_day29.NoopRuntime())
        saved = json.loads(self.results.read_text())
        saved["questions"][0]["runs"]["baseline"].append({
            "prompt": "day28", "messages_sha256": "stale", "model_digest": None})
        self.results.write_text(json.dumps(saved), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Сообщени"):
            rag_day29.compare(self.questions, self.results, index_path=self.index,
                profiles=self.profiles, context_builder=self.builder,
                runtime=rag_day29.NoopRuntime())

    def test_resume_rejects_changed_model_digest(self):
        profile = {**self.profiles[0], "model_digest": "expected-digest"}
        class ChangedRuntime(rag_day29.NoopRuntime):
            def __init__(self):
                self.aborted = False
            def begin(self, profile):
                return {"model_digest": "actual-digest"}
            def abort(self):
                self.aborted = True
        runtime = ChangedRuntime()
        with self.assertRaisesRegex(ValueError, "digest"):
            rag_day29.compare(self.questions, self.results, index_path=self.index,
                profiles=[profile], context_builder=self.builder,
                ask=Mock(side_effect=AssertionError("generation must not run")),
                runtime=runtime)
        self.assertTrue(runtime.aborted)

    def test_resume_marks_interrupted_measurement_incomplete(self):
        class InterruptedRuntime(rag_day29.NoopRuntime):
            def end(self, profile):
                raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            rag_day29.compare(self.questions, self.results, index_path=self.index,
                profiles=[self.profiles[0]], context_builder=self.builder,
                ask=Mock(return_value={"content": "Тезис."}),
                runtime=InterruptedRuntime())
        resumed = rag_day29.compare(self.questions, self.results, index_path=self.index,
            profiles=[self.profiles[0]], context_builder=self.builder,
            ask=Mock(return_value={"content": "Тезис."}),
            runtime=rag_day29.NoopRuntime())
        self.assertTrue(resumed["questions"][0]["blocks"]["baseline"][0]["end"]["measurement_incomplete"])
        self.assertIsNone(rag_day29.summarize(resumed, "baseline")["peak_rss_bytes"])

    def test_empty_context_is_separate_and_not_sent_to_model(self):
        self.builder.side_effect = lambda question, **_: {"question": question, "selected": [], "index": {"path": str(self.index)}}
        data = rag_day29.compare(self.questions, self.results, index_path=self.index,
            profiles=self.profiles, context_builder=self.builder,
            ask=Mock(side_effect=AssertionError("called")), passes=3,
            runtime=rag_day29.NoopRuntime())
        self.assertEqual(data["questions"][0]["runs"]["baseline"][0]["status"], "retrieval_empty")
        self.assertIsNone(data["questions"][0]["runs"]["baseline"][0]["seconds"])

    def test_summary_excludes_retrieval_empty_and_preserves_missing_tokens(self):
        data = rag_day29.compare(self.questions, self.results, index_path=self.index,
            profiles=self.profiles, context_builder=self.builder,
            ask=Mock(return_value={"content": "Тезис.", "prompt_tokens": 20,
                                   "completion_tokens": 5}), passes=3,
            runtime=rag_day29.NoopRuntime())
        for run in data["questions"][0]["runs"]["baseline"]:
            run["manual"] = {"score": 2, "reason": "полно"}
        data["questions"][1]["runs"]["baseline"][0]["prompt_tokens"] = None
        data["questions"][9]["context"]["selected"] = []
        data["questions"][9]["runs"]["baseline"] = [
            {"status": "retrieval_empty", "seconds": None} for _ in range(3)]
        summary = rag_day29.summarize(data, "baseline")
        self.assertEqual(summary["quality"], 2)
        self.assertEqual(summary["quality_questions"], 1)
        self.assertEqual(summary["retrieval_empty"], 3)
        self.assertEqual(summary["prompt_tokens_missing"], 1)
        self.assertEqual(summary["tokens_per_score_2"], 25)

    def test_ask_uses_profile_and_reports_local_failure(self):
        profile_path = self.root / "profile.json"
        profile_path.write_text(json.dumps(self.profiles[1]), encoding="utf-8")
        call = Mock(return_value={"content": "Тезис.", "finish_reason": "stop"})
        run = rag_day29.ask_question("Вопрос?", profile_path=profile_path,
            index_path=self.index, context_builder=self.builder, ask=call)
        self.assertEqual(run["status"], "answered")
        self.assertEqual(call.call_args.kwargs["max_tokens"], 512)
        self.assertEqual(call.call_args.args[0]["temperature"], 0)
        with self.assertRaisesRegex(RuntimeError, "Локальная Ollama"):
            rag_day29.ask_question("Вопрос?", profile_path=profile_path,
                index_path=self.index, context_builder=self.builder,
                ask=Mock(side_effect=ConnectionError("unavailable")))

    def test_ask_checks_saved_digest_and_reports_stopped_ollama(self):
        profile_path = self.root / "profile.json"
        profile_path.write_text(json.dumps({**self.profiles[0], "model_digest": "expected"}))
        with patch.object(rag_day29, "installed_digest", return_value="changed"):
            with self.assertRaisesRegex(RuntimeError, "digest"):
                rag_day29.ask_question("Вопрос?", profile_path=profile_path,
                    index_path=self.index, context_builder=self.builder)
        with patch.object(rag_day29, "installed_digest", side_effect=ConnectionError("refused")):
            with self.assertRaisesRegex(RuntimeError, "Локальная Ollama недоступна"):
                rag_day29.ask_question("Вопрос?", profile_path=profile_path,
                    index_path=self.index, context_builder=self.builder)

    def test_report_distinguishes_missing_metrics_and_shows_all_variants(self):
        data = rag_day29.compare(self.questions, self.results, index_path=self.index,
            profiles=self.profiles, context_builder=self.builder,
            ask=Mock(return_value={"content": "Тезис.", "completion_tokens": 5}),
            passes=3, runtime=rag_day29.NoopRuntime())
        report = rag_day29.render_report(data, selected="short")
        self.assertIn("baseline", report)
        self.assertIn("short", report)
        self.assertIn("не измерен", report)
        self.assertIn("Без ручной оценки", report)
        other = json.loads(json.dumps(data))
        other["settings"]["profiles"][1]["id"] = "another-stage"
        other["settings"]["profiles"] = other["settings"]["profiles"][1:]
        for entry in other["questions"]:
            entry["runs"]["another-stage"] = entry["runs"].pop("short")
            entry["runs"].pop("baseline")
        self.assertIn("another-stage", rag_day29.render_report(data, selected="short", additional=[other]))

    def test_report_rejects_mismatched_paired_inputs(self):
        data = rag_day29.compare(self.questions, self.results, index_path=self.index,
            profiles=self.profiles, context_builder=self.builder,
            ask=Mock(return_value={"content": "Тезис."}),
            runtime=rag_day29.NoopRuntime())
        pair = json.loads(json.dumps(data))
        self.assertIn("Парное сравнение", rag_day29.render_report(data, selected="short", paired=pair))
        pair["questions"][0]["snapshot_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "парн"):
            rag_day29.render_report(data, selected="short", paired=pair)
        pair = json.loads(json.dumps(data))
        pair["settings"]["profiles"][1]["temperature"] = .2
        with self.assertRaisesRegex(ValueError, "парн"):
            rag_day29.render_report(data, selected="short", paired=pair)

    def test_second_stage_reuses_saved_snapshots_and_alternates_pair_order(self):
        first = rag_day29.compare(self.questions, self.results, index_path=self.index,
            profiles=self.profiles, context_builder=self.builder,
            ask=Mock(return_value={"content": "Тезис.", "prompt_tokens": 20}),
            passes=3, runtime=rag_day29.NoopRuntime())
        second_path = self.root / "paired.json"
        seen = []
        def ask(payload, **_):
            seen.append(payload["model"])
            return {"content": "Тезис.", "prompt_tokens": 20}
        pair = [self.profiles[0], {**self.profiles[1], "model": "qwen2.5-day29-ctx8192", "num_ctx": 8192, "prompt": "day28"}]
        second = rag_day29.compare(self.questions, second_path, index_path=self.index,
            profiles=pair, context_builder=Mock(side_effect=AssertionError("retrieval repeated")),
            ask=ask, passes=3, runtime=rag_day29.NoopRuntime(),
            snapshot_source=self.results, alternate=True)
        self.assertEqual(second["questions"][0]["snapshot_sha256"], first["questions"][0]["snapshot_sha256"])
        self.assertEqual(seen[:6], ["qwen2.5:1.5b"] * 3 + ["qwen2.5-day29-ctx8192"] * 3)
        self.assertEqual(seen[6:12], ["qwen2.5-day29-ctx8192"] * 3 + ["qwen2.5:1.5b"] * 3)

    def test_warmup_failure_is_checkpointed_as_three_errors(self):
        runtime = Mock()
        runtime.begin.side_effect = RuntimeError("missing model")
        data = rag_day29.compare(self.questions, self.results, index_path=self.index,
            profiles=self.profiles, context_builder=self.builder,
            ask=Mock(side_effect=AssertionError("generation must not run")),
            passes=3, runtime=runtime)
        runs = data["questions"][0]["runs"]["baseline"]
        self.assertEqual([run["status"] for run in runs], ["error"] * 3)
        self.assertTrue(all("missing model" in run["error"] for run in runs))

    def test_cli_reports_retrieval_connection_error_without_traceback(self):
        error = StringIO()
        with unittest.mock.patch.object(rag_day29, "ask_question", side_effect=RagError("Поиск: Ollama недоступна")), redirect_stderr(error):
            status = rag_day29.main(["ask", "Вопрос?"])
        self.assertEqual(status, 1)
        self.assertIn("Ollama недоступна", error.getvalue())
        self.assertNotIn("Traceback", error.getvalue())


if __name__ == "__main__":
    unittest.main()
