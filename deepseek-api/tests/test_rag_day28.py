import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from deepseek_app.rag import rag_day28


class Day28CompareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.index = self.root / "index.sqlite3"
        self.index.write_bytes(b"same index")
        self.questions = self.root / "questions.json"
        controls = [{"id": i, "question": f"Вопрос {i}?",
                     "expected_source": "one.xhtml", "expected_theses": ["Тезис"]}
                    for i in range(1, 11)]
        self.questions.write_text(json.dumps(controls), encoding="utf-8")
        self.results = self.root / "results.json"
        self.report = self.root / "comparison.md"
        self.builder = Mock(side_effect=lambda question, **kwargs: {
            "question": question, "search_query": question,
            "index": {"path": str(self.index), "embedding_model": "embeddinggemma",
                      "epub_sha256": "hash", "strategy": "structural"},
            "settings": {"top_k_before": 20, "top_k_after": 5,
                         "min_rerank_score": 0.5, "reranker_model": "model"},
            "selected": [{"chunk_id": "c1", "source": "book:one.xhtml",
                          "section": "Раздел", "title": "Книга", "text": "Тезис.",
                          "bi_rank": 1, "bi_score": 0.9,
                          "rerank_rank": 1, "rerank_score": 0.8}],
            "timing": {"search_seconds": 0.1, "rerank_seconds": 0.2}})

    def run_compare(self, **kwargs):
        return rag_day28.compare(
            questions_path=self.questions, results_path=self.results,
            report_path=self.report, index_path=self.index,
            context_builder=self.builder, **kwargs)

    def test_plain_answer_uses_saved_local_context(self):
        context = self.builder("Вопрос?")
        ask = Mock(return_value={"content": "Краткий ответ по книге.",
                                 "finish_reason": "stop", "completion_tokens": 12})
        result = rag_day28.generate_text_from_context(context, "qwen2.5:1.5b", ask=ask)
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["answer"], "Краткий ответ по книге.")
        self.assertEqual(result["sources"][0]["chunk_id"], "c1")
        self.assertIn("Тезис.", ask.call_args.args[0]["messages"][1]["content"])
        self.assertNotIn("response_format", ask.call_args.kwargs)

    def test_plain_answer_refuses_without_retrieved_context(self):
        context = self.builder("Вопрос?")
        context["selected"] = []
        ask = Mock(side_effect=AssertionError("model called"))
        result = rag_day28.generate_text_from_context(context, "qwen2.5:1.5b", ask=ask)
        self.assertEqual(result["status"], "no_evidence")
        ask.assert_not_called()

    def test_one_snapshot_three_local_passes_and_no_cloud_without_key(self):
        seen = []
        def ask(payload, **options):
            saved = json.loads(self.results.read_text(encoding="utf-8"))
            self.assertEqual(len(saved["questions"]), len(seen) // 3 + 1)
            self.assertEqual(saved["questions"][-1]["context"]["selected"][0]["text"], "Тезис.")
            seen.append((payload["model"], payload["messages"], options))
            return {"content": "Тезис.", "finish_reason": "stop"}
        data = self.run_compare(ask=ask, cloud_key=None)
        self.assertEqual(self.builder.call_count, 10)
        self.assertEqual(len(seen), 30)
        self.assertTrue(all(model == "qwen2.5:1.5b" for model, _, _ in seen))
        self.assertTrue(all("response_format" not in options for _, _, options in seen))
        self.assertTrue(all(len(q["runs"]["qwen2.5:1.5b"]) == 3 for q in data["questions"]))
        self.assertTrue(all(len(q["runs"]["deepseek-v4-flash"]) == 3 for q in data["questions"]))
        self.assertTrue(all(q["runs"]["deepseek-v4-flash"][0]["status"] == "not_checked_no_key"
                            for q in data["questions"]))
        self.assertEqual(data["questions"][0]["runs"]["qwen2.5:1.5b"][0]["status"], "answered")
        self.assertIn("не проверено: нет ключа", self.report.read_text(encoding="utf-8"))

    def test_cloud_receives_same_snapshot_and_resumes_after_error(self):
        calls = []
        def ask(payload, **options):
            calls.append((payload["model"], payload["messages"], options))
            if len(calls) == 2:
                raise RuntimeError("cloud down")
            return {"content": "Не знаю: в отрывках нет ответа."}
        data = self.run_compare(ask=ask, cloud_key="available")
        first = data["questions"][0]
        self.assertEqual(first["runs"]["deepseek-v4-flash"][0]["status"], "error")
        self.assertEqual(first["runs"]["qwen2.5:1.5b"][0]["status"], "no_evidence")
        self.assertEqual(calls[0][1], calls[1][1])
        self.assertEqual(len(calls), 60)
        self.builder.reset_mock()
        repeated = self.run_compare(ask=Mock(side_effect=AssertionError("unexpected")),
                                    cloud_key="available")
        self.builder.assert_not_called()
        self.assertEqual(repeated, data)

    def test_resume_rejects_changed_index(self):
        self.run_compare(ask=Mock(return_value={"content": "Не знаю."}), cloud_key=None)
        self.index.write_bytes(b"different index")
        with self.assertRaisesRegex(rag_day28.RagError, "индекс"):
            self.run_compare(ask=Mock(), cloud_key=None)

    def test_empty_answer_is_error_and_checkpointed(self):
        data = self.run_compare(ask=Mock(return_value={"content": ""}),
                                cloud_key=None)
        run = data["questions"][0]["runs"]["qwen2.5:1.5b"][0]
        self.assertEqual(run["status"], "error")
        self.assertEqual(run["raw_responses"], [""])
        self.assertEqual(json.loads(self.results.read_text())["status"], "complete")

    def test_length_error_preserves_finish_reason_in_checkpoint(self):
        ask = Mock(return_value={"content": "{", "finish_reason": "length",
                                 "completion_tokens": 512})
        data = self.run_compare(ask=ask, cloud_key=None)
        run = data["questions"][0]["runs"]["qwen2.5:1.5b"][0]
        self.assertEqual(run["status"], "error")
        self.assertEqual(run["finish_reasons"], ["length"])
        self.assertEqual(run["completion_tokens"], [512])
        self.assertEqual(len(ask.call_args_list), 30)

    def test_interruption_keeps_local_result_and_resumes_same_snapshot(self):
        calls = 0
        def interrupted(payload, **options):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt()
            return {"content": "Не знаю."}
        with self.assertRaises(KeyboardInterrupt):
            self.run_compare(ask=interrupted, cloud_key="available")
        saved = json.loads(self.results.read_text(encoding="utf-8"))
        self.assertEqual(saved["status"], "incomplete")
        self.assertEqual(len(saved["questions"]), 1)
        self.assertEqual(len(saved["questions"][0]["runs"]["qwen2.5:1.5b"]), 1)
        self.builder.reset_mock()
        self.run_compare(ask=Mock(return_value={"content": "Не знаю."}),
            cloud_key="available")
        self.assertEqual(self.builder.call_count, 9)

    def test_empty_context_skips_both_models(self):
        self.builder.side_effect = lambda question, **kwargs: {
            "question": question, "search_query": question,
            "index": {"path": str(self.index), "embedding_model": "embeddinggemma",
                      "epub_sha256": "hash", "strategy": "structural"},
            "settings": {"top_k_before": 20, "top_k_after": 5,
                         "min_rerank_score": 0.5, "reranker_model": "model"},
            "selected": [], "timing": {"search_seconds": 0.1, "rerank_seconds": 0.0}}
        ask = Mock(side_effect=AssertionError("model called"))
        data = self.run_compare(ask=ask, cloud_key="available")
        ask.assert_not_called()
        self.assertEqual(data["questions"][0]["runs"]["qwen2.5:1.5b"][0]["status"],
                         "no_evidence")
        self.assertEqual(data["questions"][0]["runs"]["deepseek-v4-flash"][0]["status"],
                         "no_evidence")
        self.assertEqual(data["summary"]["qwen2.5:1.5b"]["retrieval_empty"], 30)
        self.assertEqual(data["summary"]["qwen2.5:1.5b"]["refusals"], 0)

    def test_empty_context_records_refusal_for_cloud_without_key(self):
        self.builder.side_effect = lambda question, **kwargs: {
            "question": question, "search_query": question,
            "index": {"path": str(self.index), "embedding_model": "embeddinggemma",
                      "epub_sha256": "hash", "strategy": "structural"},
            "settings": {"top_k_before": 20, "top_k_after": 5,
                         "min_rerank_score": 0.5, "reranker_model": "model"},
            "selected": [], "timing": {"search_seconds": 0.1,
                                       "rerank_seconds": 0.0}}
        ask = Mock(side_effect=AssertionError("model called"))
        data = self.run_compare(ask=ask, cloud_key=None)
        ask.assert_not_called()
        self.assertEqual(data["questions"][0]["runs"]["deepseek-v4-flash"][0]["status"],
                         "no_evidence")

    def test_report_uses_saved_manual_assessment(self):
        data = self.run_compare(ask=Mock(return_value={"content": "Тезис."}),
                                cloud_key=None)
        run = data["questions"][0]["runs"]["qwen2.5:1.5b"][0]
        run["manual"] = {"completeness_0_1_2": 2, "semantic_support": "да",
                         "refusal_appropriate": None, "factual_errors": "нет",
                         "thesis_differences": "нет"}
        report = rag_day28._report(data)
        self.assertIn("полнота: 2", report)
        self.assertIn("смысловая опора: да", report)
        self.assertIn("фактические ошибки: нет", report)

    def test_report_command_rebuilds_from_reviewed_json(self):
        self.run_compare(ask=Mock(return_value={"content": "Тезис."}), cloud_key=None)
        data = json.loads(self.results.read_text(encoding="utf-8"))
        data["questions"][0]["runs"]["qwen2.5:1.5b"][0]["manual"]["semantic_support"] = "да"
        self.results.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        self.assertEqual(rag_day28.main(["report", "--results", str(self.results),
                                         "--report", str(self.report)]), 0)
        self.assertIn("смысловая опора: да", self.report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
