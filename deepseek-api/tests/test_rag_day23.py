import contextlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import rag_day23


class Day23QuestionTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"source": f"book.epub:{number}.xhtml", "title": "Книга",
             "section": f"Раздел {number}", "chunk_id": f"chunk-{number}",
             "text": "Текст " + str(number), "score": 1 - number / 100,
             "vector": "[1,2]"}
            for number in range(20)
        ]
        self.search = Mock(return_value=self.rows)
        self.ask = Mock(return_value={"content": "  Ответ [1]  "})
        self.scorer = Mock(side_effect=lambda pairs: [i / 20 for i in range(len(pairs))])
        self.rewriter = Mock(return_value="Переписанный запрос")
        self.loader = Mock(return_value=({"model": "embeddinggemma", "epub_sha256": "hash"}, []))
        self.embedder = object()
        self.embedder_factory = Mock(return_value=self.embedder)

    def run_question(self, mode, **kwargs):
        return rag_day23.answer_question_day23(
            " Исходный вопрос? ", mode, index_path=Path("index.db"),
            search=self.search, index_loader=self.loader, embedder_factory=self.embedder_factory,
            scorer=self.scorer, rewriter=self.rewriter, ask=self.ask, **kwargs)

    def test_baseline_uses_one_original_search_and_cosine_top5(self):
        result = self.run_question("baseline")
        self.search.assert_called_once_with(Path("index.db"), "Исходный вопрос?",
                                            "structural", self.embedder, top_k=20)
        self.scorer.assert_not_called()
        self.rewriter.assert_not_called()
        self.assertEqual([row["bi_rank"] for row in result["candidates"]], list(range(1, 21)))
        self.assertEqual([row["chunk_id"] for row in result["selected"]], [f"chunk-{i}" for i in range(5)])
        self.assertEqual(result["search_query"], "Исходный вопрос?")
        self.assertEqual(result["answer"], "Ответ [1]")
        self.assertEqual(result["status"], "answered")
        self.assertNotIn("vector", json.dumps(result))
        self.assertIn("Исходный вопрос?", self.ask.call_args.args[0]["messages"][1]["content"])

    def test_reranked_scores_entire_pool_and_retains_both_ranks(self):
        result = self.run_question("reranked", min_rerank_score=0)
        self.assertEqual(len(self.scorer.call_args.args[0]), 20)
        self.assertEqual(self.scorer.call_args.args[0][0][0], "Исходный вопрос?")
        self.assertEqual(len(result["candidates"]), 20)
        self.assertEqual(result["candidates"][19]["rerank_rank"], 1)
        self.assertEqual(result["candidates"][19]["rerank_score"], 0.95)
        self.assertEqual(result["candidates"][19]["bi_rank"], 20)
        self.assertEqual(result["candidates"][19]["bi_score"], 0.81)
        self.assertEqual([row["chunk_id"] for row in result["selected"]],
                         [f"chunk-{i}" for i in range(19, 14, -1)])
        self.assertIn("Исходный вопрос?", self.ask.call_args.args[0]["messages"][1]["content"])

    def test_rewritten_reranks_fixed_pool_but_answers_original_question(self):
        result = self.run_question("rewritten", min_rerank_score=0)
        self.rewriter.assert_called_once()
        self.assertEqual(self.rewriter.call_args.args[0], "Исходный вопрос?")
        self.assertEqual([row["chunk_id"] for row in self.rewriter.call_args.args[1]],
                         [row["chunk_id"] for row in self.rows])
        self.search.assert_called_once()
        self.assertEqual(self.scorer.call_args.args[0][0][0], "Переписанный запрос")
        self.assertEqual(result["search_query"], "Переписанный запрос")
        self.assertIn("Исходный вопрос?", self.ask.call_args.args[0]["messages"][1]["content"])
        self.assertNotIn("Переписанный запрос", self.ask.call_args.args[0]["messages"][1]["content"])

    def test_rewrite_disables_thinking_only_for_rewrite_provider_call(self):
        def rewrite(question, candidates, *, model, ask):
            response = ask({"model": model, "messages": []}, max_tokens=120)
            return response["content"]

        self.ask.side_effect = [
            {"content": "Переписанный запрос"}, {"content": "Ответ [1]"}]
        self.rewriter = rewrite
        self.run_question("rewritten")
        self.assertEqual(self.ask.call_args_list[0].kwargs, {
            "max_tokens": 120, "extra_body": {"thinking": {"type": "disabled"}}})
        self.assertEqual(self.ask.call_args_list[1].kwargs, {})

    def test_empty_search_and_filtered_context_skip_answer_provider(self):
        self.search.return_value = []
        for mode in ("baseline", "reranked", "rewritten"):
            with self.subTest(mode=mode):
                result = self.run_question(mode)
                self.assertEqual(result["answer"], rag_day23.NO_EVIDENCE_ANSWER)
                self.assertEqual(result["status"], "no_evidence")
                self.assertEqual(result["selected"], [])
        self.ask.assert_not_called()
        self.scorer.assert_not_called()
        self.rewriter.assert_not_called()
        self.search.return_value = self.rows
        self.scorer.side_effect = None
        self.scorer.return_value = [0.1] * 20
        result = self.run_question("reranked", min_rerank_score=0.5)
        self.assertEqual(result["selected"], [])
        self.assertEqual(len(result["candidates"]), 20)
        self.assertEqual(result["answer"], rag_day23.NO_EVIDENCE_ANSWER)
        self.ask.assert_not_called()

    def test_invalid_settings_rejected_before_io(self):
        cases = [
            {"top_k_before": 0}, {"top_k_after": 0}, {"top_k_before": 4, "top_k_after": 5},
            {"min_rerank_score": math.nan}, {"min_rerank_score": math.inf},
            {"min_rerank_score": -0.1}, {"min_rerank_score": 1.1},
            {"model": "unsupported"},
        ]
        for options in cases:
            with self.subTest(options=options), self.assertRaises(rag_day23.RagError):
                self.run_question("baseline", **options)
        with self.assertRaises(rag_day23.RagError):
            self.run_question("unknown")
        self.loader.assert_not_called()
        self.search.assert_not_called()

    def test_service_failures_become_user_facing_errors(self):
        scenarios = [
            ("loader", "Готовый индекс не найден", "baseline"),
            ("search", "Ollama недоступна", "baseline"),
            ("scorer", "reranker offline", "reranked"),
            ("rewriter", "rewrite offline", "rewritten"),
            ("ask", "provider offline", "baseline"),
        ]
        for attribute, message, mode in scenarios:
            with self.subTest(attribute=attribute):
                getattr(self, attribute).side_effect = RuntimeError(message)
                with self.assertRaisesRegex(rag_day23.RagError, message):
                    self.run_question(mode)
                getattr(self, attribute).side_effect = None

    def test_rewritten_reranker_failure_is_identified_after_successful_rewrite(self):
        self.rewriter.return_value = "Исходный вопрос?"
        self.scorer.side_effect = RuntimeError("model unavailable")
        with self.assertRaisesRegex(rag_day23.RagError, "^Reranker: model unavailable"):
            self.run_question("rewritten")


class Day23CliTests(unittest.TestCase):
    def test_ask_displays_queries_and_top20_top5_scores(self):
        rows = [{"chunk_id": "a", "source": "book:a", "section": "A",
                 "bi_score": 0.9, "bi_rank": 1, "rerank_score": 0.8, "rerank_rank": 2}]
        result = {"question": "Исходный?", "search_query": "Новый?",
                  "answer": "Ответ", "candidates": rows, "selected": rows,
                  "settings": {"top_k_before": 20, "top_k_after": 5}}
        out = io.StringIO()
        with patch.object(rag_day23, "answer_question_day23", return_value=result) as answer, \
             contextlib.redirect_stdout(out):
            status = rag_day23.main(["--index", "index.db", "--ollama-url", "http://ollama",
                                     "ask", "Исходный?", "--mode", "rewritten"])
        self.assertEqual(status, 0)
        self.assertEqual(answer.call_args.kwargs["index_path"], Path("index.db"))
        self.assertEqual(answer.call_args.kwargs["ollama_url"], "http://ollama")
        for text in ("Исходный?", "Новый?", "Top-20", "Top-5", "0.9", "0.8", "Ответ"):
            self.assertIn(text, out.getvalue())

    def test_ask_headings_use_configured_candidate_and_context_limits(self):
        result = {"question": "Вопрос?", "search_query": "Вопрос?", "answer": "Ответ",
                  "candidates": [], "selected": [],
                  "settings": {"top_k_before": 12, "top_k_after": 3}}
        out = io.StringIO()
        with patch.object(rag_day23, "answer_question_day23", return_value=result), \
             contextlib.redirect_stdout(out):
            status = rag_day23.main(["ask", "Вопрос?", "--top-k-before", "12",
                                     "--top-k-after", "3"])
        self.assertEqual(status, 0)
        self.assertIn("Кандидаты Top-12", out.getvalue())
        self.assertIn("Контекст Top-3", out.getvalue())


class Day23EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.questions_path = root / "questions.json"
        self.index_path = root / "index.db"
        self.results_path = root / "results.json"
        self.report_path = root / "report.md"
        self.partial_path = root / "results.json.partial"
        self.controls = [
            {"id": 1, "question": "Первый?", "expected_theses": ["а"], "expected_source": "one.xhtml"},
            {"id": 2, "question": "Второй?", "expected_theses": ["б"], "expected_source": "two.xhtml"},
        ]
        self.questions_path.write_text(json.dumps(self.controls), encoding="utf-8")
        self.index_path.write_bytes(b"index v1")
        loader = patch.object(rag_day23, "load_index", return_value=(
            {"model": "embeddinggemma", "epub_sha256": "epub-hash"}, []))
        loader.start()
        self.addCleanup(loader.stop)
        embedder = patch.object(rag_day23, "OllamaEmbedder", return_value=object())
        embedder.start()
        self.addCleanup(embedder.stop)
        search = patch.object(rag_day23, "search_index", return_value=[])
        search.start()
        self.addCleanup(search.stop)

    def run_evaluation(self, **options):
        return rag_day23.evaluate(
            self.questions_path, self.results_path, self.report_path,
            index_path=self.index_path, ollama_url="http://ollama", **options)

    def answer(self, question, mode, **kwargs):
        source = "one.xhtml" if question == "Первый?" else "two.xhtml"
        row = {"chunk_id": f"{source}-1", "source": f"book.epub/{source}",
               "title": "Книга", "section": "Раздел", "text": "Текст", "bi_score": 0.9,
               "bi_rank": 1}
        if mode != "baseline":
            row.update(rerank_score=0.8, rerank_rank=1)
        return {"question": question, "mode": mode,
                "search_query": "Новый запрос" if mode == "rewritten" else question,
                "answer": f"Ответ {mode}", "status": "answered", "model": kwargs["model"],
                "temperature": 0, "index": {"embedding_model": "embeddinggemma"},
                "settings": {"top_k_before": kwargs["top_k_before"],
                             "top_k_after": kwargs["top_k_after"],
                             "min_rerank_score": kwargs["min_rerank_score"],
                             "reranker_model": kwargs["reranker_model"]},
                "candidates": [row], "selected": [row]}

    def test_evaluate_order_shared_settings_scores_hits_and_report(self):
        with patch.object(rag_day23, "answer_question_day23", side_effect=self.answer) as answer, \
             patch.object(rag_day23, "CrossEncoderReranker") as scorer_type:
            data = self.run_evaluation(model="deepseek-v4-flash", top_k_before=12,
                                       top_k_after=3, min_rerank_score=0.7,
                                       reranker_model="local-reranker")
        self.assertEqual([(call.args[0], call.args[1]) for call in answer.call_args_list],
                         [(q["question"], mode) for q in self.controls
                          for mode in rag_day23.MODES])
        self.assertEqual({call.kwargs["model"] for call in answer.call_args_list},
                         {"deepseek-v4-flash"})
        self.assertEqual({call.kwargs["top_k_before"] for call in answer.call_args_list}, {12})
        self.assertEqual({call.kwargs["top_k_after"] for call in answer.call_args_list}, {3})
        self.assertEqual({call.kwargs["min_rerank_score"] for call in answer.call_args_list}, {0.7})
        self.assertEqual({call.kwargs["reranker_model"] for call in answer.call_args_list},
                         {"local-reranker"})
        scorer_type.assert_called_once_with("local-reranker")
        self.assertIs(answer.call_args_list[1].kwargs["scorer"], scorer_type.return_value)
        self.assertIs(answer.call_args_list[2].kwargs["scorer"], scorer_type.return_value)
        self.assertEqual([item["control"]["id"] for item in data["questions"]], [1, 2])
        self.assertTrue(all(item["expected_source_hit"][mode] for item in data["questions"]
                            for mode in rag_day23.MODES))
        saved = json.loads(self.results_path.read_text(encoding="utf-8"))
        self.assertEqual(saved, data)
        self.assertEqual(saved["questions"][0]["reranked"]["candidates"][0]["rerank_score"], 0.8)
        report = self.report_path.read_text(encoding="utf-8")
        for text in ("one.xhtml-1", "cosine=0.9000", "rerank=0.8000", "Новый запрос",
                     "0–2", "да/частично/нет", "улучшения", "ухудшения", "равные",
                     "local-reranker", "0.7", "Top-12", "Top-3", "512"):
            self.assertIn(text, report)
        self.assertFalse(self.partial_path.exists())

    def test_evaluate_uses_one_fixed_first_stage_pool_per_question(self):
        raw_rows = [
            {"source": "book.epub/one.xhtml", "title": "Книга", "section": "Раздел",
             "chunk_id": "chunk-a", "text": "Текст A", "score": 0.91, "vector": "[1]"},
            {"source": "book.epub/two.xhtml", "title": "Книга", "section": "Раздел",
             "chunk_id": "chunk-b", "text": "Текст B", "score": 0.82, "vector": "[2]"},
        ]
        search = Mock(side_effect=lambda *args, **kwargs: [dict(row) for row in raw_rows])
        scorer = Mock(return_value=[0.2, 0.9])
        original_answer = rag_day23.answer_question_day23
        def offline_answer(question, mode, **kwargs):
            return original_answer(
                question, mode, search=search,
                index_loader=Mock(return_value=(
                    {"model": "embeddinggemma", "epub_sha256": "epub-hash"}, [])),
                embedder_factory=Mock(return_value=object()),
                rewriter=Mock(return_value="Новый запрос"),
                ask=Mock(return_value={"content": "Ответ"}), **kwargs)
        with patch.object(rag_day23, "answer_question_day23", side_effect=offline_answer), \
             patch.object(rag_day23, "CrossEncoderReranker", return_value=scorer), \
             patch.object(rag_day23, "search_index", search):
            data = self.run_evaluation()
        self.assertEqual(search.call_count, len(self.controls))
        self.assertEqual([call.args[1] for call in search.call_args_list],
                         [control["question"] for control in self.controls])
        for item in data["questions"]:
            pools = [[(row["chunk_id"], row["bi_score"], row["bi_rank"])
                      for row in item[mode]["candidates"]] for mode in rag_day23.MODES]
            self.assertEqual(pools, [pools[0]] * len(rag_day23.MODES))
        self.assertNotIn("bi_score", raw_rows[0])
        self.assertNotIn("rerank_score", raw_rows[0])

    def test_checkpoint_only_after_complete_question_and_strict_resume(self):
        calls = 0
        def fail_second(question, mode, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 5:
                raise rag_day23.RagError("provider failure")
            return self.answer(question, mode, **kwargs)
        with patch.object(rag_day23, "answer_question_day23", side_effect=fail_second), \
             patch.object(rag_day23, "CrossEncoderReranker"):
            with self.assertRaisesRegex(rag_day23.RagError, "provider failure"):
                self.run_evaluation()
        saved = json.loads(self.partial_path.read_text(encoding="utf-8"))
        self.assertEqual(len(saved["questions"]), 1)
        self.assertFalse(self.results_path.exists())
        self.assertFalse(self.report_path.exists())
        with patch.object(rag_day23, "answer_question_day23", side_effect=AssertionError("called")):
            for options in ({"model": "deepseek-v4-flash"}, {"top_k_after": 4},
                            {"min_rerank_score": 0.6}, {"reranker_model": "other"},
                            {"ollama_url": "different"}):
                with self.subTest(options=options), self.assertRaises(rag_day23.RagError):
                    if "ollama_url" in options:
                        rag_day23.evaluate(self.questions_path, self.results_path, self.report_path,
                                           index_path=self.index_path, **options)
                    else:
                        self.run_evaluation(**options)
        self.index_path.write_bytes(b"index v2")
        with self.assertRaises(rag_day23.RagError):
            self.run_evaluation()
        self.index_path.write_bytes(b"index v1")
        self.questions_path.write_text(json.dumps(self.controls + [self.controls[0]]), encoding="utf-8")
        with self.assertRaises(rag_day23.RagError):
            self.run_evaluation()
        self.questions_path.write_text(json.dumps(self.controls), encoding="utf-8")
        with patch.object(rag_day23, "answer_question_day23", side_effect=self.answer) as answer, \
             patch.object(rag_day23, "CrossEncoderReranker"):
            self.run_evaluation()
        self.assertEqual(answer.call_count, 3)

    def test_resume_rejects_changed_embedding_model_and_report_is_not_touched(self):
        def fail_second(question, mode, **kwargs):
            if question == "Второй?":
                raise rag_day23.RagError("provider failure")
            return self.answer(question, mode, **kwargs)
        with patch.object(rag_day23, "answer_question_day23", side_effect=fail_second), \
             patch.object(rag_day23, "CrossEncoderReranker"):
            with self.assertRaises(rag_day23.RagError):
                self.run_evaluation()
        with patch.object(rag_day23, "load_index", return_value=(
                {"model": "other-embedding", "epub_sha256": "epub-hash"}, [])), \
             patch.object(rag_day23, "answer_question_day23", side_effect=AssertionError("called")):
            with self.assertRaisesRegex(rag_day23.RagError, "изменились"):
                self.run_evaluation()
        self.assertFalse(self.report_path.exists())

    def test_corrupt_checkpoint_item_raises_rag_error(self):
        def fail_second(question, mode, **kwargs):
            if question == "Второй?":
                raise rag_day23.RagError("provider failure")
            return self.answer(question, mode, **kwargs)
        with patch.object(rag_day23, "answer_question_day23", side_effect=fail_second), \
             patch.object(rag_day23, "CrossEncoderReranker"):
            with self.assertRaises(rag_day23.RagError):
                self.run_evaluation()
        partial = json.loads(self.partial_path.read_text(encoding="utf-8"))
        partial["questions"][0] = ["invalid"]
        self.partial_path.write_text(json.dumps(partial), encoding="utf-8")
        with self.assertRaisesRegex(rag_day23.RagError, "не соответствует"):
            self.run_evaluation()

    def test_nested_corrupt_checkpoint_is_rejected_before_resuming_or_finalizing(self):
        def fail_second(question, mode, **kwargs):
            if question == "Второй?":
                raise rag_day23.RagError("provider failure")
            return self.answer(question, mode, **kwargs)
        with patch.object(rag_day23, "answer_question_day23", side_effect=fail_second), \
             patch.object(rag_day23, "CrossEncoderReranker"):
            with self.assertRaises(rag_day23.RagError):
                self.run_evaluation()
        partial = json.loads(self.partial_path.read_text(encoding="utf-8"))
        partial["questions"][0]["reranked"]["selected"] = [{"source": "book.epub/one.xhtml"}]
        self.partial_path.write_text(json.dumps(partial), encoding="utf-8")
        with patch.object(rag_day23, "answer_question_day23", side_effect=AssertionError("called")):
            with self.assertRaisesRegex(rag_day23.RagError, "прогресс|поврежд"):
                self.run_evaluation()
        self.assertFalse(self.results_path.exists())
        self.assertFalse(self.report_path.exists())

    def test_invalid_control_report_fields_fail_before_model_calls(self):
        invalid = [
            [{key: value for key, value in self.controls[0].items() if key != "id"}],
            [{**self.controls[0], "expected_theses": ["valid", 42]}],
        ]
        for controls in invalid:
            self.questions_path.write_text(json.dumps(controls), encoding="utf-8")
            with self.subTest(controls=controls), \
                 patch.object(rag_day23, "answer_question_day23", side_effect=AssertionError("called")):
                with self.assertRaisesRegex(rag_day23.RagError, "вопрос"):
                    self.run_evaluation()
            self.assertFalse(self.results_path.exists())
            self.assertFalse(self.report_path.exists())
            self.assertFalse(self.partial_path.exists())

    def test_report_render_error_does_not_publish_final_json(self):
        with patch.object(rag_day23, "answer_question_day23", side_effect=self.answer), \
             patch.object(rag_day23, "CrossEncoderReranker"), \
             patch.object(rag_day23, "_report", side_effect=KeyError("broken report")):
            with self.assertRaisesRegex(rag_day23.RagError, "отчёт"):
                self.run_evaluation()
        self.assertFalse(self.results_path.exists())
        self.assertFalse(self.report_path.exists())
        self.assertEqual(len(json.loads(self.partial_path.read_text(encoding="utf-8"))["questions"]), 2)

    def test_expected_chapter_hit_requires_selected_source(self):
        def wrong_source(question, mode, **kwargs):
            result = self.answer(question, mode, **kwargs)
            result["selected"] = [{**result["selected"][0], "source": "book.epub/other.xhtml"}]
            return result
        with patch.object(rag_day23, "answer_question_day23", side_effect=wrong_source), \
             patch.object(rag_day23, "CrossEncoderReranker"):
            data = self.run_evaluation()
        self.assertFalse(any(item["expected_source_hit"][mode] for item in data["questions"]
                             for mode in rag_day23.MODES))

    def test_cli_evaluate_passes_settings_and_paths(self):
        out = io.StringIO()
        with patch.object(rag_day23, "evaluate", return_value={"questions": [1]}) as evaluate, \
             contextlib.redirect_stdout(out):
            status = rag_day23.main([
                "--index", "index.db", "--ollama-url", "http://ollama",
                "evaluate", "--questions", "q.json", "--results", "r.json",
                "--report", "r.md", "--model", "deepseek-v4-flash",
                "--top-k-before", "12", "--top-k-after", "3",
                "--min-rerank-score", "0.7", "--reranker-model", "local-reranker",
                "--overwrite"])
        self.assertEqual(status, 0)
        self.assertEqual(evaluate.call_args.args, (Path("q.json"), Path("r.json"), Path("r.md")))
        self.assertEqual(evaluate.call_args.kwargs["index_path"], Path("index.db"))
        self.assertEqual(evaluate.call_args.kwargs["ollama_url"], "http://ollama")
        self.assertEqual(evaluate.call_args.kwargs["model"], "deepseek-v4-flash")
        self.assertEqual(evaluate.call_args.kwargs["top_k_before"], 12)
        self.assertEqual(evaluate.call_args.kwargs["top_k_after"], 3)
        self.assertEqual(evaluate.call_args.kwargs["min_rerank_score"], 0.7)
        self.assertEqual(evaluate.call_args.kwargs["reranker_model"], "local-reranker")
        self.assertTrue(evaluate.call_args.kwargs["overwrite"])

    def test_recovers_json_then_report_and_preserves_edited_report(self):
        original_write = rag_day23._atomic_write
        def fail_report(path, content):
            if Path(path) == self.report_path:
                raise OSError("report write crashed")
            return original_write(path, content)
        with patch.object(rag_day23, "answer_question_day23", side_effect=self.answer), \
             patch.object(rag_day23, "CrossEncoderReranker"), \
             patch.object(rag_day23, "_atomic_write", side_effect=fail_report):
            with self.assertRaisesRegex(OSError, "report write crashed"):
                self.run_evaluation()
        self.assertTrue(self.results_path.exists())
        self.assertTrue(self.partial_path.exists())
        self.assertFalse(self.report_path.exists())
        original_json = self.results_path.read_text(encoding="utf-8")
        altered = json.loads(original_json)
        altered["questions_file"] = "different-questions.json"
        self.results_path.write_text(json.dumps(altered), encoding="utf-8")
        with patch.object(rag_day23, "answer_question_day23", side_effect=AssertionError("called")):
            with self.assertRaisesRegex(rag_day23.RagError, "не соответствует"):
                self.run_evaluation()
        self.results_path.write_text(original_json, encoding="utf-8")
        with patch.object(rag_day23, "answer_question_day23", side_effect=AssertionError("called")):
            self.run_evaluation()
        self.assertTrue(self.report_path.exists())
        self.assertFalse(self.partial_path.exists())
        self.report_path.write_text("Проверено вручную", encoding="utf-8")
        with self.assertRaises(rag_day23.RagError):
            self.run_evaluation()
        self.assertEqual(self.report_path.read_text(encoding="utf-8"), "Проверено вручную")

    def test_recovers_after_both_outputs_before_checkpoint_cleanup(self):
        with patch.object(rag_day23, "answer_question_day23", side_effect=self.answer), \
             patch.object(rag_day23, "CrossEncoderReranker"), \
             patch.object(rag_day23.Path, "unlink", side_effect=OSError("cleanup crashed")):
            with self.assertRaisesRegex(OSError, "cleanup crashed"):
                self.run_evaluation()
        self.report_path.write_text("Проверено вручную", encoding="utf-8")
        with patch.object(rag_day23, "answer_question_day23", side_effect=AssertionError("called")):
            self.run_evaluation()
        self.assertEqual(self.report_path.read_text(encoding="utf-8"), "Проверено вручную")
        self.assertFalse(self.partial_path.exists())

    def test_existing_results_protected_without_checkpoint(self):
        self.results_path.write_text("edited", encoding="utf-8")
        with patch.object(rag_day23, "answer_question_day23", side_effect=AssertionError("called")):
            with self.assertRaisesRegex(rag_day23.RagError, "существ"):
                self.run_evaluation()
        self.assertEqual(self.results_path.read_text(encoding="utf-8"), "edited")

    def test_overwrite_resume_keeps_completed_question_and_old_outputs_until_finish(self):
        self.results_path.write_text("old JSON", encoding="utf-8")
        self.report_path.write_text("manually edited old report", encoding="utf-8")
        def fail_second(question, mode, **kwargs):
            if question == "Второй?":
                raise rag_day23.RagError("provider failure")
            return self.answer(question, mode, **kwargs)
        with patch.object(rag_day23, "answer_question_day23", side_effect=fail_second), \
             patch.object(rag_day23, "CrossEncoderReranker"):
            with self.assertRaisesRegex(rag_day23.RagError, "provider failure"):
                self.run_evaluation(overwrite=True)
        self.assertEqual(len(json.loads(self.partial_path.read_text(encoding="utf-8"))["questions"]), 1)
        self.assertEqual(self.results_path.read_text(encoding="utf-8"), "old JSON")
        self.assertEqual(self.report_path.read_text(encoding="utf-8"), "manually edited old report")
        with patch.object(rag_day23, "answer_question_day23", side_effect=AssertionError("called")):
            with self.assertRaises(rag_day23.RagError):
                self.run_evaluation()
        with patch.object(rag_day23, "answer_question_day23", side_effect=self.answer) as answer, \
             patch.object(rag_day23, "CrossEncoderReranker"):
            data = self.run_evaluation(overwrite=True)
        self.assertEqual(answer.call_count, 3)
        self.assertTrue(all(call.args[0] == "Второй?" for call in answer.call_args_list))
        self.assertEqual(json.loads(self.results_path.read_text(encoding="utf-8")), data)
        self.assertIn("# День 23", self.report_path.read_text(encoding="utf-8"))
        self.assertFalse(self.partial_path.exists())

    def test_overwrite_recovers_after_json_write_before_replacing_old_report(self):
        self.results_path.write_text("old JSON", encoding="utf-8")
        self.report_path.write_text("old report", encoding="utf-8")
        original_write = rag_day23._atomic_write
        def fail_report(path, content):
            if Path(path) == self.report_path:
                raise OSError("report crash")
            return original_write(path, content)
        with patch.object(rag_day23, "answer_question_day23", side_effect=self.answer), \
             patch.object(rag_day23, "CrossEncoderReranker"), \
             patch.object(rag_day23, "_atomic_write", side_effect=fail_report):
            with self.assertRaisesRegex(OSError, "report crash"):
                self.run_evaluation(overwrite=True)
        self.assertEqual(self.report_path.read_text(encoding="utf-8"), "old report")
        with patch.object(rag_day23, "answer_question_day23", side_effect=AssertionError("called")):
            self.run_evaluation(overwrite=True)
        self.assertIn("# День 23", self.report_path.read_text(encoding="utf-8"))
        self.assertFalse(self.partial_path.exists())

    def test_overwrite_cleanup_recovery_preserves_edited_new_report(self):
        self.results_path.write_text("old JSON", encoding="utf-8")
        self.report_path.write_text("old report", encoding="utf-8")
        with patch.object(rag_day23, "answer_question_day23", side_effect=self.answer), \
             patch.object(rag_day23, "CrossEncoderReranker"), \
             patch.object(rag_day23.Path, "unlink", side_effect=OSError("cleanup crash")):
            with self.assertRaisesRegex(OSError, "cleanup crash"):
                self.run_evaluation(overwrite=True)
        self.report_path.write_text("manually reviewed new report", encoding="utf-8")
        with patch.object(rag_day23, "answer_question_day23", side_effect=AssertionError("called")):
            self.run_evaluation(overwrite=True)
        self.assertEqual(self.report_path.read_text(encoding="utf-8"),
                         "manually reviewed new report")
        self.assertFalse(self.partial_path.exists())

    def test_rejects_output_collisions_before_reading_inputs_or_writing(self):
        cases = [
            (self.results_path, self.results_path),
            (self.results_path, self.partial_path),
            (self.questions_path, self.report_path),
            (self.results_path, self.index_path),
            (self.questions_path.with_name("questions"), self.questions_path),
        ]
        original_questions = self.questions_path.read_bytes()
        original_index = self.index_path.read_bytes()
        for results_path, report_path in cases:
            with self.subTest(results=results_path, report=report_path), \
                 patch.object(rag_day23, "_read_questions", side_effect=AssertionError("read")):
                with self.assertRaisesRegex(rag_day23.RagError, "путь|совпад"):
                    rag_day23.evaluate(self.questions_path, results_path, report_path,
                                       index_path=self.index_path, overwrite=True)
        self.assertEqual(self.questions_path.read_bytes(), original_questions)
        self.assertEqual(self.index_path.read_bytes(), original_index)
        alias = self.results_path.with_name("result-alias.json")
        alias.symlink_to(self.results_path)
        with patch.object(rag_day23, "_read_questions", side_effect=AssertionError("read")):
            with self.assertRaisesRegex(rag_day23.RagError, "Пути"):
                rag_day23.evaluate(self.questions_path, self.results_path, alias,
                                   index_path=self.index_path, overwrite=True)
        partial_input = self.results_path.with_name("input.partial")
        partial_input.write_bytes(original_questions)
        with patch.object(rag_day23, "_read_questions", side_effect=AssertionError("read")):
            with self.assertRaisesRegex(rag_day23.RagError, "Пути"):
                rag_day23.evaluate(partial_input, partial_input.with_name("input"),
                                   self.report_path, index_path=self.index_path,
                                   overwrite=True)
        self.assertEqual(partial_input.read_bytes(), original_questions)


if __name__ == "__main__":
    unittest.main()
