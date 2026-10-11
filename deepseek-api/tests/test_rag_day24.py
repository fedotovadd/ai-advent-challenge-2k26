import json
import math
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch


from deepseek_app.rag import rag_day24
from deepseek_app.rag.rag_citations import NO_EVIDENCE_ANSWER


class Day24QuestionTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"source": f"book.epub:{i}.xhtml", "title": "Книга",
             "section": f"Раздел {i}", "chunk_id": f"c{i}",
             "text": f"Тезис номер {i}.", "score": 0.9 - i / 100,
             "vector": "[1,2]"} for i in range(20)]
        self.loader = Mock(return_value=({"model": "embeddinggemma", "epub_sha256": "hash"}, []))
        self.embedder = object()
        self.embedder_factory = Mock(return_value=self.embedder)
        self.search = Mock(return_value=self.rows)
        self.scorer = Mock(side_effect=lambda pairs: [i / 20 for i in range(len(pairs))])
        self.ask = Mock(return_value={"content": json.dumps({"status": "answered",
            "answer": "Подтверждено [1].", "citations": [
                {"chunk_id": "c19", "quote": "Тезис номер 19."}]}, ensure_ascii=False)})

    def run_question(self, **options):
        return rag_day24.answer_question_day24(
            " Исходный вопрос? ", index_path=Path("index.db"),
            index_loader=self.loader, embedder_factory=self.embedder_factory,
            search=self.search, scorer=self.scorer, ask=self.ask, **options)

    def test_search_rerank_threshold_and_verified_answer(self):
        result = self.run_question()
        self.search.assert_called_once_with(Path("index.db"), "Исходный вопрос?",
                                            "structural", self.embedder, top_k=20)
        self.assertEqual(len(self.scorer.call_args.args[0]), 20)
        self.assertTrue(all(pair[0] == "Исходный вопрос?" for pair in
                            self.scorer.call_args.args[0]))
        self.assertEqual([row["chunk_id"] for row in result["selected"]],
                         ["c19", "c18", "c17", "c16", "c15"])
        self.assertEqual(result["selected"][0]["rerank_score"], 0.95)
        self.assertEqual(result["selected"][0]["bi_rank"], 20)
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["sources"][0]["chunk_id"], "c19")
        self.assertNotIn("vector", json.dumps(result))
        self.assertEqual(result["settings"]["min_rerank_score"], 0.5)

    def test_resolved_query_and_dialogue_context(self):
        result = self.run_question(search_query="полный вопрос об агенте",
                                   conversation_context="Ранее пользователь спросил об агенте")
        self.assertEqual(self.search.call_args.args[1], "полный вопрос об агенте")
        self.assertTrue(all(pair[0] == "полный вопрос об агенте" for pair in
                            self.scorer.call_args.args[0]))
        prompt = self.ask.call_args.args[0]["messages"][1]["content"]
        self.assertIn("Исходный вопрос?", prompt)
        self.assertIn("Ранее пользователь", prompt)
        self.assertEqual(result["search_query"], "полный вопрос об агенте")

    def test_prior_citation_cannot_satisfy_current_search(self):
        self.ask.return_value = {"content": json.dumps({"status": "answered",
            "answer": "Ранее [1]", "citations": [{"chunk_id": "old", "quote": "old"}]})}
        with self.assertRaises(rag_day24.RagError):
            self.run_question(conversation_context="old: earlier citation")

    def test_equal_threshold_is_included(self):
        self.scorer.side_effect = None
        self.scorer.return_value = [0.5] + [0.1] * 19
        self.ask.return_value = {"content": json.dumps({"status": "answered",
            "answer": "Да [1].", "citations": [{"chunk_id": "c0", "quote": "Тезис номер 0."}]})}
        result = self.run_question()
        self.assertEqual([row["chunk_id"] for row in result["selected"]], ["c0"])

    def test_no_rows_skips_scorer_and_provider(self):
        self.search.return_value = []
        result = self.run_question()
        self.scorer.assert_not_called()
        self.ask.assert_not_called()
        self.assertEqual((result["status"], result["reason"]),
                         ("no_evidence", "below_threshold"))
        self.assertEqual(result["answer"], NO_EVIDENCE_ANSWER)
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["citations"], [])

    def test_below_threshold_skips_provider(self):
        self.scorer.side_effect = None
        self.scorer.return_value = [0.49] * 20
        result = self.run_question()
        self.ask.assert_not_called()
        self.assertEqual(result["selected"], [])
        self.assertEqual(result["reason"], "below_threshold")

    def test_model_abstention(self):
        self.ask.return_value = {"content": '{"status":"no_evidence","answer":"","citations":[]}'}
        result = self.run_question()
        self.assertEqual(result["reason"], "model_abstained")
        self.assertEqual(result["answer"], NO_EVIDENCE_ANSWER)

    def test_format_retry_once_with_same_excerpts(self):
        valid = self.ask.return_value
        self.ask.side_effect = [{"content": "bad json"}, valid]
        self.run_question()
        self.assertEqual(self.ask.call_count, 2)
        first, second = [call.args[0] for call in self.ask.call_args_list]
        self.assertEqual(first["messages"][1], second["messages"][1])
        self.assertIn("JSON", second["messages"][-1]["content"])
        self.assertEqual(second["messages"][-1]["content"],
                         "Исправь формат JSON и цитаты: Expecting value: line 1 column 1 (char 0). "
                         "Верни один JSON-объект по тем же фрагментам без дополнительного текста.")

    def test_chat_retry_requests_exact_quotes_or_abstention(self):
        valid = self.ask.return_value
        self.ask.side_effect = [{"content": "bad json"}, valid]
        self.run_question(search_query="resolved", conversation_context="past")
        repair = self.ask.call_args.args[0]["messages"][-1]["content"]
        self.assertIn("дословно", repair)
        self.assertIn("no_evidence", repair)

    def test_two_invalid_responses_raise(self):
        self.ask.return_value = {"content": "bad json"}
        with self.assertRaises(rag_day24.RagError):
            self.run_question()
        self.assertEqual(self.ask.call_count, 2)

    def test_provider_failure_is_not_retried(self):
        self.ask.side_effect = RuntimeError("provider down")
        with self.assertRaisesRegex(rag_day24.RagError, "provider down"):
            self.run_question()
        self.assertEqual(self.ask.call_count, 1)

    def test_index_search_and_reranker_errors(self):
        for dependency, error in ((self.loader, "index missing"),
                                  (self.search, "Ollama down"),
                                  (self.scorer, "reranker down")):
            with self.subTest(error=error):
                dependency.side_effect = RuntimeError(error)
                with self.assertRaisesRegex(rag_day24.RagError, error):
                    self.run_question()
                dependency.side_effect = None

    def test_invalid_options_rejected_before_io(self):
        for options in ({"model": "unknown"}, {"top_k_before": 0},
                        {"top_k_after": 21}, {"top_k_after": True},
                        {"min_rerank_score": math.nan},
                        {"min_rerank_score": 1.1},
                        {"min_rerank_score": -0.1}):
            with self.subTest(options=options), self.assertRaises(rag_day24.RagError):
                self.run_question(**options)
        self.loader.assert_not_called()

    def test_local_model_reuses_context_and_requests_json(self):
        context = rag_day24.prepare_context(
            "Вопрос?", index_path=Path("index.db"), model="qwen2.5:1.5b",
            index_loader=self.loader, embedder_factory=self.embedder_factory,
            search=self.search, scorer=self.scorer)
        first = rag_day24.generate_from_context(context, "qwen2.5:1.5b", ask=self.ask)
        second = rag_day24.generate_from_context(context, "qwen2.5:1.5b", ask=self.ask)
        self.assertEqual(first["status"], "answered")
        self.assertEqual(second["selected"], context["selected"])
        self.search.assert_called_once()
        fmt = self.ask.call_args.kwargs["response_format"]
        self.assertEqual(fmt["type"], "json_schema")
        self.assertEqual(fmt["json_schema"]["schema"]["properties"]["citations"]
                         ["items"]["properties"]["chunk_id"]["enum"],
                         ["c19", "c18", "c17", "c16", "c15"])
        self.assertEqual(self.ask.call_args.kwargs["max_tokens"], 512)
        self.assertEqual(self.ask.call_args.kwargs["timeout"], 30)

    def test_local_mode_rejects_external_embedding_url_before_io(self):
        with self.assertRaisesRegex(rag_day24.RagError, "loopback"):
            self.run_question(model="qwen2.5:1.5b", ollama_url="https://example.com")
        self.loader.assert_not_called()

    def test_local_mode_does_not_accept_cloud_suffix(self):
        with self.assertRaises(rag_day24.RagError):
            self.run_question(model="qwen2.5:1.5b:cloud")
        self.loader.assert_not_called()

    def test_local_invalid_response_records_error_after_one_repair(self):
        self.ask.return_value = {"content": "bad json"}
        with self.assertRaises(rag_day24.GenerationError) as caught:
            self.run_question(model="qwen2.5:1.5b")
        self.assertEqual(caught.exception.repair_count, 1)
        self.assertEqual(caught.exception.raw_responses, ["bad json", "bad json"])
        self.assertEqual(self.ask.call_count, 2)

    def test_local_length_finish_is_error_without_format_repair(self):
        self.ask.return_value = {"content": '{"status":"answered"',
                                 "finish_reason": "length", "completion_tokens": 512}
        with self.assertRaisesRegex(rag_day24.GenerationError, "лимит") as caught:
            self.run_question(model="qwen2.5:1.5b")
        self.assertEqual(caught.exception.raw_responses, ['{"status":"answered"'])
        self.assertEqual(caught.exception.finish_reasons, ["length"])
        self.assertEqual(caught.exception.completion_tokens, [512])
        self.ask.assert_called_once()

    def test_success_records_generation_finish_metadata(self):
        self.ask.return_value = {**self.ask.return_value,
                                 "finish_reason": "stop", "completion_tokens": 75}
        result = self.run_question(model="qwen2.5:1.5b")
        self.assertEqual(result["generation"]["finish_reasons"], ["stop"])
        self.assertEqual(result["generation"]["completion_tokens"], [75])

    def test_repair_repeats_citation_object_schema(self):
        self.ask.return_value = {"content": "bad json"}
        with self.assertRaises(rag_day24.GenerationError):
            self.run_question(model="qwen2.5:1.5b")
        repair = self.ask.call_args.args[0]["messages"][-1]["content"]
        self.assertIn("citations — массив объектов", repair)
        self.assertIn("400 символов", repair)
        self.assertNotIn("Ответ [1].", repair)

    def test_repair_explicitly_requests_missing_marker(self):
        bad = {"content": json.dumps({"status": "answered", "answer": "Тезис.",
            "citations": [{"chunk_id": "c19", "quote": "Тезис номер 19."}]})}
        self.ask.side_effect = [bad, self.ask.return_value]
        self.run_question(model="qwen2.5:1.5b")
        repair = self.ask.call_args.args[0]["messages"][-1]["content"]
        self.assertIn("Добавь маркер [1]", repair)


class Day24CliTests(unittest.TestCase):
    def test_ask_prints_answer_sources_and_quotes(self):
        result = {"answer": "Ответ [1].", "sources": [{"source": "book:one",
            "section": "Глава", "chunk_id": "c1"}], "citations": [{"ref": 1,
            "source": "book:one", "section": "Глава", "chunk_id": "c1", "quote": "Точно."}]}
        out = io.StringIO()
        with patch.object(rag_day24, "answer_question_day24", return_value=result), contextlib.redirect_stdout(out):
            code = rag_day24.main(["ask", "Вопрос?"])
        self.assertEqual(code, 0)
        self.assertIn("Ответ: Ответ [1].", out.getvalue())
        self.assertIn("book:one / Глава (c1)", out.getvalue())
        self.assertIn("[1] Точно.", out.getvalue())

    def test_refusal_and_error_output(self):
        out, err = io.StringIO(), io.StringIO()
        result = {"answer": NO_EVIDENCE_ANSWER, "sources": [], "citations": []}
        with patch.object(rag_day24, "answer_question_day24", return_value=result), contextlib.redirect_stdout(out):
            self.assertEqual(rag_day24.main(["ask", "Вопрос?"]), 0)
        self.assertIn("Источники: нет", out.getvalue())
        self.assertIn("Цитаты: нет", out.getvalue())
        with patch.object(rag_day24, "answer_question_day24", side_effect=rag_day24.RagError("ошибка")), contextlib.redirect_stderr(err):
            self.assertEqual(rag_day24.main(["ask", "Вопрос?"]), 1)
        self.assertIn("ошибка", err.getvalue())


class Day24EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.questions = self.root / "questions.json"
        self.index = self.root / "index.db"
        self.index.write_bytes(b"index")
        self.results = self.root / "results.json"
        self.report = self.root / "report.md"
        self.controls = [{"id": i, "question": f"Вопрос {i}?",
                          "expected_source": f"{i}.xhtml",
                          "expected_theses": [f"Тезис {i}"]} for i in range(1, 11)]
        self.questions.write_text(json.dumps(self.controls, ensure_ascii=False))
        self.calls = []

    def answerer(self, question, **options):
        self.calls.append((question, options))
        n = int(question.split()[1].rstrip("?"))
        selected = [{"chunk_id": f"c{n}", "source": f"book:{n}.xhtml",
                     "section": "Глава", "text": f"Тезис {n}",
                     "rerank_score": 0.6, "bi_score": 0.8}]
        if n == 3:
            return {"question": question, "status": "no_evidence",
                    "reason": "model_abstained", "answer": NO_EVIDENCE_ANSWER,
                    "sources": [], "citations": [], "selected": selected,
                    "model": options["model"], "settings": {"min_rerank_score": 0.5},
                    "index": {"path": str(self.index)}}
        return {"question": question, "status": "answered", "answer": "Да [1].",
                "sources": [{"source": selected[0]["source"], "section": "Глава", "chunk_id": f"c{n}"}],
                "citations": [{"ref": 1, "source": selected[0]["source"],
                               "section": "Глава", "chunk_id": f"c{n}", "quote": f"Тезис {n}"}],
                "selected": selected, "model": options["model"],
                "settings": {"min_rerank_score": 0.5},
                "index": {"path": str(self.index)}}

    def run_eval(self, **options):
        return rag_day24.evaluate(self.questions, self.results, self.report,
                                  index_path=self.index, answerer=self.answerer,
                                  model="glm-4.7-flash", **options)

    def test_ten_questions_checks_and_semantic_review_placeholder(self):
        data = self.run_eval()
        self.assertEqual(len(self.calls), 10)
        self.assertEqual(len(data["questions"]), 10)
        self.assertEqual(data["answered_count"], 9)
        self.assertEqual(data["no_evidence_count"], 1)
        self.assertEqual({call[1]["model"] for call in self.calls}, {"glm-4.7-flash"})
        self.assertTrue(data["questions"][0]["checks"]["quotes_verbatim"])
        self.assertTrue(data["questions"][0]["checks"]["selected_chunk_membership"])
        self.assertEqual(data["questions"][2]["result"]["status"], "no_evidence")
        self.assertIn("требует ручной сверки", self.report.read_text())
        self.assertEqual(self.report.read_text().count("| требует ручной сверки |"), 10)
        self.assertFalse((self.root / "results.json.partial").exists())

    def test_existing_outputs_and_colliding_paths_rejected(self):
        self.results.write_text("old")
        with self.assertRaises(rag_day24.RagError):
            self.run_eval()
        self.assertEqual(self.results.read_text(), "old")
        with self.assertRaises(rag_day24.RagError):
            rag_day24.evaluate(self.questions, self.questions, self.report,
                               index_path=self.index, answerer=self.answerer)

    def test_resume_after_failure_and_reject_changed_inputs(self):
        real_answerer = self.answerer
        def fail_second(question, **options):
            if question == "Вопрос 2?":
                raise rag_day24.RagError("provider down")
            return real_answerer(question, **options)
        self.answerer = fail_second
        with self.assertRaises(rag_day24.RagError):
            self.run_eval()
        partial = self.root / "results.json.partial"
        self.assertEqual(len(json.loads(partial.read_text())["questions"]), 1)
        self.answerer = real_answerer
        with self.assertRaises(rag_day24.RagError):
            self.run_eval(top_k_after=4)
        self.assertEqual(len(self.calls), 1)
        self.run_eval()
        self.assertEqual(len(self.calls), 10)
        self.assertFalse(partial.exists())

    def test_changed_questions_index_and_model_reject_checkpoint(self):
        original = self.answerer
        self.answerer = lambda question, **options: (
            original(question, **options) if question == "Вопрос 1?"
            else (_ for _ in ()).throw(rag_day24.RagError("stop")))
        with self.assertRaises(rag_day24.RagError):
            self.run_eval()
        self.answerer = original
        self.questions.write_text(json.dumps([{**self.controls[0], "question": "Другой?"},
                                              *self.controls[1:]], ensure_ascii=False))
        with self.assertRaises(rag_day24.RagError):
            self.run_eval()
        self.questions.write_text(json.dumps(self.controls, ensure_ascii=False))
        self.index.write_bytes(b"different index")
        with self.assertRaises(rag_day24.RagError):
            self.run_eval()
        self.index.write_bytes(b"index")
        with self.assertRaises(rag_day24.RagError):
            rag_day24.evaluate(self.questions, self.results, self.report,
                               index_path=self.index, answerer=self.answerer,
                               model="deepseek-v4-flash")
        self.assertEqual(len(self.calls), 1)

    def test_corrupt_completed_entry_cannot_be_skipped(self):
        original = self.answerer
        def fail_second(question, **options):
            if question == "Вопрос 2?":
                raise rag_day24.RagError("stop")
            return original(question, **options)
        self.answerer = fail_second
        with self.assertRaises(rag_day24.RagError):
            self.run_eval()
        partial_path = self.root / "results.json.partial"
        partial = json.loads(partial_path.read_text())
        partial["questions"][0]["result"]["answer"] = "Изменённый ответ без ссылки"
        partial_path.write_text(json.dumps(partial, ensure_ascii=False))
        self.answerer = Mock(side_effect=AssertionError("provider called"))
        with self.assertRaises(rag_day24.RagError):
            self.run_eval()
        self.answerer.assert_not_called()
        self.assertFalse(self.results.exists())

    def test_checkpoint_without_report_score_is_rejected(self):
        original = self.answerer
        def fail_second(question, **options):
            if question == "Вопрос 2?":
                raise rag_day24.RagError("stop")
            return original(question, **options)
        self.answerer = fail_second
        with self.assertRaises(rag_day24.RagError):
            self.run_eval()
        partial_path = self.root / "results.json.partial"
        partial = json.loads(partial_path.read_text())
        del partial["questions"][0]["result"]["selected"][0]["bi_score"]
        partial_path.write_text(json.dumps(partial, ensure_ascii=False))
        self.answerer = Mock(side_effect=AssertionError("provider called"))
        with self.assertRaises(rag_day24.RagError):
            self.run_eval()
        self.answerer.assert_not_called()
        self.assertFalse(self.results.exists())

    def test_overwrite_recovery_preserves_report_edited_after_checkpoint(self):
        self.report.write_text("prior report")
        self.answerer = lambda question, **options: (
            Day24EvaluationTests.answerer(self, question, **options)
            if question == "Вопрос 1?"
            else (_ for _ in ()).throw(rag_day24.RagError("stop")))
        with self.assertRaises(rag_day24.RagError):
            self.run_eval(overwrite=True)
        partial = json.loads((self.root / "results.json.partial").read_text())
        self.assertIsNotNone(partial["prior_report_sha256"])
        self.report.write_text("manual edits")
        self.answerer = lambda question, **options: Day24EvaluationTests.answerer(
            self, question, **options)
        with self.assertRaises(rag_day24.RagError):
            self.run_eval()
        self.assertEqual(self.report.read_text(), "manual edits")
        self.assertTrue((self.root / "results.json.partial").exists())
        self.answerer = Mock(side_effect=AssertionError("provider called"))
        recovered_report = self.root / "recovered.md"
        rag_day24.evaluate(self.questions, self.results, recovered_report,
                           index_path=self.index, answerer=self.answerer,
                           model="glm-4.7-flash")
        self.answerer.assert_not_called()
        self.assertTrue(recovered_report.exists())
        self.assertEqual(self.report.read_text(), "manual edits")
        self.assertFalse((self.root / "results.json.partial").exists())

    def test_completed_overwrite_checkpoint_replaces_old_json_without_provider(self):
        old = self.run_eval()
        def crash_after_tenth(done, total):
            if done == total:
                raise rag_day24.RagError("crash before final JSON")
        with self.assertRaises(rag_day24.RagError):
            self.run_eval(overwrite=True, progress=crash_after_tenth)
        partial = json.loads((self.root / "results.json.partial").read_text())
        self.assertEqual(len(partial["questions"]), 10)
        self.assertEqual(json.loads(self.results.read_text())["created_at"], old["created_at"])
        self.answerer = Mock(side_effect=AssertionError("provider called"))
        new = self.run_eval()
        self.answerer.assert_not_called()
        self.assertEqual(new["created_at"], partial["created_at"])
        self.assertFalse((self.root / "results.json.partial").exists())

    def test_final_json_recovers_missing_report_without_provider(self):
        self.run_eval()
        self.report.unlink()
        self.answerer = Mock(side_effect=AssertionError("provider called"))
        data = self.run_eval()
        self.assertEqual(data["status"], "complete")
        self.answerer.assert_not_called()
        self.assertTrue(self.report.exists())

    def test_manual_report_is_preserved_during_recovery(self):
        data = self.run_eval()
        data["status"] = "incomplete"
        (self.root / "results.json.partial").write_text(json.dumps(data, ensure_ascii=False))
        self.report.write_text("manual edits")
        self.answerer = Mock(side_effect=AssertionError("provider called"))
        with self.assertRaises(rag_day24.RagError):
            self.run_eval()
        self.assertEqual(self.report.read_text(), "manual edits")
        self.assertTrue((self.root / "results.json.partial").exists())


if __name__ == "__main__":
    unittest.main()
