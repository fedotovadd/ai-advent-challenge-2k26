import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rag_day24
from rag_citations import NO_EVIDENCE_ANSWER


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


if __name__ == "__main__":
    unittest.main()
