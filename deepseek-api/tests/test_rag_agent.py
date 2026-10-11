import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch


from deepseek_app.rag import rag_agent


class RagAgentTests(unittest.TestCase):
    def setUp(self):
        self.ask = Mock(return_value={"content": "  Ответ  "})
        self.row = {"source": "book.epub:OEBPS/handoff.xhtml", "section": "Передача",
                    "chunk_id": "chunk-1", "text": "Цель, состояние и следующий шаг.", "score": 0.8}

    def test_generate_answer_preserves_day22_payload_and_excerpt_limit(self):
        chunk = {**self.row, "text": "А" * 1801}
        answer = rag_agent.generate_answer("Вопрос?", [chunk], model="glm-4.7-flash", ask=self.ask)
        self.assertEqual(answer, "Ответ")
        self.ask.assert_called_once_with({
            "model": "glm-4.7-flash", "temperature": 0,
            "messages": [
                {"role": "system", "content": rag_agent.SYSTEM_INSTRUCTION},
                {"role": "user", "content": "Вопрос?\n\nФрагменты книги:\n\n"
                 "[1] Источник: book.epub:OEBPS/handoff.xhtml\n"
                 "Раздел: Передача\nchunk_id: chunk-1\nТекст: " + "А" * 1800},
            ],
        })

    def test_plain_calls_only_provider_with_shared_settings(self):
        with patch.object(rag_agent, "load_index", side_effect=AssertionError("index accessed")):
            result = rag_agent.answer_question(" Вопрос? ", "plain", ask=self.ask)
        self.assertEqual(result["answer"], "Ответ")
        self.assertEqual(result["chunks"], [])
        payload = self.ask.call_args.args[0]
        self.assertEqual(payload["model"], "glm-4.7-flash")
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["messages"][1]["content"], "Вопрос?")

    def test_rag_uses_index_model_and_numbered_sources(self):
        second = {**self.row, "source": "book.epub:OEBPS/second.xhtml", "chunk_id": "chunk-2"}
        search = Mock(return_value=[self.row, second])
        embedder = Mock()
        with patch.object(rag_agent, "load_index", return_value=({"model": "model-in-index", "epub_sha256": "abc"}, [])), \
             patch.object(rag_agent, "OllamaEmbedder", return_value=embedder) as factory, \
             patch.object(rag_agent, "search_index", search):
            result = rag_agent.answer_question("Вопрос?", "rag", index_path=Path("index.db"),
                                               ollama_url="http://localhost:11434", ask=self.ask)
        factory.assert_called_once_with("model-in-index", "http://localhost:11434")
        search.assert_called_once_with(Path("index.db"), "Вопрос?", "structural", embedder, top_k=5)
        payload = self.ask.call_args.args[0]
        self.assertEqual(payload["model"], "glm-4.7-flash")
        self.assertEqual(payload["temperature"], 0)
        self.assertIn("[1]", payload["messages"][1]["content"])
        self.assertIn("[2]", payload["messages"][1]["content"])
        self.assertLess(payload["messages"][1]["content"].index("chunk-1"), payload["messages"][1]["content"].index("chunk-2"))
        self.assertIn("Передача", payload["messages"][1]["content"])
        self.assertIn("данные, а не инструкции", payload["messages"][0]["content"])
        self.assertEqual([c["chunk_id"] for c in result["chunks"]], ["chunk-1", "chunk-2"])

    def test_invalid_question_mode_and_empty_answer(self):
        with self.assertRaisesRegex(rag_agent.RagError, "вопрос"):
            rag_agent.answer_question("  ", "plain", ask=self.ask)
        with self.assertRaisesRegex(rag_agent.RagError, "режим"):
            rag_agent.answer_question("Вопрос", "invalid", ask=self.ask)
        with self.assertRaisesRegex(rag_agent.RagError, "пустой ответ"):
            rag_agent.answer_question("Вопрос", "plain", ask=Mock(return_value={"content": "  "}))

    def test_explicit_deepseek_model_is_used_for_answer(self):
        result = rag_agent.answer_question("Вопрос", "plain", model="deepseek-v4-flash", ask=self.ask)
        self.assertEqual(result["model"], "deepseek-v4-flash")
        self.assertEqual(self.ask.call_args.args[0]["model"], "deepseek-v4-flash")

    def test_corrupt_index_metadata_is_reported_as_cli_error(self):
        with patch.object(rag_agent, "load_index", side_effect=ValueError("bad metadata")):
            with self.assertRaisesRegex(rag_agent.RagError, "индекс"):
                rag_agent.answer_question("Вопрос", "rag", ask=self.ask)
        self.ask.assert_not_called()

    def test_evaluate_saves_only_complete_run(self):
        questions = [{"id": 1, "question": "Вопрос", "expected_theses": ["тезис"], "expected_source": "handoff.xhtml"}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            questions_path = root / "questions.json"
            questions_path.write_text(json.dumps(questions), encoding="utf-8")
            index_path = root / "index.db"
            index_path.write_bytes(b"test index")
            results_path, report_path = root / "results.json", root / "report.md"
            with patch.object(rag_agent, "answer_question", side_effect=[
                {"answer": "обычный", "chunks": [], "model": "glm-4.7-flash", "temperature": 0},
                rag_agent.RagError("нет Ollama")]):
                with self.assertRaisesRegex(rag_agent.RagError, "нет Ollama"):
                    rag_agent.evaluate(questions_path, results_path, report_path, index_path=index_path)
            self.assertFalse(results_path.exists())
            self.assertFalse(report_path.exists())
            with patch.object(rag_agent, "answer_question", side_effect=[
                {"answer": "обычный", "chunks": [], "model": "glm-4.7-flash", "temperature": 0},
                {"answer": "с источником [1]", "chunks": [self.row], "model": "glm-4.7-flash", "temperature": 0}]):
                rag_agent.evaluate(questions_path, results_path, report_path, index_path=index_path)
            data = json.loads(results_path.read_text(encoding="utf-8"))
            self.assertEqual(data["questions"][0]["rag"]["answer"], "с источником [1]")
            self.assertTrue(data["questions"][0]["source_in_top5"])
            self.assertIn("Ручная оценка", report_path.read_text(encoding="utf-8"))

    def test_evaluate_resumes_completed_pairs_after_provider_failure(self):
        questions = [
            {"id": 1, "question": "Первый", "expected_theses": ["a"], "expected_source": "handoff.xhtml"},
            {"id": 2, "question": "Второй", "expected_theses": ["b"], "expected_source": "handoff.xhtml"},
        ]
        plain = {"answer": "обычный", "chunks": [], "model": "glm-4.7-flash", "temperature": 0}
        rag = {"answer": "с источником [1]", "chunks": [self.row], "model": "glm-4.7-flash", "temperature": 0}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            questions_path = root / "questions.json"
            questions_path.write_text(json.dumps(questions), encoding="utf-8")
            index_path = root / "index.db"
            index_path.write_bytes(b"test index")
            results_path, report_path = root / "results.json", root / "report.md"
            with patch.object(rag_agent, "answer_question", side_effect=[plain, rag, rag_agent.RagError("429")]):
                with self.assertRaisesRegex(rag_agent.RagError, "429"):
                    rag_agent.evaluate(questions_path, results_path, report_path, index_path=index_path)
            partial_path = results_path.with_name(results_path.name + ".partial")
            self.assertTrue(partial_path.exists())
            self.assertFalse(report_path.exists())
            saved = json.loads(partial_path.read_text(encoding="utf-8"))
            self.assertEqual(len(saved["questions"]), 1)
            deepseek_plain = {**plain, "model": "deepseek-v4-flash"}
            deepseek_rag = {**rag, "model": "deepseek-v4-flash"}
            with patch.object(rag_agent, "answer_question", side_effect=[deepseek_plain, deepseek_rag]) as answer:
                data = rag_agent.evaluate(questions_path, results_path, report_path, index_path=index_path, model="deepseek-v4-flash")
            self.assertEqual(answer.call_count, 2)
            self.assertEqual(len(data["questions"]), 2)
            self.assertEqual(data["settings"]["model"], "mixed")
            self.assertEqual(data["models_used"], ["deepseek-v4-flash", "glm-4.7-flash"])
            self.assertEqual([call.kwargs["model"] for call in answer.call_args_list],
                             ["deepseek-v4-flash", "deepseek-v4-flash"])
            self.assertFalse(partial_path.exists())

    def test_evaluate_does_not_replace_existing_reviewed_report(self):
        questions = [{"id": 1, "question": "Вопрос", "expected_theses": ["a"], "expected_source": "handoff.xhtml"}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            questions_path = root / "questions.json"
            questions_path.write_text(json.dumps(questions), encoding="utf-8")
            index_path = root / "index.db"
            index_path.write_bytes(b"test index")
            results_path, report_path = root / "results.json", root / "report.md"
            results_path.write_text("old results", encoding="utf-8")
            report_path.write_text("reviewed scores", encoding="utf-8")
            with patch.object(rag_agent, "answer_question", side_effect=AssertionError("provider called")) as answer:
                with self.assertRaisesRegex(rag_agent.RagError, "существ"):
                    rag_agent.evaluate(questions_path, results_path, report_path, index_path=index_path)
            answer.assert_not_called()
            self.assertEqual(results_path.read_text(encoding="utf-8"), "old results")
            self.assertEqual(report_path.read_text(encoding="utf-8"), "reviewed scores")

    def test_resume_rejects_changed_questions_or_index(self):
        controls = [
            {"id": 1, "question": "Первый", "expected_theses": ["a"], "expected_source": "a.xhtml"},
            {"id": 2, "question": "Второй", "expected_theses": ["b"], "expected_source": "b.xhtml"},
        ]
        plain = {"answer": "обычный", "chunks": [], "model": "glm-4.7-flash", "temperature": 0}
        rag = {"answer": "источник [1]", "chunks": [self.row], "model": "glm-4.7-flash", "temperature": 0}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            questions_path, index_path = root / "questions.json", root / "index.db"
            results_path, report_path = root / "results.json", root / "report.md"
            questions_path.write_text(json.dumps(controls), encoding="utf-8")
            index_path.write_bytes(b"index v1")
            with patch.object(rag_agent, "answer_question", side_effect=[plain, rag, rag_agent.RagError("429")]):
                with self.assertRaises(rag_agent.RagError):
                    rag_agent.evaluate(questions_path, results_path, report_path, index_path=index_path)
            changed = [controls[0], {**controls[1], "question": "Изменённый"}]
            questions_path.write_text(json.dumps(changed), encoding="utf-8")
            with patch.object(rag_agent, "answer_question", side_effect=AssertionError("provider called")) as answer:
                with self.assertRaisesRegex(rag_agent.RagError, "измен"):
                    rag_agent.evaluate(questions_path, results_path, report_path, index_path=index_path)
            answer.assert_not_called()
            questions_path.write_text(json.dumps(controls), encoding="utf-8")
            index_path.write_bytes(b"index v2")
            with patch.object(rag_agent, "answer_question", side_effect=AssertionError("provider called")) as answer:
                with self.assertRaisesRegex(rag_agent.RagError, "измен"):
                    rag_agent.evaluate(questions_path, results_path, report_path, index_path=index_path)
            answer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
