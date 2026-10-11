"""Citation-grounded Day 24 RAG CLI and evaluation."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

from deepseek_app.rag.document_index import DEFAULT_DB, OllamaEmbedder, load_index, search_index
from deepseek_app.providers.providers import ask_model
from deepseek_app.rag.rag_agent import MODEL, RagError, _atomic_write, _input_fingerprint, _read_questions
from deepseek_app.rag.rag_citations import (MODELS, NO_EVIDENCE_ANSWER, build_answer_payload,
                           validate_answer_json)
from deepseek_app.rag.rag_reranker import (DEFAULT_RERANKER_MODEL, CrossEncoderReranker,
                          rerank_candidates)


TOP_K_BEFORE = 20
TOP_K_AFTER = 5
MIN_RERANK_SCORE = 0.50
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_QUESTIONS = PROJECT_ROOT / "docs/day-22/control_questions.json"
DEFAULT_RESULTS = PROJECT_ROOT / "docs/day-24/results.json"
DEFAULT_REPORT = PROJECT_ROOT / "docs/day-24/evaluation.md"


def _validate(question, model, top_k_before, top_k_after, min_rerank_score):
    if not isinstance(question, str) or not question.strip():
        raise RagError("Укажите непустой вопрос")
    if model not in MODELS:
        raise RagError("Неизвестная модель")
    if (isinstance(top_k_before, bool) or not isinstance(top_k_before, int)
            or isinstance(top_k_after, bool) or not isinstance(top_k_after, int)
            or not top_k_before >= top_k_after > 0):
        raise RagError("Требуется top_k_before >= top_k_after > 0")
    if (isinstance(min_rerank_score, bool)
            or not isinstance(min_rerank_score, (int, float))
            or not math.isfinite(min_rerank_score)
            or not 0 <= min_rerank_score <= 1):
        raise RagError("Порог min_rerank_score должен быть конечным числом в [0, 1]")


def answer_question_day24(question, *, index_path=DEFAULT_DB,
                          ollama_url="http://localhost:11434", model=MODEL,
                          top_k_before=TOP_K_BEFORE, top_k_after=TOP_K_AFTER,
                          min_rerank_score=MIN_RERANK_SCORE,
                          reranker_model=DEFAULT_RERANKER_MODEL,
                          index_loader=load_index, embedder_factory=OllamaEmbedder,
                          search=search_index, scorer=None, ask=ask_model,
                          search_query=None, conversation_context=None):
    """Search once, rerank, and release an answer only after quote validation."""
    _validate(question, model, top_k_before, top_k_after, min_rerank_score)
    question = question.strip()
    effective_query = question if search_query is None else search_query.strip()
    if not effective_query:
        raise RagError("Укажите непустой поисковый запрос")
    try:
        metadata, _ = index_loader(index_path)
        embedder = embedder_factory(metadata["model"], ollama_url)
        found = search(index_path, effective_query, "structural", embedder,
                       top_k=top_k_before)
    except Exception as exc:
        raise RagError(f"Поиск: {exc}") from exc
    candidates = []
    for rank, row in enumerate(found, 1):
        candidate = {key: value for key, value in row.items() if key != "vector"}
        candidate["bi_score"] = row["score"]
        candidate["bi_rank"] = rank
        candidates.append(candidate)
    selected = []
    if candidates:
        try:
            selected = rerank_candidates(
                effective_query, candidates,
                scorer if scorer is not None else CrossEncoderReranker(reranker_model),
                top_k_after=top_k_after, min_rerank_score=min_rerank_score)
        except Exception as exc:
            raise RagError(f"Reranker: {exc}") from exc
    result = {
        "question": question, "search_query": effective_query,
        "model": model, "temperature": 0,
        "index": {"path": str(index_path), "embedding_model": metadata["model"],
                  "epub_sha256": metadata["epub_sha256"], "strategy": "structural"},
        "settings": {"top_k_before": top_k_before, "top_k_after": top_k_after,
                     "min_rerank_score": min_rerank_score,
                     "reranker_model": reranker_model},
        "selected": selected,
    }
    if not selected:
        return {**result, "status": "no_evidence", "reason": "below_threshold",
                "answer": NO_EVIDENCE_ANSWER, "sources": [], "citations": []}
    payload = build_answer_payload(question, selected, model,
                                   conversation_context=conversation_context)
    for attempt in range(2):
        try:
            response = ask(payload)
        except Exception as exc:
            raise RagError(f"Провайдер ответа: {exc}") from exc
        content = response.get("content") if isinstance(response, dict) else None
        try:
            verified = validate_answer_json(content, selected)
        except ValueError as exc:
            if attempt:
                raise RagError(f"Ответ модели не прошёл проверку: {exc}") from exc
            repair = f"Исправь формат JSON и цитаты: {exc}. "
            if search_query is not None or conversation_context is not None:
                repair += (
                    "Скопируй каждую quote дословно из текста соответствующего chunk_id "
                    "в переданных фрагментах, включая пробелы и знаки препинания. "
                    "Если не можешь подтвердить ответ дословной цитатой, верни "
                    'status=no_evidence, answer="", citations=[]. '
                )
            repair += "Верни один JSON-объект по тем же фрагментам без дополнительного текста."
            payload = {**payload, "messages": [*payload["messages"], {
                "role": "user", "content": (
                    repair
                )}]}
        else:
            if verified["status"] == "no_evidence":
                verified["reason"] = "model_abstained"
            return {**result, **verified}


def _validate_controls(controls):
    if len(controls) != 10:
        raise RagError("Нужно ровно десять контрольных вопросов")
    ids = set()
    for control in controls:
        identifier = control.get("id")
        theses = control.get("expected_theses")
        if (isinstance(identifier, bool) or not isinstance(identifier, int)
                or identifier <= 0 or identifier in ids
                or not isinstance(control.get("expected_source"), str)
                or not control["expected_source"].strip()
                or not isinstance(theses, list) or not theses
                or any(not isinstance(item, str) or not item.strip() for item in theses)):
            raise RagError("Некорректный контрольный вопрос: id, глава и тезисы обязательны")
        ids.add(identifier)


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _machine_checks(result):
    selected = {row["chunk_id"]: row for row in result["selected"]}
    citations = result["citations"]
    sources = result["sources"]
    answered = result["status"] == "answered"
    return {
        "sources_present": bool(sources) if answered else False,
        "citations_present": bool(citations) if answered else False,
        "selected_chunk_membership": (bool(citations) and all(
            citation["chunk_id"] in selected for citation in citations)) if answered else False,
        "quotes_verbatim": (bool(citations) and all(
            citation["chunk_id"] in selected and
            citation["quote"] in selected[citation["chunk_id"]]["text"][:1800]
            for citation in citations)) if answered else False,
    }


def _source_hit(rows, expected_source):
    return any(row["source"].replace(":", "/").endswith("/" + expected_source)
               for row in rows)


def _report(data):
    settings = data["settings"]
    lines = ["# День 24 — цитаты и отказ при слабом контексте", "",
             f"Модель: `{settings['model']}`; порог cross-encoder: "
             f"{settings['min_rerank_score']}; Top-{settings['top_k_before']} → "
             f"Top-{settings['top_k_after']}. Score не является калиброванной вероятностью.",
             f"Содержательных ответов: {data['answered_count']}; "
             f"отказов: {data['no_evidence_count']}.", "",
             "## Проверка десяти вопросов", "",
             "| № | Статус | Источники | Цитаты | Дословность | Смысл ответа подтверждён | Комментарий |",
             "|---:|---|---|---|---|---|---|"]
    for item in data["questions"]:
        check = item["checks"]
        yn = lambda value: "да" if value else "нет"
        lines.append(f"| {item['control']['id']} | {item['result']['status']} | "
                     f"{yn(check['sources_present'])} | {yn(check['citations_present'])} | "
                     f"{yn(check['quotes_verbatim'])} | требует ручной сверки | ожидает оценки |")
    lines += ["", "## Доказательства и ручная оценка", "",
              "Дословность проверена кодом только в первых 1800 символах переданного чанка. "
              "Смысл каждого существенного утверждения требуется сверить вручную с цитатой "
              "и полным выбранным чанком. Отметьте да/частично/нет в таблице выше.", ""]
    for item in data["questions"]:
        control, result = item["control"], item["result"]
        lines += [f"### {control['id']}. {control['question']}", "",
                  f"Ожидаемая глава: `{control['expected_source']}`; "
                  f"попадание: {'да' if item['expected_source_hit'] else 'нет'}.",
                  f"Ожидаемые тезисы: {'; '.join(control['expected_theses'])}.",
                  f"Статус: {result['status']}"
                  + (f" ({result['reason']})" if result.get("reason") else ""),
                  f"Ответ: {result['answer']}", "", "Цитаты:", ""]
        if result["citations"]:
            for citation in result["citations"]:
                lines.append(f"- [{citation['ref']}] `{citation['source']}` / "
                             f"{citation['section']} (`{citation['chunk_id']}`): "
                             f"«{citation['quote']}»")
        else:
            lines.append("- нет")
        lines += ["", "Выбранные чанки и scores:", ""]
        if result["selected"]:
            for row in result["selected"]:
                lines.append(f"- `{row['chunk_id']}` — `{row['source']}`; "
                             f"rerank={row['rerank_score']:.4f}; bi={row['bi_score']:.4f}")
        else:
            lines.append("- нет")
        lines += ["", "Ручная оценка: требует ручной сверки. Комментарий: ожидает оценки.", ""]
    return "\n".join(lines)


def _same_run(data, controls, fingerprint, settings):
    entries = data.get("questions")
    return (data.get("settings") == settings
            and data.get("input_fingerprint") == fingerprint
            and isinstance(entries, list) and len(entries) <= len(controls)
            and all(_valid_entry(item, control, settings)
                    for item, control in zip(entries, controls[:len(entries)])))


def _valid_entry(item, control, settings):
    """Reject incomplete or altered checkpoint entries before skipping them."""
    try:
        if not isinstance(item, dict) or item["control"] != control:
            return False
        result = item["result"]
        if (not isinstance(result, dict)
                or result["question"] != control["question"]
                or result["model"] != settings["model"]
                or result["index"]["path"] != settings["index_path"]
                or result["settings"]["min_rerank_score"] != settings["min_rerank_score"]
                or not isinstance(result["selected"], list)
                or any(not isinstance(row, dict) or "vector" in row
                       for row in result["selected"])):
            return False
        selected = result["selected"]
        ids = [row["chunk_id"] for row in selected]
        if len(ids) != len(set(ids)):
            return False
        for row in selected:
            if (any(not isinstance(row[key], str) or not row[key]
                    for key in ("chunk_id", "source", "section", "text"))
                    or any(isinstance(row[key], bool)
                           or not isinstance(row[key], (int, float))
                           or not math.isfinite(row[key])
                           for key in ("rerank_score", "bi_score"))
                    or not settings["min_rerank_score"] <= row["rerank_score"] <= 1
                    or not -1 <= row["bi_score"] <= 1):
                return False
        if result["status"] == "answered":
            proposed = [{"chunk_id": citation["chunk_id"], "quote": citation["quote"]}
                        for citation in result["citations"]]
            verified = validate_answer_json(json.dumps({
                "status": "answered", "answer": result["answer"],
                "citations": proposed}, ensure_ascii=False), selected)
            if any(result[key] != verified[key] for key in
                   ("answer", "sources", "citations")):
                return False
        elif result["status"] == "no_evidence":
            if (result["answer"] != NO_EVIDENCE_ANSWER or result["sources"] != []
                    or result["citations"] != []
                    or result["reason"] not in ("below_threshold", "model_abstained")
                    or (result["reason"] == "below_threshold" and selected)
                    or (result["reason"] == "model_abstained" and not selected)):
                return False
        else:
            return False
        return (item["checks"] == _machine_checks(result)
                and item["expected_source_hit"] == _source_hit(
                    selected, control["expected_source"]))
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def evaluate(questions_path=DEFAULT_QUESTIONS, results_path=DEFAULT_RESULTS,
             report_path=DEFAULT_REPORT, *, index_path=DEFAULT_DB,
             ollama_url="http://localhost:11434", model=MODEL,
             top_k_before=TOP_K_BEFORE, top_k_after=TOP_K_AFTER,
             min_rerank_score=MIN_RERANK_SCORE,
             reranker_model=DEFAULT_RERANKER_MODEL, overwrite=False,
             answerer=answer_question_day24, progress=None):
    """Checkpoint each question and preserve manual report edits on recovery."""
    _validate("контрольный вопрос", model, top_k_before, top_k_after,
              min_rerank_score)
    questions_path, results_path, report_path, index_path = map(
        Path, (questions_path, results_path, report_path, index_path))
    partial_path = results_path.with_name(results_path.name + ".partial")
    paths = (questions_path, results_path, report_path, index_path, partial_path)
    if len({path.resolve() for path in paths}) != len(paths):
        raise RagError("Пути входных и выходных файлов должны различаться")
    controls = _read_questions(questions_path)
    _validate_controls(controls)
    fingerprint = _input_fingerprint(controls, index_path)
    settings = {"index_path": str(index_path), "ollama_url": ollama_url,
                "model": model, "temperature": 0,
                "top_k_before": top_k_before, "top_k_after": top_k_after,
                "min_rerank_score": min_rerank_score,
                "reranker_model": reranker_model}
    if partial_path.exists():
        try:
            data = json.loads(partial_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RagError(f"Повреждён файл прогресса {partial_path}: {exc}") from exc
        if data.get("status") != "incomplete" or not _same_run(
                data, controls, fingerprint, settings):
            raise RagError("Вопросы, индекс, модель или настройки не совпадают с прогрессом")
    elif results_path.exists() and not report_path.exists() and not overwrite:
        try:
            data = json.loads(results_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RagError(f"Повреждён итоговый JSON: {exc}") from exc
        if data.get("status") != "complete" or not _same_run(
                data, controls, fingerprint, settings) or len(data["questions"]) != 10:
            raise RagError("Итоговый JSON не соответствует текущему прогону")
        _atomic_write(report_path, _report(data))
        return data
    else:
        if not overwrite and (results_path.exists() or report_path.exists()):
            raise RagError("Результат или отчёт уже существует; используйте --overwrite")
        data = {"created_at": datetime.now(timezone.utc).isoformat(),
                "questions_file": str(questions_path), "settings": settings,
                "input_fingerprint": fingerprint, "status": "incomplete",
                "prior_report_sha256": _sha256_file(report_path) if report_path.exists() else None,
                "questions": []}
        _atomic_write(partial_path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    entries = data["questions"]
    same_final_run = False
    if results_path.exists() and len(entries) == 10:
        try:
            final = json.loads(results_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RagError(f"Повреждён итоговый JSON: {exc}") from exc
        same_final_run = final.get("created_at") == data["created_at"]
        if same_final_run and (final.get("status") != "complete" or not _same_run(
                final, controls, fingerprint, settings) or final["questions"] != entries):
            raise RagError("Итоговый JSON не совпадает с прогрессом")
    if same_final_run:
        data = final
    else:
        for control in controls[len(entries):]:
            result = answerer(control["question"], index_path=index_path,
                              ollama_url=ollama_url, model=model,
                              top_k_before=top_k_before, top_k_after=top_k_after,
                              min_rerank_score=min_rerank_score,
                              reranker_model=reranker_model)
            checks = _machine_checks(result)
            entries.append({"control": control, "result": result, "checks": checks,
                            "expected_source_hit": _source_hit(result["selected"],
                                                               control["expected_source"])})
            _atomic_write(partial_path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")
            if progress:
                progress(len(entries), len(controls))
        data["status"] = "complete"
        data["completed_at"] = datetime.now(timezone.utc).isoformat()
        data["answered_count"] = sum(item["result"]["status"] == "answered"
                                     for item in entries)
        data["no_evidence_count"] = sum(item["result"]["status"] == "no_evidence"
                                        for item in entries)
        _atomic_write(results_path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    prior_hash = data.get("prior_report_sha256")
    if report_path.exists() and (prior_hash is None
                                 or _sha256_file(report_path) != prior_hash):
        raise RagError(f"Отчёт {report_path} изменён вручную; итоговый JSON сохранён, "
                       "файл прогресса оставлен. Укажите новый --report для восстановления")
    _atomic_write(report_path, _report(data))
    partial_path.unlink(missing_ok=True)
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description="День 24: вопросы по книге с проверенными цитатами")
    parser.add_argument("--index", type=Path, default=DEFAULT_DB)
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    commands = parser.add_subparsers(dest="command", required=True)
    ask_parser = commands.add_parser("ask", help="Ответить на вопрос")
    ask_parser.add_argument("question")
    eval_parser = commands.add_parser("evaluate", help="Проверить десять вопросов")
    eval_parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    eval_parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    eval_parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    eval_parser.add_argument("--overwrite", action="store_true")
    for command in (ask_parser, eval_parser):
        command.add_argument("--model", choices=MODELS, default=MODEL)
        command.add_argument("--top-k-before", type=int, default=TOP_K_BEFORE)
        command.add_argument("--top-k-after", type=int, default=TOP_K_AFTER)
        command.add_argument("--min-rerank-score", type=float, default=MIN_RERANK_SCORE)
        command.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    args = parser.parse_args(argv)
    options = {"index_path": args.index, "ollama_url": args.ollama_url,
               "model": args.model, "top_k_before": args.top_k_before,
               "top_k_after": args.top_k_after,
               "min_rerank_score": args.min_rerank_score,
               "reranker_model": args.reranker_model}
    try:
        if args.command == "ask":
            result = answer_question_day24(args.question, **options)
            print(f"Ответ: {result['answer']}")
            if result["sources"]:
                print("Источники:")
                for source in result["sources"]:
                    print(f"- {source['source']} / {source['section']} ({source['chunk_id']})")
            else:
                print("Источники: нет")
            if result["citations"]:
                print("Цитаты:")
                for citation in result["citations"]:
                    print(f"[{citation['ref']}] {citation['quote']}")
            else:
                print("Цитаты: нет")
        else:
            data = evaluate(args.questions, args.results, args.report,
                            overwrite=args.overwrite, **options,
                            progress=lambda done, total: print(
                                f"Вопросы: {done}/{total}", file=sys.stderr, flush=True))
            print(f"Готово: {len(data['questions'])} вопросов; {args.results}; {args.report}")
        return 0
    except RagError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
