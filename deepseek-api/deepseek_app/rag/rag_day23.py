"""Day 23 retrieval, reranking, rewrite, and one-question CLI."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

from deepseek_app.rag.document_index import DEFAULT_DB, OllamaEmbedder, load_index, search_index
from deepseek_app.providers.providers import ask_model
from deepseek_app.rag.rag_agent import (MODEL, RagError, _atomic_write, _input_fingerprint,
                       _read_questions, generate_answer)
from deepseek_app.rag.rag_reranker import (DEFAULT_RERANKER_MODEL, CrossEncoderReranker,
                          rerank_candidates, rewrite_query)


TOP_K_BEFORE = 20
TOP_K_AFTER = 5
MIN_RERANK_SCORE = 0.50
NO_EVIDENCE_ANSWER = "В найденных фрагментах книги недостаточно сведений для ответа на этот вопрос."
MODES = ("baseline", "reranked", "rewritten")
MODELS = ("glm-4.7-flash", "deepseek-v4-flash", "deepseek-v4-pro")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_QUESTIONS = PROJECT_ROOT / "docs/day-22/control_questions.json"
DEFAULT_RESULTS = PROJECT_ROOT / "docs/day-23/results.json"
DEFAULT_REPORT = PROJECT_ROOT / "docs/day-23/comparison.md"


def _validate(question, mode, model, top_k_before, top_k_after, min_rerank_score):
    if not isinstance(question, str) or not question.strip():
        raise RagError("Укажите непустой вопрос")
    if mode not in MODES:
        raise RagError("Неизвестный режим; выберите baseline, reranked или rewritten")
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


def _public_row(row):
    return {key: value for key, value in row.items() if key != "vector"}


def answer_question_day23(question, mode, *, index_path=DEFAULT_DB,
                          ollama_url="http://localhost:11434", model=MODEL,
                          top_k_before=TOP_K_BEFORE, top_k_after=TOP_K_AFTER,
                          min_rerank_score=MIN_RERANK_SCORE,
                          reranker_model=DEFAULT_RERANKER_MODEL,
                          index_loader=load_index, embedder_factory=OllamaEmbedder,
                          search=search_index, scorer=None, rewriter=rewrite_query,
                          ask=ask_model, first_stage=None):
    """Use a supplied first-stage pool, or search once for a standalone question."""
    _validate(question, mode, model, top_k_before, top_k_after, min_rerank_score)
    question = question.strip()
    try:
        if first_stage is None:
            metadata, _ = index_loader(index_path)
            embedder = embedder_factory(metadata["model"], ollama_url)
            found = search(index_path, question, "structural", embedder, top_k=top_k_before)
        else:
            metadata, found = first_stage
            if (not isinstance(metadata, dict) or not isinstance(found, list)
                    or "model" not in metadata or "epub_sha256" not in metadata):
                raise ValueError("некорректный готовый результат первого поиска")
    except Exception as exc:
        raise RagError(f"Поиск: {exc}") from exc

    candidates = []
    for rank, row in enumerate(found, 1):
        candidate = _public_row(row)
        candidate["bi_score"] = row["score"]
        candidate["bi_rank"] = rank
        candidates.append(candidate)
    search_query = question
    selected = []
    if candidates:
        if mode == "baseline":
            selected = candidates[:top_k_after]
        else:
            if mode == "rewritten":
                def rewrite_ask(payload, **options):
                    options["extra_body"] = {"thinking": {"type": "disabled"}}
                    return ask(payload, **options)

                try:
                    search_query = rewriter(question, candidates, model=model, ask=rewrite_ask)
                except Exception as exc:
                    raise RagError(f"Rewrite: {exc}") from exc
            try:
                ranked = rerank_candidates(
                    search_query, candidates,
                    scorer if scorer is not None else CrossEncoderReranker(reranker_model),
                    top_k_after=len(candidates), min_rerank_score=0)
            except Exception as exc:
                raise RagError(f"Reranker: {exc}") from exc
            by_bi_rank = {row["bi_rank"]: row for row in ranked}
            candidates = [by_bi_rank[row["bi_rank"]] for row in candidates]
            selected = [row for row in ranked
                        if row["rerank_score"] >= min_rerank_score][:top_k_after]

    answer = generate_answer(question, selected, model=model, ask=ask) if selected else NO_EVIDENCE_ANSWER
    return {
        "question": question, "mode": mode, "search_query": search_query,
        "answer": answer, "status": "answered" if selected else "no_evidence",
        "model": model, "temperature": 0,
        "index": {"path": str(index_path), "embedding_model": metadata["model"],
                  "epub_sha256": metadata["epub_sha256"], "strategy": "structural"},
        "settings": {"top_k_before": top_k_before, "top_k_after": top_k_after,
                     "min_rerank_score": min_rerank_score,
                     "reranker_model": reranker_model if mode != "baseline" else None},
        "candidates": candidates, "selected": selected,
    }


def _show_rows(label, rows):
    print(f"\n{label}:")
    for row in rows:
        scores = f"cosine={row['bi_score']:.4f} (#{row['bi_rank']})"
        if "rerank_score" in row:
            scores += f", rerank={row['rerank_score']:.4f} (#{row['rerank_rank']})"
        print(f"- {row['chunk_id']} {row['source']} / {row['section']}: {scores}")


def _source_hit(rows, expected_source):
    return any(row["source"].replace(":", "/").endswith("/" + expected_source)
               for row in rows)


def _report(data):
    settings = data["settings"]
    lines = ["# День 23 — сравнение режимов RAG", "",
             f"Прогон: {data['created_at']}. Статус: черновик; проверено: —; дата ручной проверки: —.",
             f"Модель ответа: `{settings['model']}`; embedding: `{settings['embedding_model']}`; "
             f"cross-encoder: `{settings['reranker_model']}`; температура: 0.",
             f"Индекс: `{settings['index_path']}`; Ollama: `{settings['ollama_url']}`. "
             f"Top-{settings['top_k_before']} до rerank, Top-{settings['top_k_after']} после; "
             f"порог: {settings['min_rerank_score']}.",
             "", "## Источники и ответы", ""]
    for item in data["questions"]:
        control = item["control"]
        lines += [f"### {control['id']}. {control['question']}", "",
                  f"Ожидаемая глава: `{control['expected_source']}`; "
                  f"ожидаемые тезисы: {'; '.join(control['expected_theses'])}.", ""]
        for mode in MODES:
            result = item[mode]
            selected = result["selected"]
            lines += [f"#### {mode}", "",
                      f"Поисковый запрос: {result['search_query']}",
                      f"Ожидаемая глава в выбранных источниках: "
                      f"{'да' if item['expected_source_hit'][mode] else 'нет'}.",
                      f"Ответ: {result['answer']}", "",
                      "Выбранные источники в порядке контекста:", ""]
            if selected:
                for row in selected:
                    score = f"cosine={row['bi_score']:.4f} (#{row['bi_rank']})"
                    if "rerank_score" in row:
                        score += f", rerank={row['rerank_score']:.4f} (#{row['rerank_rank']})"
                    lines.append(f"- `{row['chunk_id']}` — `{row['source']}`; {score}")
            else:
                lines.append("- нет")
            lines += ["", "Кандидаты первого этапа в порядке bi-encoder (полный список):", ""]
            for row in result["candidates"]:
                score = f"cosine={row['bi_score']:.4f} (#{row['bi_rank']})"
                if "rerank_score" in row:
                    score += f", rerank={row['rerank_score']:.4f} (#{row['rerank_rank']})"
                lines.append(f"- `{row['chunk_id']}` — `{row['source']}`; {score}")
            lines.append("")
    lines += ["## Ручная оценка", "",
              "Полнота каждого ответа: 0–2 (0 — нет существенных тезисов, 1 — часть, "
              "2 — все). Опора утверждений на выбранный текст: да/частично/нет. "
              "До ручной сверки баллы не выставлены.", "",
              "| № | baseline 0–2 | reranked 0–2 | rewritten 0–2 | "
              "Опора baseline | Опора reranked | Опора rewritten | Комментарий |",
              "|---:|---:|---:|---:|---|---|---|---|"]
    for item in data["questions"]:
        lines.append(f"| {item['control']['id']} | — | — | — | — | — | — | требует ручной сверки |")
    lines += ["", "## Сравнение после ручной проверки", "",
              "Улучшения: —; ухудшения: —; равные результаты: —. "
              "Оценить по полноте и опоре на текст; сокращение числа фрагментов само по себе "
              "не означает улучшения.", "", "## Ограничения", "",
              "Cross-encoder обрезает текст токенизатором до 512 токенов; score после sigmoid "
              "не является калиброванной вероятностью. Rewrite не возвращает документы вне "
              "исходного Top-K. Попадание главы в источники не доказывает правильность ответа. "
              "Один прогон десяти вопросов не измеряет общую точность. Вопросы вне книги "
              "проверяются отдельно; порог после итогового прогона не подгоняется.", ""]
    return "\n".join(lines)


def _validate_controls(controls):
    ids = set()
    for control in controls:
        identifier = control.get("id")
        theses = control.get("expected_theses")
        if (isinstance(identifier, bool) or not isinstance(identifier, int)
                or identifier <= 0 or identifier in ids
                or not isinstance(theses, list)
                or any(not isinstance(thesis, str) or not thesis.strip()
                       for thesis in theses)
                or not control["expected_source"].strip()):
            raise RagError("Некорректный контрольный вопрос: требуются уникальный id, "
                           "непустой expected_source и строковые expected_theses")
        ids.add(identifier)


def _valid_row(row, reranked):
    if not isinstance(row, dict):
        return False
    if any(not isinstance(row.get(key), str) or not row[key]
           for key in ("chunk_id", "source")):
        return False
    score = row.get("bi_score")
    rank = row.get("bi_rank")
    if (isinstance(score, bool) or not isinstance(score, (int, float))
            or not math.isfinite(score) or isinstance(rank, bool)
            or not isinstance(rank, int) or rank <= 0):
        return False
    if reranked:
        rerank_score = row.get("rerank_score")
        rerank_rank = row.get("rerank_rank")
        if (isinstance(rerank_score, bool)
                or not isinstance(rerank_score, (int, float))
                or not math.isfinite(rerank_score)
                or not 0 <= rerank_score <= 1
                or isinstance(rerank_rank, bool)
                or not isinstance(rerank_rank, int) or rerank_rank <= 0):
            return False
    return True


def _valid_result(result, control, mode, model):
    if not isinstance(result, dict):
        return False
    if (result.get("question") != control["question"] or result.get("mode") != mode
            or result.get("model") != model
            or not isinstance(result.get("search_query"), str)
            or not result["search_query"].strip()
            or not isinstance(result.get("answer"), str)
            or not isinstance(result.get("candidates"), list)
            or not isinstance(result.get("selected"), list)):
        return False
    reranked = mode != "baseline"
    return all(_valid_row(row, reranked) for row in result["candidates"] + result["selected"])


def _render_report(data):
    try:
        return _report(data)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise RagError(f"Не удалось составить отчёт из результатов: {exc}") from exc


def _validate_partial(data, controls, settings, fingerprint, partial_path):
    results = data.get("questions") if isinstance(data, dict) else None
    if (not isinstance(results, list) or len(results) > len(controls)
            or data.get("status") != "incomplete"
            or data.get("settings") != settings
            or data.get("input_fingerprint") != fingerprint
            or [item.get("control") if isinstance(item, dict) else None for item in results]
            != controls[:len(results)]
            or any(not all(mode in item for mode in MODES)
                   or not isinstance(item.get("expected_source_hit"), dict)
                   or set(item["expected_source_hit"]) != set(MODES)
                   or any(not isinstance(item["expected_source_hit"][mode], bool)
                          or not _valid_result(item[mode], item["control"], mode,
                                               settings["model"])
                          for mode in MODES)
                   for item in results)):
        raise RagError(f"Настройки, вопросы или индекс изменились либо результаты повреждены; "
                       f"файл прогресса {partial_path} не соответствует прогону")
    return results


def _matching_final(saved, data):
    return (isinstance(saved, dict) and saved.get("status") == "complete"
            and saved.get("settings") == data["settings"]
            and saved.get("input_fingerprint") == data["input_fingerprint"]
            and saved.get("questions") == data["questions"]
            and saved.get("created_at") == data["created_at"]
            and saved.get("questions_file") == data["questions_file"]
            and saved.get("prior_report_sha256") == data.get("prior_report_sha256")
            and isinstance(saved.get("completed_at"), str)
            and bool(saved["completed_at"]))


def _file_sha256(path):
    if not path.exists():
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise RagError(f"Не удалось прочитать существующий отчёт {path}: {exc}") from exc
    return digest.hexdigest()


def evaluate(questions_path=DEFAULT_QUESTIONS, results_path=DEFAULT_RESULTS,
             report_path=DEFAULT_REPORT, *, index_path=DEFAULT_DB,
             ollama_url="http://localhost:11434", model=MODEL,
             top_k_before=TOP_K_BEFORE, top_k_after=TOP_K_AFTER,
             min_rerank_score=MIN_RERANK_SCORE,
             reranker_model=DEFAULT_RERANKER_MODEL, overwrite=False,
             progress=None):
    """Run all control questions in mode order, saving complete-question checkpoints."""
    _validate("контрольный вопрос", "baseline", model, top_k_before, top_k_after,
              min_rerank_score)
    questions_path, index_path = Path(questions_path), Path(index_path)
    results_path, report_path = Path(results_path), Path(report_path)
    partial_path = results_path.with_name(results_path.name + ".partial")
    output_paths = [results_path.resolve(), report_path.resolve(), partial_path.resolve()]
    input_paths = {questions_path.resolve(), index_path.resolve()}
    if len(set(output_paths)) != len(output_paths) or input_paths.intersection(output_paths):
        raise RagError("Пути результатов, отчёта, прогресса, вопросов и индекса не должны совпадать")
    if not overwrite and not partial_path.exists() and (results_path.exists() or report_path.exists()):
        raise RagError("Итоговый результат или отчёт уже существует; укажите --overwrite")
    controls = _read_questions(questions_path)
    _validate_controls(controls)
    fingerprint = _input_fingerprint(controls, index_path)
    try:
        index_meta, _ = load_index(index_path)
    except Exception as exc:
        raise RagError(f"Индекс: {exc}") from exc
    settings = {"index_path": str(index_path), "ollama_url": ollama_url,
                "embedding_model": index_meta["model"], "model": model,
                "temperature": 0, "top_k_before": top_k_before,
                "top_k_after": top_k_after, "min_rerank_score": min_rerank_score,
                "reranker_model": reranker_model}
    if partial_path.exists():
        try:
            data = json.loads(partial_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RagError(f"Повреждён файл прогресса {partial_path}: {exc}") from exc
        results = _validate_partial(data, controls, settings, fingerprint, partial_path)
    else:
        data = {"created_at": datetime.now(timezone.utc).isoformat(),
                "questions_file": str(questions_path), "settings": settings,
                "input_fingerprint": fingerprint, "status": "incomplete", "questions": [],
                "prior_report_sha256": _file_sha256(report_path) if overwrite else None}
        results = data["questions"]
    if results_path.exists():
        try:
            saved = json.loads(results_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            if not overwrite:
                raise RagError(f"Итоговый JSON повреждён: {exc}") from exc
            saved = None
        if len(results) == len(controls) and _matching_final(saved, data):
            report_text = _render_report(saved)
            if not report_path.exists():
                _atomic_write(report_path, report_text)
            elif (overwrite and data.get("prior_report_sha256") is not None
                  and _file_sha256(report_path) == data["prior_report_sha256"]
                  and _file_sha256(report_path) != hashlib.sha256(
                      report_text.encode("utf-8")).hexdigest()):
                _atomic_write(report_path, report_text)
            partial_path.unlink(missing_ok=True)
            return saved
        if not overwrite:
            raise RagError("Существующий итоговый JSON не соответствует файлу прогресса")
    elif report_path.exists() and not overwrite:
        raise RagError("Отчёт уже существует без итогового JSON; требуется --overwrite")
    scorer = CrossEncoderReranker(reranker_model)
    try:
        embedder = OllamaEmbedder(index_meta["model"], ollama_url)
    except Exception as exc:
        raise RagError(f"Поиск: {exc}") from exc
    for control in controls[len(results):]:
        modes = {}
        hits = {}
        try:
            found = search_index(index_path, control["question"], "structural",
                                 embedder, top_k=top_k_before)
        except Exception as exc:
            raise RagError(f"Поиск: {exc}") from exc
        first_stage = (index_meta, found)
        for mode in MODES:
            result = answer_question_day23(
                control["question"], mode, index_path=index_path,
                ollama_url=ollama_url, model=model, top_k_before=top_k_before,
                top_k_after=top_k_after, min_rerank_score=min_rerank_score,
                reranker_model=reranker_model, scorer=scorer, first_stage=first_stage)
            if not _valid_result(result, control, mode, model):
                raise RagError(f"Некорректный результат режима {mode} для вопроса {control['id']}")
            modes[mode] = result
            hits[mode] = _source_hit(result["selected"], control["expected_source"])
        results.append({"control": control, **modes, "expected_source_hit": hits})
        _atomic_write(partial_path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        if progress:
            progress(len(results), len(controls))
    data["status"] = "complete"
    data["completed_at"] = datetime.now(timezone.utc).isoformat()
    report_text = _render_report(data)
    _atomic_write(results_path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    _atomic_write(report_path, report_text)
    partial_path.unlink(missing_ok=True)
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description="День 23: Top-20, reranking, rewrite и ответ")
    parser.add_argument("--index", type=Path, default=DEFAULT_DB)
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    commands = parser.add_subparsers(dest="command", required=True)
    ask_parser = commands.add_parser("ask", help="Ответить на один вопрос")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--mode", choices=MODES, default="baseline")
    ask_parser.add_argument("--model", choices=MODELS, default=MODEL)
    ask_parser.add_argument("--top-k-before", type=int, default=TOP_K_BEFORE)
    ask_parser.add_argument("--top-k-after", type=int, default=TOP_K_AFTER)
    ask_parser.add_argument("--min-rerank-score", type=float, default=MIN_RERANK_SCORE)
    ask_parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    eval_parser = commands.add_parser("evaluate", help="Сравнить три режима на контрольных вопросах")
    eval_parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    eval_parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    eval_parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    eval_parser.add_argument("--model", choices=MODELS, default=MODEL)
    eval_parser.add_argument("--top-k-before", type=int, default=TOP_K_BEFORE)
    eval_parser.add_argument("--top-k-after", type=int, default=TOP_K_AFTER)
    eval_parser.add_argument("--min-rerank-score", type=float, default=MIN_RERANK_SCORE)
    eval_parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    eval_parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "evaluate":
            data = evaluate(args.questions, args.results, args.report, index_path=args.index,
                            ollama_url=args.ollama_url, model=args.model,
                            top_k_before=args.top_k_before, top_k_after=args.top_k_after,
                            min_rerank_score=args.min_rerank_score,
                            reranker_model=args.reranker_model, overwrite=args.overwrite,
                            progress=lambda done, total: print(
                                f"Вопросы: {done}/{total}", file=sys.stderr, flush=True))
            print(f"Готово: {len(data['questions'])} вопросов; {args.results}; {args.report}")
            return 0
        result = answer_question_day23(
            args.question, args.mode, index_path=args.index, ollama_url=args.ollama_url,
            model=args.model, top_k_before=args.top_k_before,
            top_k_after=args.top_k_after, min_rerank_score=args.min_rerank_score,
            reranker_model=args.reranker_model)
    except RagError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    print(f"Исходный вопрос: {result['question']}")
    print(f"Поисковый запрос: {result['search_query']}")
    print(f"Ответ: {result['answer']}")
    _show_rows(f"Кандидаты Top-{result['settings']['top_k_before']}", result["candidates"])
    _show_rows(f"Контекст Top-{result['settings']['top_k_after']}", result["selected"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
