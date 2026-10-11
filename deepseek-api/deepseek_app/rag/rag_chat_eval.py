"""Run scripted Day 25 conversations through the production turn path."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from deepseek_app.rag.document_index import DEFAULT_DB
from deepseek_app.rag.rag_chat import DEFAULT_CHAT_MODEL, run_turn
from deepseek_app.rag.rag_chat_store import ChatStore
from deepseek_app.rag.rag_agent import _atomic_write
from deepseek_app.rag.rag_day24 import (MIN_RERANK_SCORE, TOP_K_AFTER, TOP_K_BEFORE,
                       DEFAULT_RERANKER_MODEL)


def summarize(data):
    turns = data["turns"]
    return {"attempted_count": len(turns),
            "retrieval_count": sum(bool(item.get("search_query")) for item in turns),
            "answered_count": sum(item.get("status") == "answered" for item in turns),
            "refusal_count": sum(item.get("status") == "no_evidence" for item in turns),
            "error_count": sum(item.get("status") == "error" for item in turns)}


def render_report(data, reviews):
    """Render machine evidence and separately supplied human grounding judgments."""
    summary = data.get("summary") or summarize(data)
    lines = ["# День 25 — сценарная оценка RAG-чата", "",
             f"Поиск: {summary['retrieval_count']}/{summary['attempted_count']} ходов; "
             f"ответов: {summary['answered_count']}; отказов: {summary['refusal_count']}; "
             f"ошибок итоговых ходов: {summary['error_count']}.", "",
             "Оценка «опора» ниже — ручная сверка значимых утверждений с цитатами "
             "текущего хода; дословность и принадлежность чанка проверяются кодом.", "",
             "| Сценарий/ход | Статус | Поиск | Источники | Цитаты | Цель | "
             "Действующие ограничения | Опора | Комментарий |",
             "|---|---|---|---|---:|---|---|---|---|"]
    for entry in data["turns"]:
        key = (entry["scenario"], entry["turn"])
        grade, note = reviews.get(key, ("не оценено", ""))
        state = entry["state"]
        goal = state["goal"]
        goal_label = f"{goal['text']} (№{goal['turn']})" if goal else "нет"
        constraints = "; ".join(f"{item['text']} (№{item['turn']})"
                                for item in state["constraints"]) or "нет"
        sources = "; ".join(dict.fromkeys(source["source"] for source in
                                          entry.get("sources", []))) or "нет"
        cells = [f"{entry['scenario']}/{entry['turn']}", entry["status"],
                 "да" if entry.get("search_query") else "нет", sources,
                 str(len(entry.get("citations", []))), goal_label, constraints,
                 grade, note]
        lines.append("| " + " | ".join(str(cell).replace("|", "\\|").replace("\n", " ")
                                      for cell in cells) + " |")
    if data.get("failed_attempts"):
        lines += ["", "## Неудачные попытки, сохранённые при повторе", ""]
        for entry in data["failed_attempts"]:
            lines.append(f"- {entry['scenario']}/{entry['turn']}: {entry['error']}")
    return "\n".join(lines) + "\n"


def _turn_entry(name, number, message, result, state, settings, query=None):
    return {"scenario": name, "turn": number, "user_text": message,
            "settings": settings, "status": result["status"],
            "search_query": query if query is not None else result.get("search_query"),
            "answer": result["answer"], "reason": result.get("reason"),
            "index_metadata": result.get("index"), "sources": result["sources"],
            "citations": result["citations"], "selected": result.get("selected", []),
            "state": state}


def run_scenarios(scenarios_path, results_path, db_path, *, model=DEFAULT_CHAT_MODEL,
                  index_path=None, ollama_url="http://localhost:11434",
                  runner=run_turn, progress=None):
    scenarios_path, results_path, db_path = map(Path, (scenarios_path, results_path, db_path))
    partial = results_path.with_name(results_path.name + ".partial")
    effective_index = Path(index_path) if index_path is not None else DEFAULT_DB
    paths = (scenarios_path, results_path, db_path, partial, effective_index)
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Пути сценариев, результата, базы, прогресса и индекса должны различаться")
    if results_path.exists() or (partial.exists() != db_path.exists()):
        raise FileExistsError("Результат уже существует или неполная пара прогресса и базы")
    scenarios = json.loads(scenarios_path.read_text(encoding="utf-8"))
    settings = {"model": model, "index_path": str(effective_index.resolve()),
                "index_sha256": (hashlib.sha256(effective_index.read_bytes()).hexdigest()
                                 if effective_index.is_file() else None),
                "ollama_url": ollama_url, "top_k_before": TOP_K_BEFORE,
                "top_k_after": TOP_K_AFTER, "min_rerank_score": MIN_RERANK_SCORE,
                "reranker_model": DEFAULT_RERANKER_MODEL}
    if partial.exists():
        data = json.loads(partial.read_text(encoding="utf-8"))
        if (data.get("status") not in {"incomplete", "running"}
                or data.get("scenarios_file") != str(scenarios_path)
                or data.get("settings") != settings):
            raise ValueError("Прогресс не соответствует сценарию")
        if data["turns"] and data["turns"][-1]["status"] == "error":
            data.setdefault("failed_attempts", []).append(data["turns"].pop())
        data["status"] = "running"
    else:
        data = {"created_at": datetime.now(timezone.utc).isoformat(), "status": "running",
                "scenarios_file": str(scenarios_path), "turns": [], "final_states": {},
                "failed_attempts": [], "settings": settings}
    store = ChatStore(db_path)
    counts = {scenario["id"]: 0 for scenario in scenarios}
    for entry in data["turns"]:
        name = entry["scenario"]
        if name not in counts or entry["turn"] != counts[name] + 1 or entry["status"] == "error":
            raise ValueError("Некорректный прогресс сценария")
        if entry.get("settings") != settings:
            raise ValueError("Настройки завершённого хода не совпадают с прогрессом")
        if scenarios[[scenario["id"] for scenario in scenarios].index(name)]["turns"][counts[name]] != entry["user_text"]:
            raise ValueError("Реплики сценария изменились")
        counts[name] += 1
    scenario_map = {scenario["id"]: scenario for scenario in scenarios}
    for name, count in counts.items():
        session = store.load_session(name)
        if session["version"] == count + 1 and count < len(scenario_map[name]["turns"]):
            saved = session["turns"][count]
            if saved["user_text"] != scenario_map[name]["turns"][count]:
                raise ValueError("Сохранённый ход не соответствует сценарию")
            data["turns"].append(_turn_entry(name, count + 1, saved["user_text"],
                                             saved["result"], saved["state"], settings,
                                             query=saved["search_query"]))
            data.setdefault("recovery_events", []).append(
                f"{name}/{count + 1}: ход восстановлен из SQLite после записи без checkpoint")
            counts[name] += 1
            _atomic_write(partial, json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        if session["version"] != counts[name]:
            raise ValueError("База не соответствует сохранённому прогрессу")
    for scenario in scenarios:
        name = scenario["id"]
        for number, message in enumerate(scenario["turns"], 1):
            if number <= counts[name]:
                continue
            if scenario.get("restart_after") == number - 1:
                store = ChatStore(db_path)
            try:
                kwargs = {"model": model}
                if index_path is not None:
                    kwargs["index_path"] = index_path
                kwargs["ollama_url"] = ollama_url
                result = runner(store, name, message, **kwargs)
                entry = _turn_entry(name, number, message, result,
                                    store.load_session(name)["state"], settings)
            except Exception as exc:
                entry = {"scenario": name, "turn": number, "user_text": message,
                         "settings": settings,
                         "status": "error", "error": f"{type(exc).__name__}: {exc}",
                         "search_query": None,
                         "state": store.load_session(name)["state"]}
            data["turns"].append(entry)
            _atomic_write(partial, json.dumps(data, ensure_ascii=False, indent=2) + "\n")
            if progress:
                progress(entry, summarize(data))
            if entry["status"] == "error":
                data["status"] = "incomplete"
                data["summary"] = summarize(data)
                _atomic_write(partial, json.dumps(data, ensure_ascii=False, indent=2) + "\n")
                return data
        data["final_states"][name] = store.load_session(name)["state"]
    data["status"] = "complete"
    data["summary"] = summarize(data)
    data["completed_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_write(results_path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    partial.unlink()
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description="Запуск сценариев RAG-чата")
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--model", default=DEFAULT_CHAT_MODEL)
    args = parser.parse_args(argv)
    def progress(entry, summary):
        print(f"{entry['scenario']} {entry['turn']}: {entry['status']} "
              f"({summary['retrieval_count']} searches)", flush=True)
    result = run_scenarios(args.scenarios, args.results, args.db,
                           model=args.model, index_path=args.index,
                           ollama_url=args.ollama_url, progress=progress)
    print(result["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
