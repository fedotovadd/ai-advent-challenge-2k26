"""Run scripted Day 25 conversations through the production turn path."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from rag_chat import DEFAULT_CHAT_MODEL, run_turn
from rag_chat_store import ChatStore
from rag_agent import _atomic_write


def summarize(data):
    turns = data["turns"]
    return {"attempted_count": len(turns),
            "retrieval_count": sum(bool(item.get("search_query")) for item in turns),
            "answered_count": sum(item.get("status") == "answered" for item in turns),
            "refusal_count": sum(item.get("status") == "no_evidence" for item in turns),
            "error_count": sum(item.get("status") == "error" for item in turns)}


def run_scenarios(scenarios_path, results_path, db_path, *, model=DEFAULT_CHAT_MODEL,
                  index_path=None, ollama_url="http://localhost:11434",
                  runner=run_turn, progress=None):
    scenarios_path, results_path, db_path = map(Path, (scenarios_path, results_path, db_path))
    partial = results_path.with_name(results_path.name + ".partial")
    if results_path.exists() or (partial.exists() != db_path.exists()):
        raise FileExistsError("Результат уже существует или неполная пара прогресса и базы")
    scenarios = json.loads(scenarios_path.read_text(encoding="utf-8"))
    if partial.exists():
        data = json.loads(partial.read_text(encoding="utf-8"))
        if data.get("status") != "incomplete" or data.get("scenarios_file") != str(scenarios_path):
            raise ValueError("Прогресс не соответствует сценарию")
        if data["turns"] and data["turns"][-1]["status"] == "error":
            data.setdefault("failed_attempts", []).append(data["turns"].pop())
        data["status"] = "running"
    else:
        data = {"created_at": datetime.now(timezone.utc).isoformat(), "status": "running",
                "scenarios_file": str(scenarios_path), "turns": [], "final_states": {},
                "failed_attempts": []}
    store = ChatStore(db_path)
    counts = {scenario["id"]: 0 for scenario in scenarios}
    for entry in data["turns"]:
        name = entry["scenario"]
        if name not in counts or entry["turn"] != counts[name] + 1 or entry["status"] == "error":
            raise ValueError("Некорректный прогресс сценария")
        if scenarios[[scenario["id"] for scenario in scenarios].index(name)]["turns"][counts[name]] != entry["user_text"]:
            raise ValueError("Реплики сценария изменились")
        counts[name] += 1
    if any(store.load_session(name)["version"] != count for name, count in counts.items()):
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
                entry = {"scenario": name, "turn": number, "user_text": message,
                         "status": result["status"], "search_query": result.get("search_query"),
                         "answer": result["answer"], "sources": result["sources"],
                         "citations": result["citations"],
                         "selected": result.get("selected", []),
                         "state": store.load_session(name)["state"]}
            except Exception as exc:
                entry = {"scenario": name, "turn": number, "user_text": message,
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
