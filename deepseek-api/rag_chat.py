"""Persistent, citation-grounded CLI chat over the Day 21 book index."""

import argparse
from pathlib import Path

from document_index import DEFAULT_DB
from providers import ask_model
from rag_chat_memory import (conversation_context, prepare_payload,
                             validate_preparation)
from rag_chat_store import ChatStore, DEFAULT_CHAT_DB
from rag_day24 import answer_question_day24


DEFAULT_CHAT_MODEL = "deepseek-v4-flash"


def run_turn(store, session_name, message, *, model=DEFAULT_CHAT_MODEL, prepare=ask_model,
             answer=answer_question_day24, index_path=DEFAULT_DB,
             ollama_url="http://localhost:11434"):
    if not isinstance(message, str) or not message.strip() or message.startswith("/"):
        raise ValueError("Ожидается обычная непустая реплика")
    session = store.load_session(session_name)
    recent = session["turns"][-4:]
    for attempt in range(2):
        payload = prepare_payload(message, session["state"], recent, model,
                                  feedback=feedback if attempt else None)
        response = prepare(payload)
        try:
            state, query = validate_preparation(
                response.get("content") if isinstance(response, dict) else response,
                message, session["state"], session["version"] + 1)
            break
        except ValueError as exc:
            feedback = str(exc)
            if attempt:
                raise
    result = answer(message, search_query=query,
                    conversation_context=conversation_context(state, recent),
                    index_path=index_path, ollama_url=ollama_url, model=model)
    if result.get("status") not in {"answered", "no_evidence"}:
        raise ValueError("Некорректный статус ответа")
    if result["status"] == "answered" and (not result.get("sources") or not result.get("citations")):
        raise ValueError("Ответ без проверенных источников и цитат")
    store.commit_turn(session_name, session["version"], message, result, query, state)
    return result


def _print_state(state):
    goal = state["goal"]
    print("Цель: " + (f"{goal['text']} (реплика №{goal['turn']})" if goal else "нет"))
    for title, key in (("Уточнения", "clarifications"), ("Ограничения", "constraints"),
                       ("Термины", "terms")):
        print(title + ":")
        if not state[key]:
            print("  нет")
        for fact in state[key]:
            label = fact["term"] + " — " if key == "terms" else ""
            print(f"  {label}{fact['text']} (реплика №{fact['turn']})")


def _print_result(result):
    print("Ответ: " + result["answer"])
    if result.get("sources"):
        print("Источники:")
        for source in result["sources"]:
            print(f"  {source['source']} / {source['section']} ({source['chunk_id']})")
    else:
        print("Источники: нет")
    if result.get("citations"):
        print("Цитаты:")
        for citation in result["citations"]:
            print(f"  [{citation['ref']}] {citation['quote']}")
    else:
        print("Цитаты: нет")


def main(argv=None):
    parser = argparse.ArgumentParser(description="День 25: RAG-чат с памятью задачи")
    sub = parser.add_subparsers(dest="command", required=True)
    chat = sub.add_parser("chat", help="начать или продолжить сессию")
    chat.add_argument("--session", required=True)
    chat.add_argument("--db", type=Path, default=DEFAULT_CHAT_DB)
    chat.add_argument("--index", type=Path, default=DEFAULT_DB)
    chat.add_argument("--ollama-url", default="http://localhost:11434")
    chat.add_argument("--model", default=DEFAULT_CHAT_MODEL)
    args = parser.parse_args(argv)
    store = ChatStore(args.db)
    print(f"Сессия: {args.session}. Команды: /state, /history, /exit")
    while True:
        try:
            message = input("> ").strip()
        except (EOFError, KeyboardInterrupt, StopIteration):
            print()
            break
        if message == "/exit":
            break
        if not message:
            continue
        try:
            if message == "/state":
                _print_state(store.load_session(args.session)["state"])
                continue
            if message == "/history":
                turns = store.load_session(args.session)["turns"]
                if not turns:
                    print("История: нет")
                for turn in turns:
                    print(f"Реплика №{turn['turn_no']}: {turn['user_text']}")
                    _print_result(turn["result"])
                continue
            if message.startswith("/"):
                print("Ошибка: Неизвестная команда")
                continue
            result = run_turn(store, args.session, message, model=args.model,
                              index_path=args.index, ollama_url=args.ollama_url)
        except Exception as exc:
            print(f"Ошибка: {exc}")
            continue
        _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
