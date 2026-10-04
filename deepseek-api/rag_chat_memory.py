"""Bounded, evidence-checked task memory for the Day 25 chat."""

import copy
import json


MAX_FACTS = 20
MAX_CONTEXT = 35000


def empty_state():
    return {"goal": None, "clarifications": [], "constraints": [], "terms": []}


def _short(value, label):
    if not isinstance(value, str) or not value.strip() or len(value) > 300:
        raise ValueError(f"Некорректное поле {label}")
    return value.strip()


def validate_preparation(content, current_message, state, turn_number):
    """Apply a complete patch to a copy; every operation needs current-turn evidence."""
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Некорректный JSON памяти: {exc}") from exc
    if not isinstance(content, dict) or not isinstance(content.get("changes"), list):
        raise ValueError("Некорректная структура памяти")
    query = content.get("search_query")
    if not isinstance(query, str) or not 1 <= len(query.strip()) <= 500:
        raise ValueError("Некорректный поисковый запрос")
    if not isinstance(turn_number, int) or turn_number < 1:
        raise ValueError("Некорректный номер реплики")
    updated = copy.deepcopy(state)
    for change in content["changes"]:
        if not isinstance(change, dict):
            raise ValueError("Некорректное изменение памяти")
        op = change.get("op")
        if op not in {"set_goal", "add_clarification", "add_constraint", "set_term",
                      "remove_clarification", "remove_constraint", "remove_term"}:
            raise ValueError("Неизвестная операция памяти")
        evidence = _short(change.get("evidence"), "evidence")
        if evidence not in current_message:
            raise ValueError("Свидетельство отсутствует в текущей реплике")
        if op == "set_goal":
            text = _short(change.get("text"), "text")
            if updated["goal"] is None or updated["goal"]["text"] != text:
                updated["goal"] = {"text": text, "turn": turn_number}
            continue
        if op in {"set_term", "remove_term"}:
            term = _short(change.get("term"), "term")
            entries = updated["terms"]
            prior = next((item for item in entries if item["term"] == term), None)
            if op == "remove_term":
                if prior is None:
                    raise ValueError("Термин не найден")
                entries.remove(prior)
            else:
                text = _short(change.get("text"), "text")
                if prior is None:
                    entries.append({"term": term, "text": text, "turn": turn_number})
                elif prior["text"] != text:
                    prior.update(text=text, turn=turn_number)
            continue
        text = _short(change.get("text"), "text")
        key = "clarifications" if "clarification" in op else "constraints"
        entries = updated[key]
        prior = next((item for item in entries if item["text"] == text), None)
        if op.startswith("remove_"):
            if prior is None:
                raise ValueError("Факт для удаления не найден")
            entries.remove(prior)
        elif prior is None:
            entries.append({"text": text, "turn": turn_number})
    if any(len(updated[key]) > MAX_FACTS for key in ("clarifications", "constraints", "terms")):
        raise ValueError("Память переполнена")
    return updated, query.strip()


def conversation_context(state, recent_turns):
    """A bounded dialogue summary; previous answers are never book evidence."""
    summary = ["Состояние задачи (из реплик пользователя):"]
    goal = state.get("goal")
    summary.append("Цель: " + (f"{goal['text']} (№{goal['turn']})" if goal else "нет"))
    for key in ("clarifications", "constraints", "terms"):
        summary.append(key + ":")
        for fact in state.get(key, [])[-20:]:
            label = fact.get("term", "") + " = " if key == "terms" else ""
            summary.append(f"  {label}{fact['text']} (№{fact['turn']})")
    state_text = "\n".join(summary)
    lines = [state_text,
             "Предыдущие ответы — контекст диалога, не доказательство из книги."]
    for turn in recent_turns[-4:]:
        lines.append("Пользователь: " + str(turn["user_text"])[:450])
        lines.append("Ассистент: " + str(turn["result"]["answer"])[:450])
    value = "\n".join(lines)
    if len(value) > MAX_CONTEXT:
        raise ValueError("Контекст памяти превышает допустимый размер")
    return value


def prepare_payload(message, state, recent_turns, model, feedback=None):
    system = (
        "Подготовь следующий ход RAG-чата. Верни только JSON: "
        '{"search_query":"самостоятельный запрос по книге","changes":[]}. '
        "Разреши местоимения и ссылки на прежнюю тему в запросе. Не отвечай на вопрос. "
        "Извлекай цель, уточнения, ограничения и определения только из новой реплики. "
        "Каждое изменение содержит op, text (для терминов также term), evidence — "
        "дословный непустой фрагмент новой реплики. Исправления удаляют прежний факт "
        "и добавляют новый. Допустимые op: set_goal, add_clarification, "
        "add_constraint, set_term, remove_clarification, remove_constraint, remove_term. "
        "Не выдумывай изменения, если пользователь только задаёт вопрос."
    )
    user = conversation_context(state, recent_turns) + "\nТекущая реплика: " + message
    if feedback:
        user += "\nИсправь ошибку формата: " + feedback
    return {"model": model, "temperature": 0, "messages": [
        {"role": "system", "content": system}, {"role": "user", "content": user}]}
