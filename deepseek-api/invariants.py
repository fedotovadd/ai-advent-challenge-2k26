"""Persistent, explicit rules that constrain an agent's solutions."""

import re


MAX_INVARIANTS = 24
MAX_INVARIANT_LENGTH = 500


class InvariantCommandError(ValueError):
    """Raised for invalid local invariant commands or data."""


def normalize_invariant(value):
    if not isinstance(value, str):
        raise InvariantCommandError("Инвариант должен быть текстом.")
    normalized = re.sub(r"\s+", " ", value).strip()
    if not normalized:
        raise InvariantCommandError("После /invariant укажите правило.")
    if len(normalized) > MAX_INVARIANT_LENGTH:
        raise InvariantCommandError("Инвариант не должен быть длиннее 500 символов.")
    return normalized


def valid_invariants(value):
    if not isinstance(value, list) or len(value) > MAX_INVARIANTS:
        return False
    try:
        return (all(isinstance(item, str) and item == normalize_invariant(item) for item in value)
                and len({item.casefold() for item in value}) == len(value))
    except InvariantCommandError:
        return False


def _terms(value):
    return tuple(
        term for term in re.split(r"\s*(?:,|\s+и\s+)\s*", value.casefold()) if term
    )


def _constraints(invariants):
    required, banned, only = set(), set(), None
    for rule in invariants:
        normalized = normalize_invariant(rule).casefold()
        for prefix, kind in (("не использовать ", "banned"), ("без ", "banned"),
                             ("только ", "only"), ("использовать ", "required")):
            if normalized.startswith(prefix):
                values = set(_terms(normalized[len(prefix):]))
                if kind == "banned":
                    banned.update(values)
                elif kind == "only":
                    only = values
                else:
                    required.update(values)
                break
    return required, banned, only


def conflict_for_addition(invariants, value):
    current_required, current_banned, current_only = _constraints(invariants)
    candidate_required, candidate_banned, candidate_only = _constraints([value])
    required = current_required | candidate_required
    banned = current_banned | candidate_banned
    if required & banned:
        return True
    if current_only is not None and candidate_only is not None and current_only != candidate_only:
        return True
    only = candidate_only if candidate_only is not None else current_only
    return only is not None and (not required <= only or bool(banned & only))


def invariant_prompt_block(invariants):
    if not invariants:
        return ""
    items = "\n".join(f"- {item}" for item in invariants)
    return (
        "[ИНВАРИАНТЫ]\n"
        "Это правила высшего приоритета. Явно сверяй решение с каждым правилом; "
        "профиль, задача и память не могут их отменить.\n"
        f"{items}"
    )


_SOLUTION_ACTION = re.compile(r"\b(?:использ\w*|добав\w*|реализ\w*|перепиш\w*|созда\w*)\b", re.IGNORECASE)


def _mentioned_terms(text, terms):
    folded = text.casefold()
    return {term for term in terms if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", folded)}


def violations_for_solution(text, invariants):
    if not isinstance(text, str) or not _SOLUTION_ACTION.search(text):
        return []
    required, banned, only = _constraints(invariants)
    known = required | banned | (only or set())
    mentioned = _mentioned_terms(text, known)
    violations = []
    for rule in invariants:
        rule_required, rule_banned, rule_only = _constraints([rule])
        if rule_banned & mentioned:
            violations.append(rule)
        elif rule_required and not rule_required <= mentioned:
            violations.append(rule)
        elif rule_only is not None and mentioned - rule_only:
            violations.append(rule)
    return violations


def invariant_refusal(violations):
    rules = "; ".join(violations)
    return f"Не могу предложить это решение: оно нарушает инварианты: {rules}. Переформулируйте запрос в допустимых рамках."


def parse_invariant_command(text):
    if not isinstance(text, str):
        return None
    command, *tail = text.strip().split(maxsplit=1)
    argument = tail[0].strip() if tail else ""
    if command == "/invariant":
        return {"action": "add", "text": normalize_invariant(argument)}
    if command == "/invariants":
        if argument:
            raise InvariantCommandError("Команда /invariants не принимает дополнительный текст.")
        return {"action": "list"}
    if command == "/clear-invariants":
        if argument:
            raise InvariantCommandError("Команда /clear-invariants не принимает дополнительный текст.")
        return {"action": "clear"}
    if command == "/remove-invariant":
        if not argument or not argument.isascii() or not argument.isdigit() or int(argument) < 1:
            raise InvariantCommandError("После /remove-invariant укажите номер правила начиная с 1.")
        return {"action": "remove", "number": int(argument)}
    return None


def apply_invariant_command(invariants, command):
    if not valid_invariants(invariants):
        raise InvariantCommandError("Список инвариантов имеет неверный формат.")
    action = command.get("action") if isinstance(command, dict) else None
    result = list(invariants)
    if action == "add":
        value = normalize_invariant(command.get("text"))
        if value.casefold() in {item.casefold() for item in result}:
            raise InvariantCommandError("Такой инвариант уже добавлен.")
        if len(result) >= MAX_INVARIANTS:
            raise InvariantCommandError("Достигнут лимит из 24 инвариантов.")
        if conflict_for_addition(result, value):
            raise InvariantCommandError("Новый инвариант противоречит уже заданным правилам.")
        result.append(value)
        return {"invariants": result, "message": "Инвариант добавлен."}
    if action == "list":
        message = "Инварианты не заданы." if not result else "\n".join(
            f"{number}. {value}" for number, value in enumerate(result, 1)
        )
        return {"invariants": result, "message": message}
    if action == "remove":
        number = command.get("number")
        if not isinstance(number, int) or isinstance(number, bool) or not 1 <= number <= len(result):
            raise InvariantCommandError("Инвариант с таким номером не найден.")
        result.pop(number - 1)
        return {"invariants": result, "message": "Инвариант удалён."}
    if action == "clear":
        return {"invariants": [], "message": "Инварианты очищены."}
    raise InvariantCommandError("Неизвестная команда инвариантов.")
