"""Explicit, bounded data layers for an agent's task and long-term memory."""
import json


MAX_WORKING_FIELDS = 24
MAX_LONG_TERM_FIELDS = 16
MAX_DECISIONS = 20
MAX_TASK_LENGTH = 300
MAX_KEY_LENGTH = 80
MAX_VALUE_LENGTH = 500


class MemoryCommandError(ValueError):
    """Raised when a recognized memory command has an invalid form."""


def parse_memory_command(text):
    """Return a structured local memory command, or None for a chat message."""
    if not isinstance(text, str):
        return None
    parts = text.strip().split(maxsplit=1)
    if not parts:
        return None
    command, argument = parts[0], parts[1].strip() if len(parts) == 2 else ""
    text_commands = {"/working": "working", "/decision": "decision"}
    pair_commands = {
        "/working-data": "working-data", "/profile": "profile", "/knowledge": "knowledge",
    }
    clear_commands = {
        "/clear-working": "clear-working", "/clear-working-data": "clear-working-data",
        "/clear-profile": "clear-profile", "/clear-decisions": "clear-decisions",
        "/clear-knowledge": "clear-knowledge", "/clear-long": "clear-long", "/clear-memory": "clear-memory",
    }
    if command not in {*text_commands, *pair_commands, *clear_commands}:
        return None
    if command in text_commands:
        if not argument:
            raise MemoryCommandError(f"После {command} укажите текст для сохранения.")
        return {"action": text_commands[command], "text": argument}
    if command in pair_commands:
        key, separator, value = argument.partition(":")
        if not argument:
            raise MemoryCommandError(f"После {command} укажите текст для сохранения.")
        if not separator:
            return {"action": pair_commands[command], "value": argument}
        if not key.strip() or not value.strip():
            raise MemoryCommandError(f"После {command} укажите пару «ключ: значение».")
        return {"action": pair_commands[command], "key": key.strip(), "value": value.strip()}
    if argument:
        raise MemoryCommandError(f"Команда {command} не принимает дополнительный текст.")
    return {"action": clear_commands[command]}


def default_working():
    return {"task": "", "data": {}}


def default_long_term():
    return {"profile": {}, "decisions": [], "knowledge": {}}


def default_memory_layers():
    return {"working": default_working(), "longTerm": default_long_term()}


def _valid_nonempty_text(value, maximum):
    return isinstance(value, str) and bool(value.strip()) and len(value) <= maximum


def _valid_entries(entries, maximum):
    return (
        isinstance(entries, dict)
        and len(entries) <= maximum
        and all(
            _valid_nonempty_text(key, MAX_KEY_LENGTH)
            and _valid_nonempty_text(value, MAX_VALUE_LENGTH)
            for key, value in entries.items()
        )
    )


def valid_working(value):
    return (
        isinstance(value, dict)
        and set(value) == {"task", "data"}
        and isinstance(value["task"], str)
        and (not value["task"] or _valid_nonempty_text(value["task"], MAX_TASK_LENGTH))
        and _valid_entries(value["data"], MAX_WORKING_FIELDS)
    )


def valid_long_term(value):
    return (
        isinstance(value, dict)
        and set(value) == {"profile", "decisions", "knowledge"}
        and _valid_entries(value["profile"], MAX_LONG_TERM_FIELDS)
        and isinstance(value["decisions"], list)
        and len(value["decisions"]) <= MAX_DECISIONS
        and all(_valid_nonempty_text(item, MAX_VALUE_LENGTH) for item in value["decisions"])
        and _valid_entries(value["knowledge"], MAX_LONG_TERM_FIELDS)
    )


def valid_memory_layers(value):
    return (
        isinstance(value, dict)
        and set(value) == {"working", "longTerm"}
        and valid_working(value["working"])
        and valid_long_term(value["longTerm"])
    )


def working_has_content(working):
    return bool(working["task"] or working["data"])


def long_term_has_content(long_term):
    return bool(long_term["profile"] or long_term["decisions"] or long_term["knowledge"])


def stable_block(label, value):
    return f"[{label}]\n" + json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
