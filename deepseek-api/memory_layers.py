"""Explicit, bounded data layers for an agent's task and long-term memory."""
import json


MAX_WORKING_ITEMS = 24
MAX_LONG_TERM_ITEMS = 52
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
    text_commands = {"/working": "working", "/long": "long"}
    clear_commands = {
        "/clear-working": "clear-working", "/clear-long": "clear-long", "/clear-memory": "clear-memory",
    }
    if command not in {*text_commands, *clear_commands}:
        return None
    if command in text_commands:
        if not argument:
            raise MemoryCommandError(f"После {command} укажите текст для сохранения.")
        return {"action": text_commands[command], "text": argument}
    if argument:
        raise MemoryCommandError(f"Команда {command} не принимает дополнительный текст.")
    return {"action": clear_commands[command]}


def default_working():
    return []


def default_long_term():
    return []


def default_memory_layers():
    return {"working": default_working(), "longTerm": default_long_term()}


def _valid_nonempty_text(value, maximum):
    return isinstance(value, str) and bool(value.strip()) and len(value) <= maximum


def _valid_items(entries, maximum):
    return (
        isinstance(entries, list)
        and len(entries) <= maximum
        and all(_valid_nonempty_text(value, MAX_VALUE_LENGTH) for value in entries)
    )


def valid_working(value):
    return _valid_items(value, MAX_WORKING_ITEMS)


def valid_long_term(value):
    return _valid_items(value, MAX_LONG_TERM_ITEMS)


def valid_memory_layers(value):
    return (
        isinstance(value, dict)
        and set(value) == {"working", "longTerm"}
        and valid_working(value["working"])
        and valid_long_term(value["longTerm"])
    )


def working_has_content(working):
    return bool(working)


def long_term_has_content(long_term):
    return bool(long_term)


def stable_block(label, value):
    return f"[{label}]\n" + json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
