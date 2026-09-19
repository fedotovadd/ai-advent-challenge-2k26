"""Explicit, bounded data layers for an agent's task and long-term memory."""
import json


MAX_WORKING_FIELDS = 24
MAX_LONG_TERM_FIELDS = 16
MAX_DECISIONS = 20
MAX_TASK_LENGTH = 300
MAX_KEY_LENGTH = 80
MAX_VALUE_LENGTH = 500


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
