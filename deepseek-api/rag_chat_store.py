"""Transactional SQLite persistence for named RAG chat sessions."""

import json
from contextlib import contextmanager
from pathlib import Path
import sqlite3

from rag_chat_memory import empty_state


DEFAULT_CHAT_DB = Path(__file__).resolve().parent / "data" / "rag_chat.sqlite3"


class StaleSessionError(RuntimeError):
    """The session changed after it was read."""


class ChatStore:
    def __init__(self, path=DEFAULT_CHAT_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    name TEXT PRIMARY KEY, version INTEGER NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
                CREATE TABLE IF NOT EXISTS turns (
                    session_name TEXT NOT NULL REFERENCES sessions(name),
                    turn_no INTEGER NOT NULL, user_text TEXT NOT NULL,
                    result_json TEXT NOT NULL, search_query TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(session_name, turn_no));
            """)

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA busy_timeout=10000")
        try:
            with db:
                yield db
        finally:
            db.close()

    def load_session(self, name):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Имя сессии обязательно")
        with self._connect() as db:
            db.execute("INSERT OR IGNORE INTO sessions(name, version, state_json) VALUES (?, 0, ?)",
                       (name, json.dumps(empty_state(), ensure_ascii=False)))
            version, state_json = db.execute(
                "SELECT version, state_json FROM sessions WHERE name=?", (name,)).fetchone()
            rows = db.execute("SELECT turn_no, user_text, result_json, search_query, state_json "
                              "FROM turns WHERE session_name=? ORDER BY turn_no", (name,)).fetchall()
        turns = [{"turn_no": row[0], "user_text": row[1], "result": json.loads(row[2]),
                  "search_query": row[3], "state": json.loads(row[4])} for row in rows]
        return {"name": name, "version": version, "state": json.loads(state_json), "turns": turns}

    def commit_turn(self, name, expected_version, user_text, result, search_query, state):
        with self._connect() as db:
            try:
                db.execute("BEGIN IMMEDIATE")
                row = db.execute("SELECT version FROM sessions WHERE name=?", (name,)).fetchone()
                if row is None or row[0] != expected_version:
                    raise StaleSessionError("Сессия изменилась; повторите вопрос")
                turn_no = expected_version + 1
                db.execute("INSERT INTO turns(session_name, turn_no, user_text, result_json, "
                           "search_query, state_json) VALUES (?, ?, ?, ?, ?, ?)",
                           (name, turn_no, user_text, json.dumps(result, ensure_ascii=False),
                            search_query, json.dumps(state, ensure_ascii=False)))
                db.execute("UPDATE sessions SET version=?, state_json=?, updated_at=CURRENT_TIMESTAMP "
                           "WHERE name=?", (turn_no, json.dumps(state, ensure_ascii=False), name))
                db.commit()
            except BaseException:
                db.rollback()
                raise
        return turn_no
