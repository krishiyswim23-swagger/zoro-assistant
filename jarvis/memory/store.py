"""Stage 5: persistent memory.

Four kinds of memory, all backed by a single SQLite file so they survive
process restarts:

    facts         -- long-term memory: things explicitly worth remembering
    projects      -- named projects, each with its own running notes
    preferences   -- simple key/value settings (response style, etc.)
    conversation  -- an append-only log of every turn, for continuity/debugging

Short-term memory (the current conversation) still lives in AIBrain's
in-process history; this store is for what should outlive a single run.
"""

from __future__ import annotations

import datetime
import json
import sqlite3
from pathlib import Path


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class ProjectNotFoundError(Exception):
    pass


class MemoryStore:
    def __init__(self, db_path: str, brain_activity=None) -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._brain_activity = brain_activity
        self._init_schema()

    def _pulse(self) -> None:
        if self._brain_activity is not None:
            self._brain_activity.pulse("memory")

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                tags TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS project_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL REFERENCES projects(id),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS preferences (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS conversation_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # -- long-term facts --------------------------------------------------

    def remember(self, content: str, tags: list[str] | None = None) -> int:
        self._pulse()
        cur = self._conn.execute(
            "INSERT INTO facts (content, tags, created_at) VALUES (?, ?, ?)",
            (content, json.dumps(tags or []), _now()),
        )
        self._conn.commit()
        return cur.lastrowid

    def fact_count(self) -> int:
        self._pulse()
        return self._conn.execute("SELECT COUNT(*) AS c FROM facts").fetchone()["c"]

    def recall(self, query: str, limit: int = 5) -> list[dict]:
        self._pulse()
        terms = [t for t in query.lower().split() if t]
        rows = self._conn.execute("SELECT * FROM facts ORDER BY id DESC").fetchall()
        scored = []
        for row in rows:
            lowered = row["content"].lower()
            score = sum(1 for t in terms if t in lowered)
            if score > 0:
                scored.append((score, row))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            {"id": row["id"], "content": row["content"], "tags": json.loads(row["tags"])}
            for _, row in scored[:limit]
        ]

    # -- projects -----------------------------------------------------------

    def create_project(self, name: str) -> int:
        self._pulse()
        existing = self._conn.execute("SELECT id FROM projects WHERE name = ?", (name,)).fetchone()
        if existing:
            return existing["id"]
        cur = self._conn.execute(
            "INSERT INTO projects (name, created_at) VALUES (?, ?)", (name, _now())
        )
        self._conn.commit()
        return cur.lastrowid

    def _get_project_id(self, name: str) -> int:
        row = self._conn.execute("SELECT id FROM projects WHERE name = ?", (name,)).fetchone()
        if not row:
            raise ProjectNotFoundError(name)
        return row["id"]

    def add_project_note(self, project_name: str, note: str) -> int:
        self._pulse()
        project_id = self.create_project(project_name)  # convenience: create on first use
        cur = self._conn.execute(
            "INSERT INTO project_notes (project_id, content, created_at) VALUES (?, ?, ?)",
            (project_id, note, _now()),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_project_notes(self, project_name: str) -> list[dict]:
        self._pulse()
        project_id = self._get_project_id(project_name)
        rows = self._conn.execute(
            "SELECT * FROM project_notes WHERE project_id = ? ORDER BY id ASC", (project_id,)
        ).fetchall()
        return [{"content": row["content"], "created_at": row["created_at"]} for row in rows]

    def list_projects(self) -> list[str]:
        self._pulse()
        rows = self._conn.execute("SELECT name FROM projects ORDER BY name ASC").fetchall()
        return [row["name"] for row in rows]

    # -- preferences ----------------------------------------------------------

    def set_preference(self, key: str, value: str) -> None:
        self._pulse()
        self._conn.execute(
            "INSERT INTO preferences (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self._conn.commit()

    def get_preference(self, key: str) -> str | None:
        self._pulse()
        row = self._conn.execute("SELECT value FROM preferences WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def all_preferences(self) -> dict[str, str]:
        rows = self._conn.execute("SELECT key, value FROM preferences").fetchall()
        return {row["key"]: row["value"] for row in rows}

    # -- conversation log -------------------------------------------------

    def log_turn(self, role: str, content: str) -> None:
        self._conn.execute(
            "INSERT INTO conversation_log (role, content, created_at) VALUES (?, ?, ?)",
            (role, content, _now()),
        )
        self._conn.commit()
