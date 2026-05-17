"""Conversation history (SQLite) + long-term facts (JSON).

Conversations live in ``memory/conversations.db`` — one SQLite file holds
the index and all messages. On first startup, any legacy JSON files in
``memory/conversations/*.json`` are migrated into the DB automatically.

Long-term facts are still kept as ``memory/long_term.json`` since the file
is small, human-readable, and rarely written.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

MEMORY_DIR = Path("memory")
DB_PATH = MEMORY_DIR / "conversations.db"
LONG_TERM_FILE = MEMORY_DIR / "long_term.json"
LEGACY_CONV_DIR = MEMORY_DIR / "conversations"


def _ensure_dir() -> None:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)


def _connect() -> sqlite3.Connection:
    """Open the SQLite DB, creating tables and migrating legacy files on first run."""
    _ensure_dir()
    is_new = not DB_PATH.exists()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS conversations(
            id          TEXT PRIMARY KEY,
            title       TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages(
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            seq             INTEGER NOT NULL,
            role            TEXT NOT NULL,
            content         TEXT NOT NULL,
            agent           TEXT,
            workflow        TEXT,           -- JSON, optional (state-trace snapshots)
            timestamp       TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_msg_conv_seq ON messages(conversation_id, seq);
        """
    )
    if is_new:
        _migrate_legacy_json(conn)
    return conn


def _migrate_legacy_json(conn: sqlite3.Connection) -> None:
    """One-time import of memory/conversations/*.json from the old layout."""
    if not LEGACY_CONV_DIR.exists():
        return
    for path in sorted(LEGACY_CONV_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        conv_id = data.get("id") or path.stem
        title = data.get("title") or "(imported)"
        created = data.get("created_at") or datetime.now().isoformat()
        updated = data.get("updated_at") or created
        conn.execute(
            "INSERT OR IGNORE INTO conversations(id,title,created_at,updated_at) VALUES (?,?,?,?)",
            (conv_id, title, created, updated),
        )
        for i, m in enumerate(data.get("messages", [])):
            workflow_json = (
                json.dumps(m.get("workflow") or [], default=str)
                if m.get("workflow") else None
            )
            conn.execute(
                "INSERT INTO messages(conversation_id,seq,role,content,agent,workflow,timestamp) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    conv_id, i, m.get("role", ""), m.get("content", ""),
                    m.get("agent", "") or "", workflow_json,
                    m.get("timestamp") or datetime.now().isoformat(),
                ),
            )
    conn.commit()


# ---------------------------------------------------------------------------
# Conversation Memory (SQLite-backed)
# ---------------------------------------------------------------------------

class ConversationMemory:
    """One chat session — title + ordered list of messages.

    The object is the in-memory view; ``save()`` flushes pending changes
    to SQLite. ``load(id)`` rehydrates from the DB.
    """

    def __init__(self, conversation_id: str | None = None):
        self.conversation_id = conversation_id or uuid.uuid4().hex[:8]
        self.messages: list[dict[str, Any]] = []
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        self.title = "New conversation"
        # How many messages are already in the DB — save() only inserts the tail.
        self._persisted_count = 0

    def add_message(
        self,
        role: str,
        content: str,
        agent: str = "",
        workflow: list[dict] | None = None,
    ) -> None:
        """Append a message. ``workflow`` (optional) is the list of state
        snapshots captured during the agent run that produced this message —
        saved so the UI can replay the workflow after a page refresh.
        """
        entry: dict[str, Any] = {
            "role": role,
            "content": content,
            "agent": agent,
            "timestamp": datetime.now().isoformat(),
        }
        if workflow:
            entry["workflow"] = workflow
        self.messages.append(entry)
        if role == "user" and self.title == "New conversation":
            # Fallback title from the first question; an LLM may replace it
            # with a better summary after the first reply (see main._generate_title).
            self.title = content[:60] + ("..." if len(content) > 60 else "")

    def get_messages(self) -> list[dict[str, Any]]:
        return self.messages

    def get_recent_context(self, max_messages: int = 20) -> list[dict[str, Any]]:
        return self.messages[-max_messages:]

    def save(self) -> Path:
        """Flush all unsaved messages and the (possibly updated) title."""
        self.updated_at = datetime.now().isoformat()
        conn = _connect()
        try:
            conn.execute(
                "INSERT INTO conversations(id,title,created_at,updated_at) VALUES (?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET title=excluded.title, updated_at=excluded.updated_at",
                (self.conversation_id, self.title, self.created_at, self.updated_at),
            )
            for i in range(self._persisted_count, len(self.messages)):
                m = self.messages[i]
                workflow_json = (
                    json.dumps(m.get("workflow") or [], default=str)
                    if m.get("workflow") else None
                )
                conn.execute(
                    "INSERT INTO messages(conversation_id,seq,role,content,agent,workflow,timestamp) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (
                        self.conversation_id, i, m["role"], m["content"],
                        m.get("agent", "") or "", workflow_json, m["timestamp"],
                    ),
                )
            self._persisted_count = len(self.messages)
            conn.commit()
        finally:
            conn.close()
        return DB_PATH

    @classmethod
    def load(cls, conversation_id: str) -> "ConversationMemory":
        """Rehydrate a conversation from the DB; return a fresh one if absent."""
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT title, created_at, updated_at FROM conversations WHERE id=?",
                (conversation_id,),
            ).fetchone()
            if not row:
                return cls(conversation_id)
            mem = cls(conversation_id)
            mem.title = row["title"]
            mem.created_at = row["created_at"]
            mem.updated_at = row["updated_at"]
            rows = conn.execute(
                "SELECT role, content, agent, workflow, timestamp FROM messages "
                "WHERE conversation_id=? ORDER BY seq",
                (conversation_id,),
            ).fetchall()
            for r in rows:
                entry = {
                    "role": r["role"],
                    "content": r["content"],
                    "agent": r["agent"] or "",
                    "timestamp": r["timestamp"],
                }
                if r["workflow"]:
                    try:
                        entry["workflow"] = json.loads(r["workflow"])
                    except Exception:
                        pass
                mem.messages.append(entry)
            mem._persisted_count = len(mem.messages)
        finally:
            conn.close()
        return mem


def list_conversations() -> list[dict[str, Any]]:
    """List all conversations, newest update first."""
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT c.id, c.title, c.created_at, c.updated_at,
                   COALESCE((SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id), 0) AS message_count
            FROM conversations c
            ORDER BY c.updated_at DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Long-Term Memory (still JSON — small, human-editable)
# ---------------------------------------------------------------------------

class LongTermMemory:
    """Persistent key-value memory for facts and preferences."""

    def __init__(self):
        _ensure_dir()
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if LONG_TERM_FILE.exists():
            try:
                return json.loads(LONG_TERM_FILE.read_text(encoding="utf-8"))
            except Exception:
                return {"facts": [], "preferences": {}}
        return {"facts": [], "preferences": {}}

    def _save(self) -> None:
        LONG_TERM_FILE.write_text(
            json.dumps(self._data, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )

    def add_fact(self, fact: str) -> None:
        entry = {"fact": fact, "timestamp": datetime.now().isoformat()}
        self._data.setdefault("facts", []).append(entry)
        self._data["facts"] = self._data["facts"][-100:]  # cap to 100 most recent
        self._save()

    def get_facts(self, limit: int = 20) -> list[str]:
        facts = self._data.get("facts", [])
        return [f["fact"] for f in facts[-limit:]]

    def set_preference(self, key: str, value: str) -> None:
        self._data.setdefault("preferences", {})[key] = value
        self._save()

    def get_preferences(self) -> dict[str, str]:
        return self._data.get("preferences", {})

    def get_context_string(self) -> str:
        parts = []
        facts = self.get_facts(10)
        if facts:
            parts.append("REMEMBERED FACTS:\n" + "\n".join(f"- {f}" for f in facts))
        prefs = self.get_preferences()
        if prefs:
            parts.append(
                "USER PREFERENCES:\n"
                + "\n".join(f"- {k}: {v}" for k, v in prefs.items())
            )
        return "\n\n".join(parts) if parts else ""

    def clear(self) -> None:
        self._data = {"facts": [], "preferences": {}}
        self._save()
