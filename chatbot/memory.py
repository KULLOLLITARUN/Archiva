"""
chatbot/memory.py — Per-session conversation history.

Fix #16: Added optional JSON persistence so session history survives
         server restarts.  Controlled by config.PERSIST_MEMORY (default True).

         Write path: atomic write-to-temp then os.replace() so a crash
         during write never corrupts the existing sessions file.

         MAX_TURNS is enforced before writing so the file stays bounded.

Multi-tenant upgrade (v4):
         Session keys are now namespaced by the caller as
             f"{user_id}::{session_id}"
         enforced in main.py (_memory_key helper).
         The memory class itself is key-agnostic — it stores whatever
         string key is passed to add() / get_history() / clear().
         This guarantees that two users with the same session_id value
         will NEVER share conversation context.

Multi-process (v5):
         PostgresConversationMemory keeps sessions in the chat_sessions table
         instead of sessions.json. The file was rewritten whole on every
         message, so with several worker processes each one's write silently
         erased the others' turns (and each kept a stale in-memory copy).
         Rows are appended under a row lock, so no turn is lost and any
         worker sees any session's latest history. make_memory() picks the
         implementation; ConversationMemory (file-backed) is kept as-is for
         the simple single-process case and for tests.
"""

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

from config import PERSIST_MEMORY
from models.schemas import MemoryEntry, SourceRef

# Fix #16: sessions are stored here when PERSIST_MEMORY is True
_SESSIONS_FILE     = "sessions.json"
_SESSIONS_FILE_TMP = "sessions.json.tmp"


def _serialize_entry(entry: MemoryEntry) -> dict:
    """Convert a MemoryEntry Pydantic model to a plain dict for JSON."""
    return {
        "query":     entry.query,
        "answer":    entry.answer,
        "intent":    entry.intent,
        "timestamp": entry.timestamp,
        "checks":    entry.checks,
        "sources": [
            {
                "filename": s.filename,
                "page":     s.page,
                "text":     s.text,
                "score":    s.score,
            }
            for s in entry.sources
        ],
    }


def _deserialize_entry(d: dict) -> MemoryEntry:
    """Restore a MemoryEntry from a plain dict that was loaded from JSON."""
    return MemoryEntry(
        query=d["query"],
        answer=d["answer"],
        intent=d.get("intent", "qa"),
        timestamp=d.get("timestamp", ""),
        checks=d.get("checks"),
        sources=[
            SourceRef(
                filename=s["filename"],
                page=s["page"],
                text=s["text"],
                score=s["score"],
            )
            for s in d.get("sources", [])
        ],
    )


class ConversationMemory:
    """Stores per-session conversation history (max 10 turns).

    Fix #16: When PERSIST_MEMORY=True (the default), sessions are written to
    sessions.json after every add() and loaded from disk on startup so history
    survives server restarts.
    """

    MAX_TURNS = 10

    def __init__(self):
        self.sessions: Dict[str, List[MemoryEntry]] = {}

        # Fix #16: load persisted data on startup
        if PERSIST_MEMORY:
            self._load_from_disk()

    # ── Persistence helpers (Fix #16) ─────────────────────────────────────────

    def _load_from_disk(self) -> None:
        """Load sessions from disk if the file exists."""
        if not os.path.exists(_SESSIONS_FILE):
            return
        try:
            with open(_SESSIONS_FILE, "r", encoding="utf-8") as f:
                raw: dict = json.load(f)
            self.sessions = {
                sid: [_deserialize_entry(e) for e in entries]
                for sid, entries in raw.items()
            }
            print(f"[OK] Loaded {len(self.sessions)} session(s) from {_SESSIONS_FILE}")
        except Exception as exc:
            print(f"  ⚠️  Could not load sessions from disk ({exc}) — starting fresh.")
            self.sessions = {}

    def _save_to_disk(self) -> None:
        """Atomically write current sessions to disk."""
        try:
            serializable = {
                sid: [_serialize_entry(e) for e in entries[-self.MAX_TURNS:]]
                for sid, entries in self.sessions.items()
            }
            with open(_SESSIONS_FILE_TMP, "w", encoding="utf-8") as f:
                json.dump(serializable, f, ensure_ascii=False, indent=2)
            os.replace(_SESSIONS_FILE_TMP, _SESSIONS_FILE)
        except Exception as exc:
            print(f"  ⚠️  Could not persist sessions ({exc})")

    # ── Public API ─────────────────────────────────────────────────────────────

    def add(
        self,
        session_id: str,
        query: str,
        answer: str,
        sources: List[SourceRef],
        intent: str,
        checks: Optional[dict] = None,
    ) -> None:
        entry = MemoryEntry(
            query=query,
            answer=answer,
            sources=sources,
            intent=intent,
            timestamp=datetime.now(timezone.utc).isoformat(),
            checks=checks,
        )
        history = self.sessions.setdefault(session_id, [])
        history.append(entry)
        # Enforce MAX_TURNS before persisting so the file stays bounded
        if len(history) > self.MAX_TURNS:
            self.sessions[session_id] = history[-self.MAX_TURNS:]

        # Fix #16: persist after every add when enabled
        if PERSIST_MEMORY:
            self._save_to_disk()

    def get_last(self, session_id: str) -> Optional[MemoryEntry]:
        history = self.sessions.get(session_id, [])
        return history[-1] if history else None

    def get_history(self, session_id: str) -> List[MemoryEntry]:
        return list(self.sessions.get(session_id, []))

    def clear(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)
        if PERSIST_MEMORY:
            self._save_to_disk()

    def list_sessions(self, limit: int = 30) -> List[dict]:
        """Most recent conversations first: id, first question, turn count, last update."""
        rows = [
            {"session_id": sid, "title": h[0].query, "turns": len(h), "updated_at": h[-1].timestamp}
            for sid, h in self.sessions.items() if h
        ]
        return sorted(rows, key=lambda r: r["updated_at"], reverse=True)[:limit]

    def clear_all(self) -> None:
        """Wipe all sessions from memory and disk."""
        self.sessions = {}
        if PERSIST_MEMORY:
            # Remove files rather than writing an empty JSON
            for path in (_SESSIONS_FILE, _SESSIONS_FILE_TMP):
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass


# ── Postgres-backed implementation (multi-process safe) ─────────────────────────

class PostgresConversationMemory(ConversationMemory):
    """
    Same interface as ConversationMemory, but every session lives in the
    chat_sessions table, so several worker processes share one consistent
    history.

    If Postgres is unreachable, calls fall back to an in-process dict for
    that call (with a warning) rather than failing the chat request - the
    conversation keeps working, just without cross-process sharing until
    the database is back.
    """

    def __init__(self):
        self.sessions: Dict[str, List[MemoryEntry]] = {}   # degraded-mode fallback only
        self._import_legacy_sessions_file()

    # ── One-time migration from sessions.json ──────────────────────────────────

    def _import_legacy_sessions_file(self) -> None:
        """
        Move sessions.json into Postgres once, then rename it so it is not
        re-imported. Existing rows win (ON CONFLICT DO NOTHING), and losing a
        race to another worker doing the same import is harmless.
        """
        if not os.path.exists(_SESSIONS_FILE):
            return
        try:
            from db import postgres as pg
            with open(_SESSIONS_FILE, "r", encoding="utf-8") as f:
                raw: dict = json.load(f)
            for sid, entries in raw.items():
                pg.db_import_session(sid, entries[-self.MAX_TURNS:])
            os.replace(_SESSIONS_FILE, _SESSIONS_FILE + ".migrated")
            print(f"[OK] Migrated {len(raw)} session(s) from {_SESSIONS_FILE} to Postgres")
        except Exception as exc:
            print(f"  [WARN]  Could not migrate {_SESSIONS_FILE} to Postgres ({exc}) - leaving it in place.")

    # ── Public API ─────────────────────────────────────────────────────────────

    def add(self, session_id, query, answer, sources, intent, checks=None) -> None:
        entry = MemoryEntry(
            query=query, answer=answer, sources=sources, intent=intent,
            timestamp=datetime.now(timezone.utc).isoformat(), checks=checks,
        )
        try:
            from db import postgres as pg
            pg.db_append_session_entry(session_id, _serialize_entry(entry), self.MAX_TURNS)
        except Exception as exc:
            print(f"  [WARN]  Could not persist session {session_id!r} to Postgres ({exc}) - kept in this process only.")
            history = self.sessions.setdefault(session_id, [])
            history.append(entry)
            self.sessions[session_id] = history[-self.MAX_TURNS:]

    def get_history(self, session_id: str) -> List[MemoryEntry]:
        try:
            from db import postgres as pg
            return [_deserialize_entry(e) for e in pg.db_get_session_entries(session_id)]
        except Exception as exc:
            print(f"  [WARN]  Could not read session {session_id!r} from Postgres ({exc}) - using this process's copy.")
            return list(self.sessions.get(session_id, []))

    def get_last(self, session_id: str) -> Optional[MemoryEntry]:
        history = self.get_history(session_id)
        return history[-1] if history else None

    def list_sessions(self, limit: int = 30) -> List[dict]:
        try:
            from db import postgres as pg
            return pg.db_list_sessions(limit)
        except Exception as exc:
            print(f"  [WARN]  Could not list sessions from Postgres ({exc}) - using this process's copy.")
            return super().list_sessions(limit)

    def clear(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)
        try:
            from db import postgres as pg
            pg.db_clear_session(session_id)
        except Exception as exc:
            print(f"  [WARN]  Could not clear session {session_id!r} in Postgres ({exc})")

    def clear_all(self) -> None:
        self.sessions = {}
        try:
            from db import postgres as pg
            pg.db_clear_all_sessions()
        except Exception as exc:
            print(f"  [WARN]  Could not clear sessions in Postgres ({exc})")


def make_memory() -> ConversationMemory:
    """
    Session store for the running app: Postgres-backed when PERSIST_MEMORY is
    on (the default; shared correctly across worker processes), otherwise
    plain in-process memory that vanishes on restart.
    """
    return PostgresConversationMemory() if PERSIST_MEMORY else ConversationMemory()
