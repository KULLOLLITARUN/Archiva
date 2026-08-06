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
    ) -> None:
        entry = MemoryEntry(
            query=query,
            answer=answer,
            sources=sources,
            intent=intent,
            timestamp=datetime.now(timezone.utc).isoformat(),
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
