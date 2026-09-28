"""
chatbot/export.py — Render conversation history as a downloadable document.

Consumes chatbot.memory.ConversationMemory's history (List[MemoryEntry]) — the
same data the /chat endpoint stores after every turn — and formats it for a
user to save/share. Kept separate from memory.py: memory.py owns storage and
persistence, this module owns presentation.
"""

from datetime import datetime, timezone
from typing import List

from models.schemas import MemoryEntry


def _format_timestamp(raw: str) -> str:
    """Best-effort human-readable timestamp; falls back to the raw string
    if it isn't a valid ISO timestamp (e.g. missing/blank from old sessions)."""
    if not raw:
        return "unknown time"
    try:
        return datetime.fromisoformat(raw).strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return raw


def to_markdown(session_id: str, history: List[MemoryEntry]) -> str:
    """Render a session's conversation history as a Markdown document.

    Includes citations (filename/page) under each answer so the export is
    self-contained and traceable back to source documents, matching the
    grounding emphasis of the live chat UI.
    """
    lines = [
        f"# Conversation Export",
        "",
        f"Session: `{session_id}`",
        f"Exported: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "---",
        "",
    ]

    for i, entry in enumerate(history, start=1):
        lines.append(f"## {i}. {entry.query}")
        lines.append("")
        lines.append(f"*{_format_timestamp(entry.timestamp)}*")
        lines.append("")
        lines.append(entry.answer)
        lines.append("")

        if entry.sources:
            lines.append("**Sources:**")
            seen = set()
            for src in entry.sources:
                key = (src.filename, src.page)
                if key in seen:
                    continue
                seen.add(key)
                page_part = f", page {src.page}" if src.page else ""
                lines.append(f"- {src.filename}{page_part}")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)
