"""
chatbot/export.py — Render conversation history as a downloadable document.

Consumes chatbot.memory.ConversationMemory's history (List[MemoryEntry]) — the
same data the /chat endpoint stores after every turn — and formats it for a
user to save/share. Kept separate from memory.py: memory.py owns storage and
persistence, this module owns presentation.
"""

from datetime import datetime, timezone
from typing import List

from fpdf import FPDF
from fpdf.enums import XPos, YPos

from models.schemas import MemoryEntry

# fpdf2's core fonts (Helvetica/Times/Courier) only support Latin-1. Real LLM
# answers routinely contain Unicode punctuation the core fonts can't render
# (em/en dashes, curly quotes, narrow no-break spaces, full-width brackets
# around citations like "【Source: ...】") - map the common ones to a Latin-1
# equivalent instead of crashing or garbling the PDF, and let unicode_to_latin1()
# quietly drop what's left rather than letting them raise on encode.
_UNICODE_TO_LATIN1 = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "--", "…": "...", "•": "-",
    "‑": "-", " ": " ", " ": " ", " ": " ",
    "【": "[", "】": "]",
}


def _sanitize_for_pdf(text: str) -> str:
    for src, replacement in _UNICODE_TO_LATIN1.items():
        text = text.replace(src, replacement)
    return text.encode("latin-1", errors="replace").decode("latin-1")


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


def to_pdf(session_id: str, history: List[MemoryEntry]) -> bytes:
    """Render a session's conversation history as a PDF document.

    Same content/structure as to_markdown() (citations included, deduped by
    filename+page), rendered instead as a simple paginated document via
    fpdf2's core Helvetica font. See _sanitize_for_pdf() for why answer/query
    text is sanitized before rendering - that font can't display arbitrary
    Unicode, and real LLM output routinely contains it.
    """
    def _line(h: float, text: str) -> None:
        # multi_cell defaults to new_x=RIGHT, which leaves the cursor at the
        # right margin - the next width-0 multi_cell would then compute zero
        # available width and raise. Always reset to the left margin.
        pdf.multi_cell(0, h, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    _line(10, "Conversation Export")
    pdf.set_font("Helvetica", "", 10)
    _line(6, f"Session: {session_id}")
    _line(6, f"Exported: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    pdf.ln(4)

    for i, entry in enumerate(history, start=1):
        pdf.set_font("Helvetica", "B", 12)
        _line(8, _sanitize_for_pdf(f"{i}. {entry.query}"))

        pdf.set_font("Helvetica", "I", 9)
        _line(6, _sanitize_for_pdf(_format_timestamp(entry.timestamp)))
        pdf.ln(1)

        pdf.set_font("Helvetica", "", 11)
        _line(6, _sanitize_for_pdf(entry.answer))
        pdf.ln(1)

        if entry.sources:
            pdf.set_font("Helvetica", "B", 10)
            _line(6, "Sources:")
            pdf.set_font("Helvetica", "", 10)
            seen = set()
            for src in entry.sources:
                key = (src.filename, src.page)
                if key in seen:
                    continue
                seen.add(key)
                page_part = f", page {src.page}" if src.page else ""
                _line(6, _sanitize_for_pdf(f"- {src.filename}{page_part}"))

        pdf.ln(4)

    return bytes(pdf.output())
