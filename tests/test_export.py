"""Tests for chatbot/export.py — Markdown rendering of conversation history."""

from chatbot.export import to_markdown
from models.schemas import MemoryEntry, SourceRef


def _entry(query, answer, sources=None, timestamp="2026-01-01T12:00:00+00:00"):
    return MemoryEntry(
        query=query, answer=answer, intent="qa", timestamp=timestamp,
        sources=sources or [],
    )


def test_to_markdown_includes_session_id_and_all_turns():
    history = [
        _entry("What is X?", "X is Y."),
        _entry("What is Z?", "Z is W."),
    ]
    md = to_markdown("session-123", history)

    assert "session-123" in md
    assert "What is X?" in md
    assert "X is Y." in md
    assert "What is Z?" in md
    assert "Z is W." in md


def test_to_markdown_includes_source_citations():
    history = [
        _entry(
            "What is the policy?",
            "20 days per year.",
            sources=[SourceRef(filename="hr_policy.txt", page=1, text="...", score=0.9)],
        ),
    ]
    md = to_markdown("s1", history)

    assert "hr_policy.txt" in md
    assert "page 1" in md


def test_to_markdown_dedupes_repeated_source_citations():
    history = [
        _entry(
            "What are the parts?",
            "Four parts.",
            sources=[
                SourceRef(filename="10.pdf", page=12, text="...", score=0.9),
                SourceRef(filename="10.pdf", page=12, text="...", score=0.7),
                SourceRef(filename="10.pdf", page=9, text="...", score=0.5),
            ],
        ),
    ]
    md = to_markdown("s1", history)

    assert md.count("10.pdf, page 12") == 1
    assert "10.pdf, page 9" in md


def test_to_markdown_handles_empty_history():
    md = to_markdown("s1", [])
    assert "s1" in md
