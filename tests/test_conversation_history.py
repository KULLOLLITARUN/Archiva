"""Tests for conversation history: turns keep the checks their answer went
through, and /conversations lists, opens and deletes sessions. Uses the
in-process memory; the Postgres listing is covered in test_postgres.py."""

import asyncio

import pytest
from fastapi import HTTPException

import main
from chatbot import memory as memory_module
from chatbot.memory import ConversationMemory, _deserialize_entry, _serialize_entry
from models.schemas import MemoryEntry, SourceRef

CHECKS = {"attempts": 2, "reflection_reason": "passed_all_checks", "confidence": 0.81, "flagged": False}


@pytest.fixture()
def mem(monkeypatch):
    monkeypatch.setattr(memory_module, "PERSIST_MEMORY", False)
    m = ConversationMemory()
    monkeypatch.setattr(main, "memory", m)
    return m


def _add(m, sid, q, checks=None):
    m.add(sid, q, f"answer to {q}", [SourceRef(filename="a.pdf", page=1, text="t", score=1.0)], "qa", checks=checks)


def test_a_turn_keeps_its_checks_through_serialisation():
    entry = MemoryEntry(query="q", answer="a", sources=[], intent="qa", timestamp="t", checks=CHECKS)
    assert _deserialize_entry(_serialize_entry(entry)).checks == CHECKS


def test_turns_saved_before_checks_were_recorded_have_none():
    old = {"query": "q", "answer": "a", "intent": "qa", "timestamp": "t", "sources": []}
    assert _deserialize_entry(old).checks is None


def test_list_is_newest_first_and_titled_by_the_first_question(mem):
    _add(mem, "s1", "first in s1")
    _add(mem, "s2", "only in s2")
    _add(mem, "s1", "second in s1")
    rows = run(main.list_conversations())["conversations"]
    assert [(r["session_id"], r["title"], r["turns"]) for r in rows] == [
        ("s1", "first in s1", 2), ("s2", "only in s2", 1)]


def test_open_returns_turns_with_checks(mem):
    _add(mem, "s1", "q1", checks=CHECKS)
    body = run(main.get_conversation("s1"))
    assert body["turns"][0]["query"] == "q1"
    assert body["turns"][0]["checks"] == CHECKS


def test_open_unknown_conversation_is_404(mem):
    with pytest.raises(HTTPException) as exc:
        run(main.get_conversation("nope"))
    assert exc.value.status_code == 404


def run(coro):
    return asyncio.run(coro)


def test_a_provider_outage_is_not_kept_as_an_answer():
    # Saving "Service temporarily unavailable" would anchor the next follow-up to it.
    assert main._produced_answer({"reflection_reason": "provider_unavailable"}) is False
    assert main._produced_answer({"reflection_reason": "passed_all_checks"}) is True
    assert main._produced_answer({"reflection_reason": "explicit_not_found"}) is True
