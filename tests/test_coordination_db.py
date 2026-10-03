"""
Real-Postgres tests for the multi-process coordination primitives in
db/postgres.py (store_version, atomic document replace, OCR leases, chat
sessions) and chatbot/memory.py's PostgresConversationMemory.

Same isolation convention as tests/test_postgres.py: runs against whatever
DATABASE_URL points at, truncating tables around every test, and is skipped
when Postgres isn't reachable.
"""

import json
import os
import threading

import pytest

psycopg = pytest.importorskip("psycopg")

from db import postgres as pg


def _postgres_available() -> bool:
    try:
        with pg.get_db():
            return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _postgres_available(), reason="Postgres not reachable at DATABASE_URL"
)

import chatbot.memory as memory_module
from chatbot.memory import PostgresConversationMemory
from models.schemas import SourceRef


@pytest.fixture(autouse=True)
def _clean():
    pg.init_db()
    with pg.get_db() as db:
        db.execute("TRUNCATE chunks, documents, feedback_logs, chat_sessions")
    yield
    with pg.get_db() as db:
        db.execute("TRUNCATE chunks, documents, feedback_logs, chat_sessions")


def _chunk(cid, text="some text"):
    return {"chunk_id": cid, "text": text, "metadata": {"content_hash": f"h-{cid}"}}


# ── store_version ───────────────────────────────────────────────────────────────

def test_every_document_mutation_bumps_the_version():
    v0 = pg.db_get_store_version()
    v1 = pg.db_replace_document("d1", "a.txt", "txt", "hash1", [_chunk("c1")])
    v2 = pg.db_set_document_status("d1", "failed", "because")
    v3 = pg.db_delete_document("d1")
    assert (v1, v2, v3) == (v0 + 1, v0 + 2, v0 + 3)
    assert pg.db_get_store_version() == v3


def test_upsert_and_create_and_soft_delete_bump_the_version():
    v0 = pg.db_get_store_version()
    assert pg.db_create_document("d1", "a.txt", "txt", 0, "h1") == v0 + 1
    assert pg.db_upsert_document("d2", "b.txt", "txt", 0, "h2", status="processing") == v0 + 2
    assert pg.db_soft_delete_document("d1") == v0 + 3


def test_reads_do_not_bump_the_version():
    pg.db_replace_document("d1", "a.txt", "txt", "h", [_chunk("c1")])
    v = pg.db_get_store_version()
    pg.db_list_all_documents(); pg.db_get_chunks_for_file("d1"); pg.db_get_document("d1")
    assert pg.db_get_store_version() == v


# ── db_replace_document ─────────────────────────────────────────────────────────

def test_replace_document_writes_row_and_chunks_together():
    pg.db_replace_document("d1", "a.txt", "txt", "h", [_chunk("c1"), _chunk("c2")])
    doc = pg.db_get_document("d1")
    assert doc["chunk_count"] == 2 and doc["status"] == "ready"
    assert {c["chunk_id"] for c in pg.db_get_chunks_for_file("d1")} == {"c1", "c2"}


def test_replace_document_replaces_previous_chunks():
    pg.db_replace_document("d1", "a.txt", "txt", "h", [_chunk("old1"), _chunk("old2")])
    pg.db_replace_document("d1", "a.txt", "txt", "h", [_chunk("new1")])
    assert [c["chunk_id"] for c in pg.db_get_chunks_for_file("d1")] == ["new1"]
    assert pg.db_get_document("d1")["chunk_count"] == 1


def test_replace_document_is_atomic_when_a_chunk_fails():
    pg.db_replace_document("d1", "a.txt", "txt", "h", [_chunk("keep")])
    v = pg.db_get_store_version()
    bad = [_chunk("x1"), {"chunk_id": "x2", "text": None, "metadata": {}}]   # NOT NULL violation

    with pytest.raises(Exception):
        pg.db_replace_document("d1", "a.txt", "txt", "h", bad)

    # Rolled back as a whole: old chunks intact, no version bump, no half-write.
    assert [c["chunk_id"] for c in pg.db_get_chunks_for_file("d1")] == ["keep"]
    assert pg.db_get_store_version() == v


def test_delete_document_removes_chunks_and_hides_the_row_in_one_step():
    pg.db_replace_document("d1", "a.txt", "txt", "h", [_chunk("c1")])
    pg.db_delete_document("d1")
    assert pg.db_get_chunks_for_file("d1") == []
    assert pg.db_get_document("d1") is None


# ── OCR lease ───────────────────────────────────────────────────────────────────

def _processing(doc_id="d1"):
    pg.db_upsert_document(doc_id, "scan.pdf", "pdf", 0, f"hash-{doc_id}", status="processing")


def test_unowned_job_can_be_claimed_once():
    _processing()
    assert pg.db_claim_ocr_job("d1", "worker-a", lease_s=60) is True
    assert pg.db_claim_ocr_job("d1", "worker-b", lease_s=60) is False      # live lease blocks b


def test_owner_can_reclaim_its_own_job():
    _processing()
    assert pg.db_claim_ocr_job("d1", "worker-a", 60) is True
    assert pg.db_claim_ocr_job("d1", "worker-a", 60) is True


def test_expired_lease_can_be_taken_over():
    _processing()
    pg.db_claim_ocr_job("d1", "worker-a", 60)
    with pg.get_db() as db:
        db.execute("UPDATE documents SET ocr_heartbeat = now() - interval '10 minutes'")
    assert pg.db_claim_ocr_job("d1", "worker-b", lease_s=60) is True
    assert pg.db_touch_ocr_job("d1", "worker-a") is False                  # a has lost it


def test_only_one_of_many_concurrent_claimers_wins():
    _processing()
    winners = []

    def attempt(i):
        if pg.db_claim_ocr_job("d1", f"worker-{i}", lease_s=60):
            winners.append(i)

    threads = [threading.Thread(target=attempt, args=(i,)) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(winners) == 1


def test_touch_renews_only_for_the_owner():
    _processing()
    pg.db_claim_ocr_job("d1", "worker-a", 60)
    assert pg.db_touch_ocr_job("d1", "worker-a") is True
    assert pg.db_touch_ocr_job("d1", "worker-b") is False


def test_touch_fails_after_the_document_is_deleted():
    _processing()
    pg.db_claim_ocr_job("d1", "worker-a", 60)
    pg.db_delete_document("d1")
    assert pg.db_touch_ocr_job("d1", "worker-a") is False      # cancels the job cross-process


def test_cannot_claim_a_document_that_is_not_processing():
    pg.db_replace_document("d1", "a.txt", "txt", "h", [_chunk("c1")])
    assert pg.db_claim_ocr_job("d1", "worker-a", 60) is False


def test_finishing_or_failing_releases_the_lease():
    _processing()
    pg.db_claim_ocr_job("d1", "worker-a", 60)
    pg.db_set_document_status("d1", "failed", "nope")
    with pg.get_db() as db:
        row = db.execute("SELECT ocr_worker, ocr_heartbeat FROM documents WHERE id = 'd1'").fetchone()
    assert row["ocr_worker"] is None and row["ocr_heartbeat"] is None


# ── chat sessions ───────────────────────────────────────────────────────────────

def _entry(i):
    return {"query": f"q{i}", "answer": f"a{i}", "intent": "qa", "timestamp": "t", "sources": []}


def test_session_entries_append_in_order_and_are_bounded():
    for i in range(15):
        pg.db_append_session_entry("s1", _entry(i), max_turns=10)
    entries = pg.db_get_session_entries("s1")
    assert [e["query"] for e in entries] == [f"q{i}" for i in range(5, 15)]


def test_concurrent_appends_from_many_workers_lose_no_turns():
    # The sessions.json failure mode: N writers, N turns, all must survive.
    errors = []

    def writer(i):
        try:
            pg.db_append_session_entry("shared", _entry(i), max_turns=100)
        except Exception as exc:   # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(20)]
    [t.start() for t in threads]
    [t.join() for t in threads]

    assert errors == []
    assert sorted(e["query"] for e in pg.db_get_session_entries("shared")) == sorted(f"q{i}" for i in range(20))


def test_unknown_session_is_empty_and_clear_works():
    assert pg.db_get_session_entries("nope") == []
    pg.db_append_session_entry("s1", _entry(1), 10)
    pg.db_clear_session("s1")
    assert pg.db_get_session_entries("s1") == []


def test_import_session_does_not_overwrite_an_existing_one():
    pg.db_append_session_entry("s1", _entry(1), 10)
    pg.db_import_session("s1", [_entry(99)])
    assert [e["query"] for e in pg.db_get_session_entries("s1")] == ["q1"]


# ── PostgresConversationMemory ─────────────────────────────────────────────────

def _src():
    return [SourceRef(filename="a.pdf", page=1, text="t", score=0.5)]


def test_memory_roundtrip_through_postgres():
    mem = PostgresConversationMemory()
    mem.add("s1", "what is x?", "x is 1", _src(), "qa")

    other_worker = PostgresConversationMemory()            # a different process's view
    last = other_worker.get_last("s1")
    assert last.query == "what is x?" and last.answer == "x is 1"
    assert last.sources[0].filename == "a.pdf"


def test_memory_keeps_only_the_last_max_turns():
    mem = PostgresConversationMemory()
    for i in range(mem.MAX_TURNS + 3):
        mem.add("s1", f"q{i}", f"a{i}", [], "qa")
    history = mem.get_history("s1")
    assert len(history) == mem.MAX_TURNS and history[-1].query == f"q{mem.MAX_TURNS + 2}"


def test_memory_clear_and_clear_all():
    mem = PostgresConversationMemory()
    mem.add("s1", "q", "a", [], "qa"); mem.add("s2", "q", "a", [], "qa")
    mem.clear("s1")
    assert mem.get_history("s1") == [] and len(mem.get_history("s2")) == 1
    mem.clear_all()
    assert mem.get_history("s2") == []


def test_memory_falls_back_to_process_memory_when_postgres_is_down(monkeypatch):
    mem = PostgresConversationMemory()

    def down(*a, **k):
        raise RuntimeError("postgres is down")

    monkeypatch.setattr(pg, "db_append_session_entry", down)
    monkeypatch.setattr(pg, "db_get_session_entries", down)
    mem.add("s1", "q", "a", [], "qa")                      # must not raise
    assert mem.get_last("s1").query == "q"                 # served from the fallback


def test_memory_imports_legacy_sessions_file_once():
    legacy = {"old-session": [_entry(1), _entry(2)]}
    path = memory_module._SESSIONS_FILE                    # redirected to a tmp path by conftest
    with open(path, "w", encoding="utf-8") as f:
        json.dump(legacy, f)

    mem = PostgresConversationMemory()
    assert [e.query for e in mem.get_history("old-session")] == ["q1", "q2"]
    assert not os.path.exists(path) and os.path.exists(path + ".migrated")

    # A second worker starting afterwards finds nothing to import and loses nothing.
    assert [e.query for e in PostgresConversationMemory().get_history("old-session")] == ["q1", "q2"]


def test_legacy_import_never_overwrites_newer_postgres_data():
    pg.db_append_session_entry("s1", _entry(50), 10)
    with open(memory_module._SESSIONS_FILE, "w", encoding="utf-8") as f:
        json.dump({"s1": [_entry(1)]}, f)
    PostgresConversationMemory()
    assert [e["query"] for e in pg.db_get_session_entries("s1")] == ["q50"]


def test_make_memory_selects_implementation_from_config(monkeypatch):
    monkeypatch.setattr(memory_module, "PERSIST_MEMORY", True)
    assert isinstance(memory_module.make_memory(), PostgresConversationMemory)
    monkeypatch.setattr(memory_module, "PERSIST_MEMORY", False)
    assert not isinstance(memory_module.make_memory(), PostgresConversationMemory)
