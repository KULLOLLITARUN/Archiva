"""
tests/test_postgres.py — Integration tests against a REAL local Postgres
instance (db/postgres.py).

Skipped automatically when Postgres isn't reachable at DATABASE_URL (e.g.
in CI, which has no Postgres service configured — this is the one part
of the suite that genuinely needs a live database, by design: db/postgres.py's
whole job is talking to one). Run locally with a real Postgres up to
exercise this file; everything else in the suite stays hermetic.

Uses the same database DATABASE_URL points at (this project runs single-
developer/local-only) — tables are truncated in an autouse fixture rather
than provisioning a second database, so tests never leak state into each
other or assume anything about what was there before.
"""

import numpy as np
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


@pytest.fixture(autouse=True)
def _clean_tables():
    pg.init_db()
    with pg.get_db() as db:
        db.execute("TRUNCATE chunks, documents, feedback_logs")
    yield
    with pg.get_db() as db:
        db.execute("TRUNCATE chunks, documents, feedback_logs")


# ── Schema / init ────────────────────────────────────────────────────────────────

def test_init_db_is_idempotent():
    pg.init_db()
    pg.init_db()  # must not raise on the second call


# ── Documents ─────────────────────────────────────────────────────────────────

def test_create_and_get_document_round_trips():
    pg.db_create_document("doc1", "report.pdf", "pdf", 5, content_hash="hash1")
    doc = pg.db_get_document("doc1")
    assert doc["filename"] == "report.pdf"
    assert doc["file_type"] == "pdf"
    assert doc["chunk_count"] == 5
    assert doc["content_hash"] == "hash1"
    assert doc["is_deleted"] is False


def test_get_document_by_content_hash():
    pg.db_create_document("doc1", "report.pdf", "pdf", 5, content_hash="hash1")
    found = pg.db_get_document_by_content_hash("hash1")
    assert found["id"] == "doc1"
    assert pg.db_get_document_by_content_hash("no-such-hash") is None


def test_duplicate_content_hash_on_active_documents_is_rejected():
    pg.db_create_document("doc1", "a.pdf", "pdf", 1, content_hash="dup-hash")
    with pytest.raises(psycopg.errors.UniqueViolation):
        pg.db_create_document("doc2", "b.pdf", "pdf", 1, content_hash="dup-hash")


def test_content_hash_reusable_after_soft_delete():
    pg.db_create_document("doc1", "a.pdf", "pdf", 1, content_hash="reused-hash")
    pg.db_soft_delete_document("doc1")
    # The unique index only applies to is_deleted=false rows, so a new
    # document with the same hash must be allowed once the old one is
    # soft-deleted (matches MultiDocStore's in-memory file_hash_map,
    # which frees the hash on delete_file()).
    pg.db_create_document("doc2", "b.pdf", "pdf", 1, content_hash="reused-hash")
    assert pg.db_get_document_by_content_hash("reused-hash")["id"] == "doc2"


def test_soft_delete_excludes_document_from_active_lookup():
    pg.db_create_document("doc1", "a.pdf", "pdf", 1)
    pg.db_soft_delete_document("doc1")
    assert pg.db_get_document("doc1") is None


def test_list_all_documents_includes_soft_deleted():
    pg.db_create_document("doc1", "a.pdf", "pdf", 1)
    pg.db_create_document("doc2", "b.pdf", "pdf", 1)
    pg.db_soft_delete_document("doc1")

    rows = pg.db_list_all_documents()
    assert {r["id"] for r in rows} == {"doc1", "doc2"}
    deleted_flags = {r["id"]: r["is_deleted"] for r in rows}
    assert deleted_flags["doc1"] is True
    assert deleted_flags["doc2"] is False


# ── Chunks ────────────────────────────────────────────────────────────────────

def _sample_chunks():
    return [
        {
            "chunk_id": "c1", "text": "hello world",
            "metadata": {"page": 1, "section": "Page 1", "parent_id": "p1"},
            "_embedding": np.array([0.1, 0.2, 0.3]),
        },
        {
            "chunk_id": "c2", "text": "second chunk",
            "metadata": {"page": 1, "section": "Page 1", "parent_id": "p1"},
            "_embedding": np.array([0.4, 0.5, 0.6]),
        },
    ]


def test_add_and_get_chunks_for_file_round_trips_embedding_and_metadata():
    pg.db_create_document("doc1", "a.pdf", "pdf", 2)
    pg.db_add_chunks("doc1", _sample_chunks())

    stored = pg.db_get_chunks_for_file("doc1")
    assert len(stored) == 2

    by_id = {c["chunk_id"]: c for c in stored}
    assert by_id["c1"]["text"] == "hello world"
    assert by_id["c1"]["metadata"] == {"page": 1, "section": "Page 1", "parent_id": "p1"}
    assert [round(v, 4) for v in by_id["c1"]["embedding"]] == [0.1, 0.2, 0.3]


def test_add_chunks_ignores_duplicate_chunk_ids():
    pg.db_create_document("doc1", "a.pdf", "pdf", 2)
    chunks = _sample_chunks()
    pg.db_add_chunks("doc1", chunks)
    pg.db_add_chunks("doc1", chunks)  # re-insert the same chunk_ids

    stored = pg.db_get_chunks_for_file("doc1")
    assert len(stored) == 2  # not 4 - ON CONFLICT DO NOTHING


def test_get_all_chunks_spans_multiple_files():
    pg.db_create_document("doc1", "a.pdf", "pdf", 2)
    pg.db_create_document("doc2", "b.pdf", "pdf", 2)
    pg.db_add_chunks("doc1", _sample_chunks())
    pg.db_add_chunks("doc2", [
        {"chunk_id": "c3", "text": "third", "metadata": {"page": 1}},
    ])

    all_chunks = pg.db_get_all_chunks()
    assert len(all_chunks) == 3


def test_delete_chunks_for_file_returns_count_and_removes_rows():
    pg.db_create_document("doc1", "a.pdf", "pdf", 2)
    pg.db_add_chunks("doc1", _sample_chunks())

    deleted = pg.db_delete_chunks_for_file("doc1")
    assert deleted == 2
    assert pg.db_get_chunks_for_file("doc1") == []


def test_chunks_without_embedding_are_stored_with_null_embedding():
    pg.db_create_document("doc1", "a.pdf", "pdf", 1)
    pg.db_add_chunks("doc1", [{"chunk_id": "c1", "text": "no embedding yet", "metadata": {}}])

    stored = pg.db_get_chunks_for_file("doc1")
    assert stored[0]["embedding"] is None


def test_hard_deleting_a_document_cascades_to_its_chunks():
    pg.db_create_document("doc1", "a.pdf", "pdf", 2)
    pg.db_add_chunks("doc1", _sample_chunks())

    with pg.get_db() as db:
        db.execute("DELETE FROM documents WHERE id = %s", ("doc1",))

    assert pg.db_get_chunks_for_file("doc1") == []


# ── Feedback / stats ─────────────────────────────────────────────────────────────

def test_log_feedback_and_system_stats():
    pg.db_create_document("doc1", "a.pdf", "pdf", 3)
    pg.db_log_feedback("fb1", "q1", "NONE", "NONE", True)
    pg.db_log_feedback("fb2", "q2", "HALLUCINATION", "STRICT_PROMPT", False)

    stats = pg.db_get_system_stats()
    assert stats["total_docs"] == 1
    assert stats["total_chunks"] == 3
    assert stats["total_queries"] == 2
    assert stats["success_rate"] == 50.0
