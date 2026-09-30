"""
db/postgres.py — Postgres connection and data-access layer for Archiva.

The persistence layer for document metadata, chunks (text + JSONB
metadata + embedding), and feedback logs — replaces the old SQLite
(database.py, now deleted) + pickled MultiDocStore (store_state.pkl)
combination entirely. See db/schema.sql's module docstring for why
there's no pgvector/ANN index on the embedding column.

MultiDocStore (retrieval/store.py) itself stays persistence-agnostic —
it doesn't import this module. db/store_sync.py is the bridge: it loads
a MultiDocStore from here at startup and syncs it back on every mutation
(upload, delete, reload, reingestion refresh). See main.py's lifespan
and the /upload, /files/{id}, /reload, /documents/clear-all endpoints
for where those calls happen.
"""

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, List, Optional
from urllib.parse import urlsplit, urlunsplit

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json

# No default connection string — a Postgres URL always bundles credentials,
# so unlike DB_PATH (just a filesystem path) there's no safe placeholder to
# fall back to. Must be set explicitly in .env; see .env.example.
DATABASE_URL = os.getenv("DATABASE_URL", "")

# psycopg.connect() has no default timeout — an unreachable host/port hangs
# for the OS's full TCP timeout (can be minutes), which would freeze a real
# request and make an availability check (see tests/test_postgres.py)
# pointless as a fast skip-check. Keep this bounded everywhere.
CONNECT_TIMEOUT_S = float(os.getenv("DATABASE_CONNECT_TIMEOUT", 5))

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def _masked_url(url: str = DATABASE_URL) -> str:
    """DATABASE_URL with the password redacted, safe to log."""
    parts = urlsplit(url)
    if parts.password:
        netloc = parts.netloc.replace(f":{parts.password}@", ":***@")
        parts = parts._replace(netloc=netloc)
    return urlunsplit(parts)


# ── Connection factory ────────────────────────────────────────────────────────

@contextmanager
def get_db() -> Generator[psycopg.Connection, None, None]:
    """Yield a connection (dict-row results) and auto-commit/rollback."""
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is not set. Add it to .env — see .env.example for the format."
        )
    conn = psycopg.connect(DATABASE_URL, row_factory=dict_row, connect_timeout=CONNECT_TIMEOUT_S)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """
    Apply db/schema.sql. Safe to call on every startup — every statement
    in the schema is idempotent (CREATE ... IF NOT EXISTS).
    """
    schema_sql = _SCHEMA_PATH.read_text(encoding="utf-8")
    with get_db() as db:
        with db.cursor() as cur:
            cur.execute(schema_sql)
    print(f"[DB] Postgres schema ready at {_masked_url()}")


# ── Document helpers ──────────────────────────────────────────────────────────

def _bump_version(db: psycopg.Connection) -> int:
    """
    Increment store_version inside the CALLER's transaction and return the
    new value. Every mutation that changes what the document store should
    contain calls this in the same transaction as the change, so a worker
    that sees the new version is guaranteed to also see the data.
    """
    return db.execute(
        "UPDATE store_version SET version = version + 1 WHERE id = 1 RETURNING version"
    ).fetchone()["version"]


def db_get_store_version() -> int:
    """Current document-set version (see db/schema.sql, 'Multi-process coordination')."""
    with get_db() as db:
        return db.execute("SELECT version FROM store_version WHERE id = 1").fetchone()["version"]


def db_create_document(
    doc_id: str, filename: str, file_type: str, chunk_count: int,
    content_hash: Optional[str] = None,
) -> int:
    """Insert a document row. Returns the new store_version."""
    with get_db() as db:
        db.execute(
            """INSERT INTO documents (id, filename, file_type, chunk_count, content_hash)
               VALUES (%s, %s, %s, %s, %s)""",
            (doc_id, filename, file_type, chunk_count, content_hash),
        )
        return _bump_version(db)


def db_upsert_document(
    doc_id: str, filename: str, file_type: str, chunk_count: int,
    content_hash: Optional[str] = None,
    status: str = "ready", status_message: Optional[str] = None,
) -> int:
    """
    Insert a document, or update it in place (and un-delete it) if a row
    with this id already exists. Used by the reingestion refresh path
    (ingestion/reingest.py via db/store_sync.py), where the document row
    already exists and only its chunk_count/content_hash may have
    changed — a plain db_create_document() would fail on the id conflict.

    *status* defaults to 'ready' (the row is searchable), so every existing
    caller keeps working unchanged; the OCR job path passes 'processing'
    when it reserves a row up front, then upserts again as 'ready' once
    the chunks exist.

    Returns the new store_version. Any OCR lease on the row is cleared:
    callers that want to own a 'processing' row claim it afterwards with
    db_claim_ocr_job().
    """
    with get_db() as db:
        db.execute(
            """INSERT INTO documents (id, filename, file_type, chunk_count, content_hash,
                                      status, status_message)
               VALUES (%s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (id) DO UPDATE SET
                   filename       = EXCLUDED.filename,
                   file_type      = EXCLUDED.file_type,
                   chunk_count    = EXCLUDED.chunk_count,
                   content_hash   = EXCLUDED.content_hash,
                   status         = EXCLUDED.status,
                   status_message = EXCLUDED.status_message,
                   ocr_worker     = NULL,
                   ocr_heartbeat  = NULL,
                   is_deleted     = false,
                   deleted_at     = NULL""",
            (doc_id, filename, file_type, chunk_count, content_hash, status, status_message),
        )
        return _bump_version(db)


def db_replace_document(
    doc_id: str, filename: str, file_type: str, content_hash: Optional[str],
    chunks: List[dict], status: str = "ready", status_message: Optional[str] = None,
) -> int:
    """
    Make the stored copy of a document exactly (row + chunks) what the
    caller holds, in ONE transaction, and bump store_version in that same
    transaction. Returns the new version.

    This replaces the old upsert-then-delete-chunks-then-insert-chunks
    sequence of three separate transactions: with several worker processes
    reloading from Postgres, a reader could land between those steps and
    load a document with missing chunks. Now it sees the old state or the
    new one, never a mixture.
    """
    with get_db() as db:
        db.execute(
            """INSERT INTO documents (id, filename, file_type, chunk_count, content_hash,
                                      status, status_message)
               VALUES (%s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (id) DO UPDATE SET
                   filename       = EXCLUDED.filename,
                   file_type      = EXCLUDED.file_type,
                   chunk_count    = EXCLUDED.chunk_count,
                   content_hash   = EXCLUDED.content_hash,
                   status         = EXCLUDED.status,
                   status_message = EXCLUDED.status_message,
                   ocr_worker     = NULL,
                   ocr_heartbeat  = NULL,
                   is_deleted     = false,
                   deleted_at     = NULL""",
            (doc_id, filename, file_type, len(chunks), content_hash, status, status_message),
        )
        db.execute("DELETE FROM chunks WHERE file_id = %s", (doc_id,))
        _insert_chunks(db, doc_id, chunks)
        return _bump_version(db)


def db_set_document_status(doc_id: str, status: str, status_message: Optional[str] = None) -> int:
    """
    Update only the processing status of an existing (non-deleted)
    document, releasing any OCR lease. Returns the new store_version.
    """
    with get_db() as db:
        db.execute(
            """UPDATE documents SET status = %s, status_message = %s,
                      ocr_worker = NULL, ocr_heartbeat = NULL
               WHERE id = %s AND is_deleted = false""",
            (status, status_message, doc_id),
        )
        return _bump_version(db)


# ── Background OCR lease ──────────────────────────────────────────────────────
# A 'processing' document is owned by at most one worker at a time. The owner
# renews its lease (ocr_heartbeat) once per page; a lease that has not been
# renewed for lease_s seconds can be taken over, so a crashed worker's job
# is picked up by another instead of hanging forever.

def db_claim_ocr_job(doc_id: str, worker_id: str, lease_s: float) -> bool:
    """Atomically take (or re-take) the lease on a 'processing' document."""
    with get_db() as db:
        row = db.execute(
            """UPDATE documents SET ocr_worker = %s, ocr_heartbeat = now()
               WHERE id = %s AND status = 'processing' AND is_deleted = false
                 AND (ocr_worker IS NULL OR ocr_worker = %s OR ocr_heartbeat IS NULL
                      OR ocr_heartbeat < now() - make_interval(secs => %s))
               RETURNING id""",
            (worker_id, doc_id, worker_id, lease_s),
        ).fetchone()
        return row is not None


def db_touch_ocr_job(doc_id: str, worker_id: str) -> bool:
    """
    Renew our lease. False means we no longer own it - the document was
    deleted, finished elsewhere, or our lease expired and another worker
    took over - and the caller must stop.
    """
    with get_db() as db:
        row = db.execute(
            """UPDATE documents SET ocr_heartbeat = now()
               WHERE id = %s AND ocr_worker = %s AND status = 'processing'
                 AND is_deleted = false
               RETURNING id""",
            (doc_id, worker_id),
        ).fetchone()
        return row is not None


def db_list_all_documents() -> List[dict]:
    with get_db() as db:
        return db.execute(
            """SELECT id, filename, file_type, chunk_count, content_hash,
                      upload_time, is_deleted, status, status_message
               FROM documents ORDER BY upload_time DESC"""
        ).fetchall()


def db_get_document(doc_id: str) -> Optional[dict]:
    with get_db() as db:
        return db.execute(
            "SELECT * FROM documents WHERE id = %s AND is_deleted = false", (doc_id,)
        ).fetchone()


def db_get_document_by_content_hash(content_hash: str) -> Optional[dict]:
    """Active (non-deleted) document with this content hash, if any —
    mirrors MultiDocStore.file_hash_map's duplicate-upload check."""
    with get_db() as db:
        return db.execute(
            """SELECT * FROM documents
               WHERE content_hash = %s AND is_deleted = false""",
            (content_hash,),
        ).fetchone()


def db_soft_delete_document(doc_id: str) -> int:
    """Soft-delete a document. Returns the new store_version."""
    with get_db() as db:
        db.execute(
            "UPDATE documents SET is_deleted = true, deleted_at = now() WHERE id = %s",
            (doc_id,),
        )
        return _bump_version(db)


def db_delete_document(doc_id: str) -> int:
    """
    Delete a document's chunks and soft-delete its row in ONE transaction,
    bumping store_version once. Returns the new version.
    """
    with get_db() as db:
        db.execute("DELETE FROM chunks WHERE file_id = %s", (doc_id,))
        db.execute(
            """UPDATE documents SET is_deleted = true, deleted_at = now(),
                      ocr_worker = NULL, ocr_heartbeat = NULL
               WHERE id = %s""",
            (doc_id,),
        )
        return _bump_version(db)


# ── Chunk helpers ─────────────────────────────────────────────────────────────

def db_add_chunks(file_id: str, chunks: List[dict]) -> None:
    """
    Bulk-insert chunks for a file. Each chunk dict is shaped like the
    ones ingestion/chunker.py produces: {chunk_id, text, metadata, ...}
    plus an optional "_embedding" key (a numpy array or list of floats —
    matches the key retrieval/dense.py's precompute_embeddings() sets).
    """
    with get_db() as db:
        _insert_chunks(db, file_id, chunks)
        if chunks:
            _bump_version(db)


def _insert_chunks(db: psycopg.Connection, file_id: str, chunks: List[dict]) -> None:
    with db.cursor() as cur:
        for chunk in chunks:
            embedding = chunk.get("_embedding")
            embedding_list = [float(x) for x in embedding] if embedding is not None else None
            cur.execute(
                """INSERT INTO chunks (chunk_id, file_id, text, metadata, embedding, content_hash)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   ON CONFLICT (chunk_id) DO NOTHING""",
                (
                    chunk["chunk_id"], file_id, chunk["text"],
                    Json(chunk.get("metadata", {})), embedding_list,
                    chunk.get("metadata", {}).get("content_hash"),
                ),
            )


def db_get_chunks_for_file(file_id: str) -> List[dict]:
    with get_db() as db:
        return db.execute(
            "SELECT * FROM chunks WHERE file_id = %s", (file_id,)
        ).fetchall()


def db_get_all_chunks() -> List[dict]:
    with get_db() as db:
        return db.execute("SELECT * FROM chunks").fetchall()


def db_delete_chunks_for_file(file_id: str) -> int:
    """Returns the number of chunks deleted."""
    with get_db() as db:
        cur = db.execute("DELETE FROM chunks WHERE file_id = %s", (file_id,))
        deleted = cur.rowcount
        if deleted:
            _bump_version(db)
        return deleted


# ── Feedback log helpers ──────────────────────────────────────────────────────

def db_log_feedback(
    log_id: str, query: str, failure_type: str, fix_applied: str, success: bool,
) -> None:
    with get_db() as db:
        db.execute(
            """INSERT INTO feedback_logs (id, query, failure_type, fix_applied, success)
               VALUES (%s, %s, %s, %s, %s)""",
            (log_id, query, failure_type, fix_applied, success),
        )


# ── Stats helpers ──────────────────────────────────────────────────────────────

def db_get_system_stats() -> dict:
    with get_db() as db:
        total_docs = db.execute(
            "SELECT COUNT(*) AS n FROM documents WHERE is_deleted = false AND status = 'ready'"
        ).fetchone()["n"]
        total_chunks = db.execute(
            "SELECT COALESCE(SUM(chunk_count), 0) AS n FROM documents "
            "WHERE is_deleted = false AND status = 'ready'"
        ).fetchone()["n"]
        total_feedback = db.execute(
            "SELECT COUNT(*) AS n FROM feedback_logs"
        ).fetchone()["n"]
        success_rate = db.execute(
            "SELECT AVG(success::int) AS avg FROM feedback_logs"
        ).fetchone()["avg"]
    return {
        "total_docs":    total_docs,
        "total_chunks":  total_chunks,
        "total_queries": total_feedback,
        "success_rate":  round(float(success_rate or 0) * 100, 1),
    }


# ── Chat session helpers ────────────────────────────────────────────────────────

def db_append_session_entry(session_id: str, entry: dict, max_turns: int) -> None:
    """
    Append one turn to a session, keeping only the last *max_turns*.

    Read-modify-write under a row lock (SELECT ... FOR UPDATE), so two
    workers appending to the same session serialise instead of the second
    overwriting the first - the failure sessions.json had.
    """
    with get_db() as db:
        db.execute(
            "INSERT INTO chat_sessions (session_id) VALUES (%s) ON CONFLICT (session_id) DO NOTHING",
            (session_id,),
        )
        row = db.execute(
            "SELECT entries FROM chat_sessions WHERE session_id = %s FOR UPDATE", (session_id,)
        ).fetchone()
        entries = (row["entries"] or []) + [entry]
        db.execute(
            "UPDATE chat_sessions SET entries = %s, updated_at = now() WHERE session_id = %s",
            (Json(entries[-max_turns:]), session_id),
        )


def db_get_session_entries(session_id: str) -> List[dict]:
    with get_db() as db:
        row = db.execute(
            "SELECT entries FROM chat_sessions WHERE session_id = %s", (session_id,)
        ).fetchone()
        return list(row["entries"]) if row else []


def db_clear_session(session_id: str) -> None:
    with get_db() as db:
        db.execute("DELETE FROM chat_sessions WHERE session_id = %s", (session_id,))


def db_clear_all_sessions() -> None:
    with get_db() as db:
        db.execute("DELETE FROM chat_sessions")


def db_count_sessions() -> int:
    with get_db() as db:
        return db.execute("SELECT COUNT(*) AS n FROM chat_sessions").fetchone()["n"]


def db_import_session(session_id: str, entries: List[dict]) -> None:
    """Insert a whole session unless one already exists (used by the sessions.json import)."""
    with get_db() as db:
        db.execute(
            """INSERT INTO chat_sessions (session_id, entries) VALUES (%s, %s)
               ON CONFLICT (session_id) DO NOTHING""",
            (session_id, Json(entries)),
        )
