"""
db/postgres.py — Postgres connection and data-access layer for Archiva.

Additive at this point — NOT yet wired into main.py/database.py/
retrieval/store.py. Those still run on SQLite + a pickled MultiDocStore.
This module exists so the schema and repository functions are built and
tested against a real database before anything in the live app is
switched over to depend on them (that cutover is a separate, deliberate
step — see the module docstring in db/schema.sql for the reasoning on
why there's no pgvector/ANN index here yet).

Mirrors database.py's style (a connection-per-call context manager, one
function per query) so the two are easy to compare while both exist, and
easy to fold into one once the app cuts over.
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

def db_create_document(
    doc_id: str, filename: str, file_type: str, chunk_count: int,
    content_hash: Optional[str] = None,
) -> None:
    with get_db() as db:
        db.execute(
            """INSERT INTO documents (id, filename, file_type, chunk_count, content_hash)
               VALUES (%s, %s, %s, %s, %s)""",
            (doc_id, filename, file_type, chunk_count, content_hash),
        )


def db_list_all_documents() -> List[dict]:
    with get_db() as db:
        return db.execute(
            """SELECT id, filename, file_type, chunk_count, content_hash,
                      upload_time, is_deleted
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


def db_soft_delete_document(doc_id: str) -> None:
    with get_db() as db:
        db.execute(
            "UPDATE documents SET is_deleted = true, deleted_at = now() WHERE id = %s",
            (doc_id,),
        )


# ── Chunk helpers ─────────────────────────────────────────────────────────────

def db_add_chunks(file_id: str, chunks: List[dict]) -> None:
    """
    Bulk-insert chunks for a file. Each chunk dict is shaped like the
    ones ingestion/chunker.py produces: {chunk_id, text, metadata, ...}
    plus an optional "_embedding" key (a numpy array or list of floats —
    matches the key retrieval/dense.py's precompute_embeddings() sets).
    """
    with get_db() as db:
        with db.cursor() as cur:
            for chunk in chunks:
                embedding = chunk.get("_embedding")
                embedding_list = list(embedding) if embedding is not None else None
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
        return cur.rowcount


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
            "SELECT COUNT(*) AS n FROM documents WHERE is_deleted = false"
        ).fetchone()["n"]
        total_chunks = db.execute(
            "SELECT COALESCE(SUM(chunk_count), 0) AS n FROM documents WHERE is_deleted = false"
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
