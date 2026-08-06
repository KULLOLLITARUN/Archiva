"""
database.py — SQLite schema and helper layer for Archiva (open/no-auth edition).

Tables:
  documents     — file metadata (id, filename, upload_time, is_deleted)
  feedback_logs — self-healing loop result log (id, query, failure_type, ...)

Auth/user tables fully removed.
"""

import sqlite3
import os
from contextlib import contextmanager
from typing import Generator

DB_PATH = os.getenv("DB_PATH", "rag_system.db")


# ── Connection factory ────────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    """Create a thread-safe SQLite connection with row_factory."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row          # rows accessible as dicts
    conn.execute("PRAGMA journal_mode=WAL") # better concurrency
    return conn


@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    """Yield a connection and auto-commit/rollback."""
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Schema ────────────────────────────────────────────────────────────────────

_SCHEMA_SQL = """
-- ── Documents ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    filename    TEXT,
    file_type   TEXT,
    chunk_count INTEGER DEFAULT 0,
    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_deleted  INTEGER DEFAULT 0,
    deleted_at  TIMESTAMP
);

-- ── Feedback / self-healing loop logs ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS feedback_logs (
    id           TEXT PRIMARY KEY,
    query        TEXT,
    failure_type TEXT,
    fix_applied  TEXT,
    success      INTEGER DEFAULT 0,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ── Indices ───────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_docs_deleted ON documents(is_deleted);
"""


def init_db() -> None:
    """
    Create all tables and indices if they don't exist.
    Safe to call on every startup — idempotent.
    """
    with get_db() as db:
        db.executescript(_SCHEMA_SQL)
    print(f"[DB] SQLite ready at '{DB_PATH}'")


# ── Document helpers ──────────────────────────────────────────────────────────

def db_create_document(doc_id: str, filename: str,
                        file_type: str, chunk_count: int) -> None:
    with get_db() as db:
        cols = [c[1] for c in db.execute("PRAGMA table_info(documents)").fetchall()]
        if "user_id" in cols:
            db.execute(
                """INSERT INTO documents (id, user_id, filename, file_type, chunk_count)
                   VALUES (?,?,?,?,?)""",
                (doc_id, "admin", filename, file_type, chunk_count),
            )
        else:
            db.execute(
                """INSERT INTO documents (id, filename, file_type, chunk_count)
                   VALUES (?,?,?,?)""",
                (doc_id, filename, file_type, chunk_count),
            )



def db_list_all_documents() -> list:
    with get_db() as db:
        return db.execute(
            """SELECT id, filename, file_type, chunk_count, upload_time, is_deleted
               FROM documents ORDER BY upload_time DESC"""
        ).fetchall()


def db_get_document(doc_id: str) -> sqlite3.Row | None:
    with get_db() as db:
        return db.execute(
            "SELECT * FROM documents WHERE id=? AND is_deleted=0", (doc_id,)
        ).fetchone()


def db_soft_delete_document(doc_id: str) -> None:
    with get_db() as db:
        db.execute(
            "UPDATE documents SET is_deleted=1, deleted_at=CURRENT_TIMESTAMP WHERE id=?",
            (doc_id,),
        )


# ── Feedback log helpers ──────────────────────────────────────────────────────

def db_log_feedback(log_id: str, query: str,
                     failure_type: str, fix_applied: str, success: bool) -> None:
    with get_db() as db:
        db.execute(
            """INSERT INTO feedback_logs
               (id, query, failure_type, fix_applied, success)
               VALUES (?,?,?,?,?)""",
            (log_id, query, failure_type, fix_applied, int(success)),
        )


# ── Stats helpers ─────────────────────────────────────────────────────────────

def db_get_system_stats() -> dict:
    with get_db() as db:
        total_docs = db.execute(
            "SELECT COUNT(*) FROM documents WHERE is_deleted=0"
        ).fetchone()[0]
        total_chunks = db.execute(
            "SELECT SUM(chunk_count) FROM documents WHERE is_deleted=0"
        ).fetchone()[0] or 0
        total_feedback = db.execute(
            "SELECT COUNT(*) FROM feedback_logs"
        ).fetchone()[0]
        success_rate = db.execute(
            "SELECT AVG(success) FROM feedback_logs"
        ).fetchone()[0]
    return {
        "total_docs":    total_docs,
        "total_chunks":  total_chunks,
        "total_queries": total_feedback,
        "success_rate":  round((success_rate or 0) * 100, 1),
    }
