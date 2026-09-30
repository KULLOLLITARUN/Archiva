-- db/schema.sql — Postgres schema for Archiva.
--
-- Consolidates what previously lived in three places (SQLite for document
-- metadata + feedback logs, an in-memory MultiDocStore pickled to disk for
-- chunks/embeddings) into one database.
--
-- No pgvector: embeddings are stored as a plain double-precision array and
-- similarity search stays application-side (numpy, brute-force cosine) —
-- same approach retrieval/dense.py already uses, just sourced from Postgres
-- instead of a pickle file. This was a deliberate choice, not a limitation
-- worked around: an ANN index isn't needed yet at the current chunk-count
-- scale (see MAX_TOTAL_CHUNKS in config.py's "Large-file guards" comment
-- for the same reasoning). Revisit with pgvector (or another ANN-capable
-- extension) if that cap is ever raised well past its current default —
-- the embedding column can be migrated to a vector(N) type at that point
-- without changing anything else in this schema.
--
-- Safe to run repeatedly — every statement is idempotent.

-- ── Documents ────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS documents (
    id           TEXT PRIMARY KEY,
    filename     TEXT NOT NULL,
    file_type    TEXT NOT NULL,
    chunk_count  INTEGER NOT NULL DEFAULT 0,
    content_hash TEXT,
    upload_time  TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_deleted   BOOLEAN NOT NULL DEFAULT false,
    deleted_at   TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_documents_deleted ON documents(is_deleted);

-- One active (non-deleted) document per content hash — mirrors the
-- duplicate-upload check MultiDocStore.file_hash_map already enforces
-- in-memory; this makes the same guarantee hold at the database level too.
CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_content_hash_active
    ON documents(content_hash)
    WHERE content_hash IS NOT NULL AND is_deleted = false;

-- ── Feedback / self-healing loop logs ──────────────────────────────────────────

CREATE TABLE IF NOT EXISTS feedback_logs (
    id           TEXT PRIMARY KEY,
    query        TEXT NOT NULL,
    failure_type TEXT,
    fix_applied  TEXT,
    success      BOOLEAN NOT NULL DEFAULT false,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── Chunks ──────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     TEXT PRIMARY KEY,
    file_id      TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    text         TEXT NOT NULL,
    metadata     JSONB NOT NULL,
    embedding    DOUBLE PRECISION[],
    content_hash TEXT
);

CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id);
CREATE INDEX IF NOT EXISTS idx_chunks_content_hash ON chunks(content_hash);

-- ── Document processing status (background OCR) ───────────────────────────────
-- 'ready'      - searchable; chunks live in the chunks table (the default, and
--                what every pre-existing row becomes)
-- 'processing' - accepted but not yet searchable: a scanned PDF whose OCR job
--                is queued/running (ingestion/ocr_jobs.py). No chunks yet.
-- 'failed'     - the background job could not produce usable text;
--                status_message says why. Kept (not deleted) so the user can
--                see the failure and remove it themselves.
-- A restart while 'processing' resumes the job from the persisted upload.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'ready';
ALTER TABLE documents ADD COLUMN IF NOT EXISTS status_message TEXT;

-- ── Multi-process coordination ─────────────────────────────────────────────────
-- Each worker process keeps its own in-memory copy of the document store
-- (chunks + BM25 + embeddings) for query speed. store_version is the single
-- counter every document-set mutation bumps (in the same transaction as the
-- change), so a worker can tell "someone changed the documents" with one
-- cheap query and reload only then. See db/store_sync.py's StoreSynchronizer.
CREATE TABLE IF NOT EXISTS store_version (
    id      INTEGER PRIMARY KEY CHECK (id = 1),
    version BIGINT NOT NULL DEFAULT 0
);
INSERT INTO store_version (id, version) VALUES (1, 0) ON CONFLICT (id) DO NOTHING;

-- Background OCR lease: which worker is processing a 'processing' document,
-- and when it last proved it was alive (once per page). A lease that stops
-- being renewed can be claimed by another worker, so a crashed worker's
-- job is picked up instead of hanging forever; a live one is never doubled.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS ocr_worker TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS ocr_heartbeat TIMESTAMPTZ;

-- ── Chat sessions ───────────────────────────────────────────────────────────────
-- Replaces sessions.json. That file was rewritten whole on every message, so
-- two processes silently overwrote each other's turns; a row per session
-- updated under a row lock cannot lose writes.
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id TEXT PRIMARY KEY,
    entries    JSONB NOT NULL DEFAULT '[]'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
