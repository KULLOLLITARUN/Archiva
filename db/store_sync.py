"""
db/store_sync.py — Loads/persists retrieval/store.py's MultiDocStore to and
from Postgres (db/postgres.py), replacing the old store_state.pkl file.

MultiDocStore itself stays persistence-agnostic (it doesn't import this
module or know Postgres exists) — this module is where "how do chunks get
from Postgres into the in-memory store, and back" lives, called from
main.py at the points that used to call pickle.load()/_save_store().
"""

from typing import List, Optional

import numpy as np

from db import postgres as pg
from retrieval.store import MultiDocStore


def load_store_from_postgres() -> MultiDocStore:
    """
    Reconstruct an in-memory MultiDocStore from Postgres (documents +
    chunks), rebuilding the BM25 index. Called once at app startup —
    replaces the old store_state.pkl load.

    Reuses MultiDocStore.add_file() rather than poking at its internals
    directly, so hydration goes through the exact same file_hash_map /
    chunk_hash / BM25-rebuild logic a live upload does. Chunks already
    carry their embedding (loaded from Postgres), so
    precompute_embeddings() — triggered by add_file() as usual — treats
    them as already-done and is a no-op.

    No STORE_VERSION migration concern here (unlike the old pickle load):
    every call builds a fresh MultiDocStore from source data, never
    deserializes an old pickled instance.
    """
    store = MultiDocStore()

    for doc in pg.db_list_all_documents():
        if doc["is_deleted"]:
            continue

        file_id = doc["id"]
        chunks: List[dict] = []
        for row in pg.db_get_chunks_for_file(file_id):
            chunk = {
                "chunk_id": row["chunk_id"],
                "text": row["text"],
                "metadata": row["metadata"],
            }
            if row["embedding"] is not None:
                chunk["_embedding"] = np.array(row["embedding"])
            chunks.append(chunk)

        store.add_file(
            file_id=file_id,
            filename=doc["filename"],
            content_hash=doc["content_hash"] or "",
            chunks=chunks,
            file_type=doc["file_type"],
        )

    return store


def sync_file_to_postgres(store: MultiDocStore, file_id: str) -> None:
    """
    Persist the CURRENT in-memory state of one file's chunks to Postgres,
    replacing whatever was stored there before (delete + re-insert
    chunks; upsert the document row so this works whether the row is new
    — first upload — or already exists — /reload, reingestion refresh).

    Waits for the background embedding precompute thread first so
    embeddings aren't written as NULL — same race this module's
    docstring-referenced old _save_store() used to guard against for
    pickling, now guarding the Postgres write instead.
    """
    store.wait_for_embeddings()

    record = store.files.get(file_id)
    if record is None:
        return  # file was removed from the store between mutation and sync

    chunks = store.get_file_chunks(file_id)

    pg.db_upsert_document(
        file_id, record["filename"], record["file_type"], len(chunks),
        content_hash=record.get("hash"),
    )
    pg.db_delete_chunks_for_file(file_id)
    pg.db_add_chunks(file_id, chunks)


def delete_file_from_postgres(file_id: str) -> None:
    """Remove a file's chunks and soft-delete its document row."""
    pg.db_delete_chunks_for_file(file_id)
    pg.db_soft_delete_document(file_id)
