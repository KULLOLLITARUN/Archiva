"""
db/store_sync.py — Loads/persists retrieval/store.py's MultiDocStore to and
from Postgres (db/postgres.py), replacing the old store_state.pkl file.

MultiDocStore itself stays persistence-agnostic (it doesn't import this
module or know Postgres exists) — this module is where "how do chunks get
from Postgres into the in-memory store, and back" lives, called from
main.py at the points that used to call pickle.load()/_save_store().
"""

import asyncio
import threading
from contextlib import contextmanager
from typing import Callable, List, Optional

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

        # Not searchable yet (queued/running OCR) or failed: register as
        # pending instead of as an active file with zero chunks, so the UI
        # can show its state and the OCR job can be resumed after a restart.
        if doc.get("status", "ready") != "ready":
            store.add_pending(
                file_id=file_id,
                filename=doc["filename"],
                content_hash=doc["content_hash"] or "",
                file_type=doc["file_type"],
                status=doc["status"],
                message=doc.get("status_message") or "",
                uploaded_at=_iso(doc.get("upload_time")),
            )
            continue

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
            uploaded_at=_iso(doc.get("upload_time")),
            ocr=bool(doc.get("ocr")),
        )

    return store


def _iso(ts) -> Optional[str]:
    """The stored upload time as ISO text (what add_file() records), or None."""
    return ts.isoformat() if ts is not None else None


def sync_file_to_postgres(store: MultiDocStore, file_id: str) -> None:
    """
    Persist the CURRENT in-memory state of one file's chunks to Postgres,
    replacing whatever was stored there before, in a single transaction
    (db_replace_document) that also bumps store_version - so other worker
    processes either see the document fully or not at all.

    Waits for the background embedding precompute thread first so
    embeddings aren't written as NULL - same race this module's
    docstring-referenced old _save_store() used to guard against for
    pickling, now guarding the Postgres write instead.
    """
    store.wait_for_embeddings()

    record = store.files.get(file_id)
    if record is None:
        return  # file was removed from the store between mutation and sync

    version = pg.db_replace_document(
        file_id, record["filename"], record["file_type"], record.get("hash"),
        store.get_file_chunks(file_id), ocr=record.get("ocr", False),
    )
    synchronizer.note_local_write(version)


def delete_file_from_postgres(file_id: str) -> None:
    """Remove a file's chunks and soft-delete its document row (one transaction)."""
    synchronizer.note_local_write(pg.db_delete_document(file_id))


def register_pending_document(file_id: str, filename: str, file_type: str, content_hash: str) -> None:
    """Record a not-yet-searchable document (queued for OCR) as 'processing'."""
    version = pg.db_upsert_document(
        file_id, filename, file_type, 0, content_hash=content_hash, status="processing",
    )
    synchronizer.note_local_write(version)


def set_document_status(file_id: str, status: str, message: Optional[str] = None) -> None:
    """Change a pending document's status (e.g. 'failed') and tell other workers."""
    synchronizer.note_local_write(pg.db_set_document_status(file_id, status, message))


# ── Multi-process synchronisation ────────────────────────────────────────────────

class StoreSynchronizer:
    """
    Keeps this process's in-memory MultiDocStore in step with Postgres when
    several worker processes share one database.

    Each process holds its own full copy of the store (retrieval stays a
    fast in-memory operation). Every mutation bumps Postgres's
    store_version in the same transaction as the change; this class's
    background task polls that one number and, when it differs from the
    version this process last loaded, rebuilds the store from Postgres and
    swaps it in (MultiDocStore.adopt).

    Correctness rules it enforces:
      * A process's own writes must not trigger a pointless reload:
        note_local_write() advances local_version when the bump is exactly
        the next number. If somebody else's bump interleaved, it does NOT
        advance, so the next poll reloads and picks their change up.
      * A reload is never applied over a local write that raced it. The
        snapshot is loaded off the event loop, so a local mutation can
        finish while it loads; every local write bumps an epoch, and a
        snapshot whose epoch is stale is discarded and retried next tick
        (it would otherwise erase the new local write until the following
        poll). write_guard() does the same for a local write that spans an
        await (in memory now, in Postgres later).
    """

    def __init__(self) -> None:
        self.store: Optional[MultiDocStore] = None
        self.local_version: int = 0
        self.reload_count: int = 0
        self.on_change: Optional[Callable[[], None]] = None   # documents changed (local or remote)
        self.on_tick: Optional[Callable[[], None]] = None     # runs on the loop every tick
        self._epoch = 0
        self._guard_depth = 0
        self._lock = threading.Lock()

    def attach(self, store: MultiDocStore, version: int, on_change=None, on_tick=None) -> None:
        self.store = store
        self.local_version = version
        self.on_change = on_change
        self.on_tick = on_tick
        self._epoch = 0
        self._guard_depth = 0

    def note_local_write(self, version: Optional[int] = None) -> None:
        """Call after a local Postgres write has committed (version = its bump)."""
        with self._lock:
            self._epoch += 1
            if version is not None and version == self.local_version + 1:
                self.local_version = version
        self._fire_on_change()

    @contextmanager
    def write_guard(self):
        """
        Hold reloads off across a local mutation that spans an await (the
        in-memory change and its Postgres write are not in the same
        synchronous stretch, so a reload could land in between).
        """
        with self._lock:
            self._guard_depth += 1
        try:
            yield
        finally:
            with self._lock:
                self._guard_depth -= 1
                self._epoch += 1

    def _fire_on_change(self) -> None:
        if self.on_change is not None:
            try:
                self.on_change()
            except Exception as exc:
                print(f"  [WARN]  [store-sync] on_change failed: {exc}")

    async def check_once(self) -> bool:
        """Reload from Postgres if another process changed the documents. True if reloaded."""
        if self.store is None or self._guard_depth:
            return False
        loop = asyncio.get_running_loop()
        epoch_before = self._epoch

        # Read the version BEFORE the data: if data lands newer than this
        # number, the next tick simply reloads once more (never misses a change).
        remote = await loop.run_in_executor(None, pg.db_get_store_version)
        if remote == self.local_version:
            return False
        fresh = await loop.run_in_executor(None, load_store_from_postgres)

        if self._guard_depth or self._epoch != epoch_before:
            return False   # a local write raced the load; retry next tick
        self.store.adopt(fresh)
        self.local_version = remote
        self.reload_count += 1
        print(f"[SYNC] Documents changed elsewhere - reloaded store "
              f"(v{remote}): {self.store.total_chunks()} chunks, {len(self.store.files)} file(s)")
        self._fire_on_change()
        return True

    async def run(self, interval: float) -> None:
        """Poll forever; a Postgres hiccup skips a tick rather than ending the task."""
        while True:
            await asyncio.sleep(interval)
            try:
                await self.check_once()
                if self.on_tick is not None:
                    self.on_tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"  [WARN]  [store-sync] tick failed: {exc}")


# One per process; main.py's lifespan attaches the live store to it.
synchronizer = StoreSynchronizer()
