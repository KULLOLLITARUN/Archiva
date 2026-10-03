"""
ingestion/ocr_jobs.py — Background OCR jobs for scanned PDFs.

/upload can't run OCR inline (a 50-page scan takes tens of seconds), so a
scanned PDF is accepted immediately as a *pending* document and finished
here:

    pending (store.pending + documents.status='processing')
        -> OCR on a dedicated worker thread          (ingestion/ocr.py)
        -> chunk + add to the store, sync to Postgres (same path as /upload)
        -> ready                                      (or 'failed' + reason)

Design notes
  * One worker thread: OCR is CPU-bound and the engine is shared, so jobs run
    one at a time in upload order rather than fighting each other (and the
    chat path) for cores.
  * Only the OCR itself runs off the event loop. Everything that mutates the
    store (add_file, pending bookkeeping) runs back on the event loop, the
    same thread /upload mutates it from, so the job adds no new concurrent-
    mutation risk to the in-memory retrieval structures.
  * Durable: the raw upload is persisted before the job starts and the
    'processing' state lives in Postgres, so resume_interrupted() can pick
    the work back up after a restart instead of silently losing it.
  * Multi-process safe: a job is owned through a lease row in Postgres
    (documents.ocr_worker / ocr_heartbeat). The owner renews it every page;
    resume_interrupted() runs on every worker every sync tick but can only
    claim a job whose lease is unowned or has expired, so a live job is
    never duplicated and a crashed worker's job is taken over. The raw
    upload is read from UPLOADED_DOCS_DIR, so workers must share that
    directory (true for several processes on one host).
  * Deleting a pending document - in this process OR another - cancels its
    job at the next page boundary (the progress callback raises
    OcrCancelled when the local record is gone or the lease renewal fails);
    a job that finishes after its document vanished discards its result.
"""

import asyncio
import functools
import os
import socket
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Set

from config import OCR_ENABLED, OCR_LEASE_SECONDS, UPLOADED_DOCS_DIR
from db import postgres as pg
from db.store_sync import set_document_status, sync_file_to_postgres, synchronizer
from ingestion.chunker import chunk_document
from ingestion.ocr import OcrCancelled, ocr_available, ocr_pdf
from ingestion.reingest import persisted_path

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr")

# Identifies this process as a lease owner (unique per process start).
WORKER_ID = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:6]}"

# file_ids with a job currently running in THIS process.
_running: Set[str] = set()

# Strong references so a running task can't be garbage-collected mid-flight
# (asyncio only keeps weak references to tasks).
_tasks: Set["asyncio.Task"] = set()


def ocr_ready() -> bool:
    """OCR is switched on AND its dependencies are importable."""
    return OCR_ENABLED and ocr_available()


def claim(file_id: str) -> bool:
    """Take the OCR lease for a 'processing' document (False: someone else owns it)."""
    return pg.db_claim_ocr_job(file_id, WORKER_ID, OCR_LEASE_SECONDS)


def submit(store, file_id: str, content: Optional[bytes] = None) -> "asyncio.Task":
    """
    Schedule the OCR job for a pending document THIS process already holds
    the lease for (see claim()). Must be called from the running event
    loop. *content* is the PDF's bytes; when omitted (recovery) they are
    read back from the persisted upload.
    """
    _running.add(file_id)
    task = asyncio.get_running_loop().create_task(_run_job(store, file_id, content))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return task


def resume_interrupted(store) -> int:
    """
    Take over 'processing' documents nobody is working on: left behind by a
    previous run (restart or crash mid-OCR) or by a worker whose lease
    expired. Safe to call on every worker repeatedly - the lease claim is
    atomic in Postgres, so only one worker wins each job, and a job with a
    live owner is left alone.

    Once THIS worker holds the lease, a document whose persisted upload is
    gone, or whose OCR dependencies are unavailable, is marked failed with
    a reason rather than left spinning forever. Returns how many jobs this
    call started.
    """
    resumed = 0
    for record in store.get_pending():
        file_id = record["file_id"]
        if record["status"] != "processing" or file_id in _running:
            continue
        if not claim(file_id):
            continue   # a live worker owns it (or it was just deleted)

        path = persisted_path(UPLOADED_DOCS_DIR, file_id, record["file_type"])
        if not os.path.exists(path):
            _fail(store, file_id, "Interrupted by a restart and the original file is no longer on disk. Please upload it again.")
        elif not ocr_ready():
            _fail(store, file_id, "OCR is unavailable (disabled or dependencies not installed).")
        else:
            submit(store, file_id)
            resumed += 1
    if resumed:
        print(f"[OCR] Resumed {resumed} interrupted OCR job(s)")
    return resumed


def _fail(store, file_id: str, message: str) -> None:
    """Record a terminal failure for a pending document (in memory and in Postgres)."""
    print(f"  [WARN]  [ocr] {file_id}: {message}")
    if file_id not in store.pending:
        return   # deleted while the job ran - nothing left to mark
    store.set_pending_state(file_id, "failed", message)
    try:
        set_document_status(file_id, "failed", message)
    except Exception as exc:
        print(f"  [WARN]  [ocr] Could not persist failed status for {file_id}: {exc}")


async def _run_job(store, file_id: str, content: Optional[bytes]) -> None:
    try:
        await _run_job_inner(store, file_id, content)
    finally:
        _running.discard(file_id)


async def _run_job_inner(store, file_id: str, content: Optional[bytes]) -> None:
    record = store.pending.get(file_id)
    if record is None:
        return
    filename, file_type = record["filename"], record["file_type"]
    loop = asyncio.get_running_loop()

    def on_progress(done: int, total: int) -> None:
        current = store.pending.get(file_id)
        if current is None:
            raise OcrCancelled()
        current["progress"] = (done, total)
        # Renew the lease. False = we no longer own the job: it was deleted
        # (possibly by another worker) or our lease lapsed and someone took
        # over. A database blip must not kill a long OCR run, so an error
        # here counts as "still ours".
        try:
            if not pg.db_touch_ocr_job(file_id, WORKER_ID):
                raise OcrCancelled()
        except OcrCancelled:
            raise
        except Exception as exc:
            print(f"  [WARN]  [ocr] Could not renew lease for {file_id}: {exc}")

    # ── 1. OCR (worker thread) ────────────────────────────────────────────────
    try:
        if content is None:
            with open(persisted_path(UPLOADED_DOCS_DIR, file_id, file_type), "rb") as f:
                content = f.read()
        pages, stats = await loop.run_in_executor(
            _executor, functools.partial(ocr_pdf, content, filename, on_progress)
        )
    except OcrCancelled:
        print(f"  [OCR]  {filename!r}: cancelled (document deleted)")
        return
    except Exception as exc:
        _fail(store, file_id, f"OCR failed: {exc}")
        return

    if file_id not in store.pending:
        return   # deleted after the last page finished

    # ── 2. Chunk + index (event loop; mirrors the /upload path) ────────────────
    try:
        chunks = chunk_document(pages=pages, file_id=file_id, filename=filename, file_type=file_type)
    except Exception as exc:
        _fail(store, file_id, f"Chunking failed after OCR: {exc}")
        return
    chunks = [c for c in chunks if c.get("text", "").strip()]

    if not chunks:
        _fail(store, file_id, f"OCR found no readable text ({stats.summary()}).")
        return

    # The document is searchable in memory from add_file() until it is
    # written to Postgres a few awaits later. Hold reloads off across that
    # gap so another worker's version bump can't rebuild the store from
    # Postgres in between and erase the not-yet-persisted document.
    with synchronizer.write_guard():
        _, add_status = store.add_file(
            file_id=file_id, filename=filename, content_hash=record["hash"],
            chunks=chunks, file_type=file_type,
        )
        if add_status != "ok":
            reason = "the store is full" if add_status == "limit" else "an identical document already exists"
            _fail(store, file_id, f"Could not index the OCR result: {reason}.")
            return

        store.remove_pending(file_id)
        print(f"  [OCR]  {filename!r}: indexed {len(chunks)} chunks ({stats.summary()})")

        # ── 3. Persist (thread: sync_file_to_postgres waits on embeddings) ──────
        try:
            await loop.run_in_executor(None, sync_file_to_postgres, store, file_id)
        except Exception as exc:
            # Searchable now, but the row is still 'processing' in Postgres, so a
            # restart re-runs the job - wasteful but safe, never lossy.
            print(f"  [WARN]  [ocr] Indexed {filename!r} but failed to persist: {exc}")
