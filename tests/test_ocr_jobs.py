"""Tests for ingestion/ocr_jobs.py: the pending -> ready / failed lifecycle,
cancellation, and restart recovery. OCR itself and Postgres are faked, so
these need neither the OCR engine nor a database."""

import asyncio

import pytest

import retrieval.store as store_module
from ingestion import ocr_jobs
from ingestion.ocr import OcrCancelled, OcrStats
from ingestion.reingest import persisted_path, save_uploaded_file
from retrieval.store import MultiDocStore


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(store_module.MultiDocStore, "_trigger_embedding_precompute", lambda self, chunks: None)
    monkeypatch.setattr(ocr_jobs, "UPLOADED_DOCS_DIR", str(tmp_path))
    monkeypatch.setattr(ocr_jobs, "ocr_ready", lambda: True)

    calls = {"status": [], "synced": [], "claims": [], "touches": []}
    owned = {"claim_ok": True, "touch_ok": True}
    calls["owned"] = owned
    monkeypatch.setattr(ocr_jobs, "set_document_status",
                        lambda fid, status, msg=None: calls["status"].append((fid, status, msg)))
    monkeypatch.setattr(ocr_jobs, "sync_file_to_postgres",
                        lambda store, fid: calls["synced"].append(fid))
    monkeypatch.setattr(ocr_jobs.pg, "db_claim_ocr_job",
                        lambda fid, worker, lease: calls["claims"].append(fid) or owned["claim_ok"])
    monkeypatch.setattr(ocr_jobs.pg, "db_touch_ocr_job",
                        lambda fid, worker: calls["touches"].append(fid) or owned["touch_ok"])
    return calls


def _pending_store(file_id="f1", filename="scan.pdf"):
    store = MultiDocStore()
    store.add_pending(file_id, filename, "hash-1", "pdf", message="Queued for OCR")
    return store


def _fake_ocr(pages, stats=None):
    stats = stats or OcrStats(pages_total=len(pages), pages_processed=len(pages), pages_with_text=len(pages))

    def _ocr(content, filename, progress=None, **kwargs):
        for i in range(len(pages)):
            if progress:
                progress(i + 1, len(pages))
        return pages, stats
    return _ocr


TEXT = "The quarterly vendor review is held in March. Acme Corp supplies Project Atlas hardware."


def run(coro):
    return asyncio.run(coro)


async def _submit_and_wait(store, file_id, content=b"%PDF"):
    await ocr_jobs.submit(store, file_id, content)


# ── success ────────────────────────────────────────────────────────────────────

def test_successful_job_moves_document_from_pending_to_indexed(monkeypatch, _isolate):
    monkeypatch.setattr(ocr_jobs, "ocr_pdf", _fake_ocr([{"page": 1, "text": TEXT}]))
    store = _pending_store()

    run(_submit_and_wait(store, "f1"))

    assert "f1" not in store.pending
    assert "f1" in store.files
    assert store.files["f1"]["status"] == "active"
    assert store.total_chunks() >= 1
    assert store.file_hash_map["hash-1"] == "f1"
    assert _isolate["synced"] == ["f1"]   # persisted as ready via the normal sync path


def test_progress_is_visible_on_the_pending_record_while_running(monkeypatch):
    store = _pending_store()
    seen = []

    def ocr(content, filename, progress=None, **kwargs):
        progress(1, 3)
        seen.append(store.pending["f1"]["progress"])
        progress(2, 3)
        seen.append(store.pending["f1"]["progress"])
        return [{"page": 1, "text": TEXT}], OcrStats(pages_total=3, pages_processed=3, pages_with_text=1)

    monkeypatch.setattr(ocr_jobs, "ocr_pdf", ocr)
    run(_submit_and_wait(store, "f1"))

    assert seen == [(1, 3), (2, 3)]


# ── failure ────────────────────────────────────────────────────────────────────

def test_job_with_no_readable_text_is_marked_failed_not_indexed(monkeypatch, _isolate):
    monkeypatch.setattr(ocr_jobs, "ocr_pdf", _fake_ocr([], OcrStats(pages_total=2, pages_processed=2)))
    store = _pending_store()

    run(_submit_and_wait(store, "f1"))

    assert store.pending["f1"]["status"] == "failed"
    assert "no readable text" in store.pending["f1"]["message"].lower()
    assert store.total_chunks() == 0 and "f1" not in store.files
    assert _isolate["status"][-1][:2] == ("f1", "failed")
    assert _isolate["synced"] == []


def test_ocr_exception_marks_the_document_failed(monkeypatch, _isolate):
    def boom(*a, **k):
        raise RuntimeError("engine exploded")

    monkeypatch.setattr(ocr_jobs, "ocr_pdf", boom)
    store = _pending_store()

    run(_submit_and_wait(store, "f1"))

    assert store.pending["f1"]["status"] == "failed"
    assert "engine exploded" in store.pending["f1"]["message"]
    assert _isolate["status"][-1][:2] == ("f1", "failed")


def test_store_full_marks_failed_instead_of_dropping_silently(monkeypatch):
    monkeypatch.setattr(ocr_jobs, "ocr_pdf", _fake_ocr([{"page": 1, "text": TEXT}]))
    monkeypatch.setattr(store_module, "MAX_TOTAL_CHUNKS", 0)
    store = _pending_store()

    run(_submit_and_wait(store, "f1"))

    assert store.pending["f1"]["status"] == "failed"
    assert "store is full" in store.pending["f1"]["message"]


# ── cancellation ───────────────────────────────────────────────────────────────

def test_deleting_the_pending_document_cancels_the_job(monkeypatch, _isolate):
    store = _pending_store()

    def ocr(content, filename, progress=None, **kwargs):
        store.remove_pending("f1")        # user deletes it mid-job
        progress(1, 5)                    # next page boundary -> OcrCancelled
        raise AssertionError("job should have been cancelled")

    monkeypatch.setattr(ocr_jobs, "ocr_pdf", ocr)
    run(_submit_and_wait(store, "f1"))

    assert store.pending == {} and store.files == {}
    assert _isolate["synced"] == [] and _isolate["status"] == []


def test_result_is_discarded_if_document_was_deleted_after_last_page(monkeypatch, _isolate):
    store = _pending_store()

    def ocr(content, filename, progress=None, **kwargs):
        store.remove_pending("f1")        # deleted after OCR finished, before commit
        return [{"page": 1, "text": TEXT}], OcrStats(pages_total=1, pages_processed=1, pages_with_text=1)

    monkeypatch.setattr(ocr_jobs, "ocr_pdf", ocr)
    run(_submit_and_wait(store, "f1"))

    assert store.files == {} and store.total_chunks() == 0
    assert _isolate["synced"] == []


# ── restart recovery ───────────────────────────────────────────────────────────

def test_resume_requeues_processing_documents_from_the_persisted_upload(monkeypatch, tmp_path, _isolate):
    save_uploaded_file(b"%PDF-bytes", "f1", "pdf", str(tmp_path))
    received = {}

    def ocr(content, filename, progress=None, **kwargs):
        received["content"] = content
        return [{"page": 1, "text": TEXT}], OcrStats(pages_total=1, pages_processed=1, pages_with_text=1)

    monkeypatch.setattr(ocr_jobs, "ocr_pdf", ocr)
    store = _pending_store()

    async def go():
        resumed = ocr_jobs.resume_interrupted(store)
        await asyncio.gather(*list(ocr_jobs._tasks))
        return resumed

    assert run(go()) == 1
    assert received["content"] == b"%PDF-bytes"
    assert "f1" in store.files and "f1" not in store.pending


def test_resume_fails_documents_whose_upload_is_gone(_isolate):
    store = _pending_store()   # nothing was persisted to disk

    async def go():
        return ocr_jobs.resume_interrupted(store)

    assert run(go()) == 0
    assert store.pending["f1"]["status"] == "failed"
    assert "upload it again" in store.pending["f1"]["message"]
    assert _isolate["status"][-1][:2] == ("f1", "failed")


def test_resume_fails_documents_when_ocr_is_unavailable(monkeypatch, tmp_path, _isolate):
    save_uploaded_file(b"%PDF", "f1", "pdf", str(tmp_path))
    monkeypatch.setattr(ocr_jobs, "ocr_ready", lambda: False)
    store = _pending_store()

    async def go():
        return ocr_jobs.resume_interrupted(store)

    assert run(go()) == 0
    assert store.pending["f1"]["status"] == "failed"
    assert "unavailable" in store.pending["f1"]["message"].lower()


def test_resume_ignores_already_failed_documents(_isolate):
    store = _pending_store()
    store.set_pending_state("f1", "failed", "earlier failure")

    async def go():
        return ocr_jobs.resume_interrupted(store)

    assert run(go()) == 0
    assert store.pending["f1"]["message"] == "earlier failure"   # untouched
    assert _isolate["status"] == []


# ── multi-process leases ───────────────────────────────────────────────────────

def test_resume_leaves_jobs_owned_by_a_live_worker_alone(monkeypatch, tmp_path, _isolate):
    save_uploaded_file(b"%PDF", "f1", "pdf", str(tmp_path))
    _isolate["owned"]["claim_ok"] = False     # another worker holds a fresh lease
    monkeypatch.setattr(ocr_jobs, "ocr_pdf", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not run")))
    store = _pending_store()

    async def go():
        return ocr_jobs.resume_interrupted(store)

    assert run(go()) == 0
    assert store.pending["f1"]["status"] == "processing"   # untouched, not failed
    assert _isolate["status"] == []


def test_resume_does_not_restart_a_job_already_running_here(monkeypatch, tmp_path, _isolate):
    save_uploaded_file(b"%PDF", "f1", "pdf", str(tmp_path))
    store = _pending_store()
    ocr_jobs._running.add("f1")
    try:
        async def go():
            return ocr_jobs.resume_interrupted(store)
        assert run(go()) == 0
        assert _isolate["claims"] == []           # didn't even try to claim
    finally:
        ocr_jobs._running.discard("f1")


def test_resume_claims_before_starting_a_job(monkeypatch, tmp_path, _isolate):
    save_uploaded_file(b"%PDF", "f1", "pdf", str(tmp_path))
    monkeypatch.setattr(ocr_jobs, "ocr_pdf", _fake_ocr([{"page": 1, "text": TEXT}]))
    store = _pending_store()

    async def go():
        resumed = ocr_jobs.resume_interrupted(store)
        await asyncio.gather(*list(ocr_jobs._tasks))
        return resumed

    assert run(go()) == 1
    assert _isolate["claims"] == ["f1"]


def test_job_stops_when_the_lease_is_lost(monkeypatch, _isolate):
    # Another worker took the job over (our lease lapsed) or it was deleted
    # from a different process: the next renewal fails and OCR must stop.
    _isolate["owned"]["touch_ok"] = False
    monkeypatch.setattr(ocr_jobs, "ocr_pdf", _fake_ocr([{"page": 1, "text": TEXT}]))
    store = _pending_store()

    run(_submit_and_wait(store, "f1"))

    assert store.files == {} and store.total_chunks() == 0
    assert _isolate["synced"] == []
    assert _isolate["touches"] == ["f1"]


def test_database_error_while_renewing_does_not_kill_the_job(monkeypatch, _isolate):
    def blip(fid, worker):
        raise RuntimeError("connection reset")
    monkeypatch.setattr(ocr_jobs.pg, "db_touch_ocr_job", blip)
    monkeypatch.setattr(ocr_jobs, "ocr_pdf", _fake_ocr([{"page": 1, "text": TEXT}]))
    store = _pending_store()

    run(_submit_and_wait(store, "f1"))

    assert "f1" in store.files                 # finished despite the blip


def test_finished_job_is_no_longer_marked_running(monkeypatch):
    monkeypatch.setattr(ocr_jobs, "ocr_pdf", _fake_ocr([{"page": 1, "text": TEXT}]))
    store = _pending_store()
    run(_submit_and_wait(store, "f1"))
    assert "f1" not in ocr_jobs._running


# ── MultiDocStore pending bookkeeping ───────────────────────────────────────────

def test_pending_documents_are_invisible_to_retrieval_and_summaries():
    store = _pending_store()
    assert store.is_empty() and store.total_chunks() == 0
    assert store.get_files() == []
    assert "No documents" in store.get_files_summary()


def test_find_pending_by_hash():
    store = _pending_store()
    assert store.find_pending_by_hash("hash-1")["file_id"] == "f1"
    assert store.find_pending_by_hash("nope") is None
