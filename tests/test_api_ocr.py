"""
HTTP-level tests for scanned-PDF ingestion via background OCR (main.py's
/upload -> ingestion/ocr_jobs.py).

Same isolation convention as tests/test_api_integration.py: runs against
whatever DATABASE_URL points at, truncating tables around every test, and
is skipped when Postgres isn't reachable. OCR itself is faked in most tests
(fast, deterministic); one end-to-end test runs the real engine.
"""

import threading
import time

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

from fastapi.testclient import TestClient

import main
import retrieval.store as store_module
from ingestion import ocr_jobs
from ingestion.ocr import OcrStats, ocr_available

OCR_TEXT = "The quarterly vendor review is held in March. Acme Corp supplies Project Atlas hardware."


@pytest.fixture(autouse=True)
def _no_background_embeddings(monkeypatch):
    monkeypatch.setattr(store_module.MultiDocStore, "_trigger_embedding_precompute", lambda self, chunks: None)


def _truncate_tables():
    with pg.get_db() as db:
        db.execute("TRUNCATE chunks, documents, feedback_logs, chat_sessions")


@pytest.fixture()
def client(monkeypatch, tmp_path):
    # Persisted uploads go to a temp dir so tests never write to uploaded_docs/.
    monkeypatch.setattr(main, "UPLOADED_DOCS_DIR", str(tmp_path))
    monkeypatch.setattr(ocr_jobs, "UPLOADED_DOCS_DIR", str(tmp_path))
    main.limiter.reset()   # slowapi counters are process-global; /upload allows 10/min
    pg.init_db()
    _truncate_tables()
    with TestClient(main.app) as c:
        yield c
    _truncate_tables()


def _fake_ocr(text=OCR_TEXT, gate=None):
    def _ocr(content, filename, progress=None, **kwargs):
        if gate is not None:
            gate.wait(timeout=10)
        if progress:
            progress(1, 1)
        pages = [{"page": 1, "text": text}] if text else []
        return pages, OcrStats(pages_total=1, pages_processed=1, pages_with_text=len(pages))
    return _ocr


def _use_fake_ocr(monkeypatch, **kwargs):
    monkeypatch.setattr(ocr_jobs, "ocr_pdf", _fake_ocr(**kwargs))
    monkeypatch.setattr(ocr_jobs, "ocr_ready", lambda: True)


def _wait_for(client, predicate, timeout=30):
    deadline = time.time() + timeout
    body = None
    while time.time() < deadline:
        body = client.get("/docs-loaded").json()
        if predicate(body):
            return body
        time.sleep(0.1)
    raise AssertionError(f"condition not met within {timeout}s; last /docs-loaded: {body}")


def _upload_scan(client, pdf, name="scan.pdf"):
    return client.post("/upload", files={"file": (name, pdf, "application/pdf")}).json()


def _has_failed(body):
    return any(f.get("status") == "failed" for f in body["files"])


# ── lifecycle ───────────────────────────────────────────────────────────────────

def test_scanned_pdf_upload_returns_processing_then_becomes_searchable(client, monkeypatch, make_scanned_pdf):
    gate = threading.Event()
    _use_fake_ocr(monkeypatch, gate=gate)

    body = _upload_scan(client, make_scanned_pdf([["scan one"]]))
    assert body["status"] == "processing"
    assert body["chunk_count"] == 0 and body["file_id"]

    # While OCR is held open: listed as processing, but not counted or searchable.
    listed = client.get("/docs-loaded").json()
    assert listed["total_files"] == 0 and listed["total_chunks"] == 0
    assert [(f["filename"], f["status"]) for f in listed["files"]] == [("scan.pdf", "processing")]
    assert main.store.is_empty()
    assert pg.db_get_document(body["file_id"])["status"] == "processing"

    gate.set()
    done = _wait_for(client, lambda b: b["total_files"] == 1)
    assert done["files"][0]["status"] == "ready" and done["files"][0]["chunk_count"] >= 1
    assert pg.db_get_document(body["file_id"])["status"] == "ready"
    assert not main.store.is_empty()


@pytest.mark.skipif(not ocr_available(), reason="OCR dependencies not installed")
def test_scanned_pdf_is_really_ocrd_end_to_end(client, make_scanned_pdf):
    body = _upload_scan(client, make_scanned_pdf([["Vendor: Acme Corp", "Renewal date: March 2027"]]))
    assert body["status"] == "processing"

    done = _wait_for(client, lambda b: b["total_files"] == 1, timeout=90)
    assert done["total_chunks"] >= 1
    indexed = " ".join(c["text"] for c in main.store.get_all_chunks()).lower()
    assert "acme" in indexed


def test_ocr_result_survives_a_restart(client, monkeypatch, make_scanned_pdf):
    _use_fake_ocr(monkeypatch)
    _upload_scan(client, make_scanned_pdf([["scan one"]]))
    _wait_for(client, lambda b: b["total_files"] == 1)

    with TestClient(main.app) as restarted:
        listed = restarted.get("/docs-loaded").json()
        assert listed["total_files"] == 1 and listed["files"][0]["status"] == "ready"


def test_interrupted_ocr_is_resumed_on_restart(client, monkeypatch, make_scanned_pdf):
    """A 'processing' row plus its persisted upload (what a killed server leaves) finishes after restart."""
    gate = threading.Event()   # never opened for the first "process": it never finishes
    _use_fake_ocr(monkeypatch, gate=gate)
    body = _upload_scan(client, make_scanned_pdf([["scan one"]]))
    assert pg.db_get_document(body["file_id"])["status"] == "processing"

    # The old process "dies": its in-flight bookkeeping is gone and its lease
    # stops being renewed. A fresh app instance (new worker id) whose OCR now
    # works must take the job over once the lease has expired.
    ocr_jobs._running.clear()
    monkeypatch.setattr(ocr_jobs, "WORKER_ID", "replacement-worker")
    monkeypatch.setattr(ocr_jobs, "ocr_pdf", _fake_ocr())
    with pg.get_db() as db:
        db.execute("UPDATE documents SET ocr_heartbeat = now() - interval '1 hour'")
    with TestClient(main.app) as restarted:
        listed = _wait_for(restarted, lambda b: b["total_files"] == 1)
        assert listed["files"][0]["status"] == "ready"
    gate.set()


# ── failure / unavailable ────────────────────────────────────────────────────────

def test_scanned_pdf_without_ocr_returns_a_clear_error(client, monkeypatch, make_scanned_pdf):
    monkeypatch.setattr(ocr_jobs, "ocr_ready", lambda: False)
    body = _upload_scan(client, make_scanned_pdf([["scan one"]]))

    assert body["status"] == "error"
    assert "scanned pdf" in body["message"].lower() and "ocr" in body["message"].lower()
    assert client.get("/docs-loaded").json()["files"] == []


def test_ocr_that_finds_no_text_is_listed_as_failed(client, monkeypatch, make_scanned_pdf):
    _use_fake_ocr(monkeypatch, text="")
    body = _upload_scan(client, make_scanned_pdf([["scan one"]]))

    listed = _wait_for(client, _has_failed)
    assert "no readable text" in listed["files"][0]["message"].lower()
    assert listed["total_files"] == 0
    assert pg.db_get_document(body["file_id"])["status"] == "failed"


def test_reuploading_a_failed_scan_retries_instead_of_being_blocked(client, monkeypatch, make_scanned_pdf):
    pdf = make_scanned_pdf([["scan one"]])
    _use_fake_ocr(monkeypatch, text="")
    first = _upload_scan(client, pdf)
    _wait_for(client, _has_failed)

    monkeypatch.setattr(ocr_jobs, "ocr_pdf", _fake_ocr())   # now it works
    second = _upload_scan(client, pdf)
    assert second["status"] == "processing" and second["file_id"] != first["file_id"]

    done = _wait_for(client, lambda b: b["total_files"] == 1)
    assert [f["status"] for f in done["files"]] == ["ready"]   # the failed record was replaced


# ── duplicates / deletion ───────────────────────────────────────────────────────

def test_duplicate_upload_while_processing_is_rejected(client, monkeypatch, make_scanned_pdf):
    gate = threading.Event()
    _use_fake_ocr(monkeypatch, gate=gate)
    pdf = make_scanned_pdf([["scan one"]])

    first = _upload_scan(client, pdf)
    second = _upload_scan(client, pdf, name="copy.pdf")
    gate.set()

    assert second["status"] == "duplicate" and second["file_id"] == first["file_id"]
    _wait_for(client, lambda b: b["total_files"] == 1)


def test_deleting_a_processing_document_cancels_it(client, monkeypatch, make_scanned_pdf):
    gate = threading.Event()
    _use_fake_ocr(monkeypatch, gate=gate)
    body = _upload_scan(client, make_scanned_pdf([["scan one"]]))

    resp = client.delete(f"/files/{body['file_id']}")
    assert resp.status_code == 200 and resp.json()["deleted"] is True
    gate.set()
    time.sleep(0.5)   # let the orphaned job finish and discard its result

    assert client.get("/docs-loaded").json() == {"files": [], "total_files": 0, "total_chunks": 0}
    assert main.store.is_empty()
    assert pg.db_get_document(body["file_id"]) is None


def test_clear_all_also_clears_processing_documents(client, monkeypatch, make_scanned_pdf):
    gate = threading.Event()
    _use_fake_ocr(monkeypatch, gate=gate)
    body = _upload_scan(client, make_scanned_pdf([["scan one"]]))

    assert client.delete("/documents/clear-all").json()["count"] >= 1
    gate.set()
    time.sleep(0.5)

    assert client.get("/docs-loaded").json()["files"] == []
    assert pg.db_get_document(body["file_id"]) is None
