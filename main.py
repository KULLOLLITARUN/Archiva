"""
main.py — Archiva FastAPI application (open / no-auth edition).

Auth fully removed. All endpoints are open — no login, register, or tokens.
"""

import asyncio
import json
import os
import re
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from starlette.requests import Request

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# ── Project imports ───────────────────────────────────────────────────────────
from config import (
    GROQ_FAST, GROQ_STRONG, GROQ_QWEN, UPLOADED_DOCS_DIR, REINGESTION_QUEUE_PATH,
    MAX_UPLOAD_BYTES, SUGGESTION_MIN_SCORE, SUGGESTION_CE_TOP_N, CHAT_REQUEST_TIMEOUT_S,
    STORE_SYNC_ENABLED, STORE_SYNC_INTERVAL_S, RATE_LIMIT_STORAGE_URI, WARM_UP_ON_START,
)
from db.postgres import (
    init_db,
    db_list_all_documents, db_get_document, db_get_store_version,
    db_log_feedback, db_get_system_stats,
)
from db.store_sync import (
    load_store_from_postgres, sync_file_to_postgres, delete_file_from_postgres,
    register_pending_document, synchronizer,
)
from models.schemas import (
    ChatRequest, ChatResponse, SourceRef,
    DeleteResponse, DocsLoadedResponse, HealthResponse,
    StatsResponse, UploadResponse,
)
from retrieval.store import MultiDocStore
from retrieval.warmup import warm_up_models
from ingestion.parser import parse_file, SUPPORTED_EXTENSIONS, compute_hash
from ingestion.chunker import chunk_document
from ingestion.reingest import save_uploaded_file, refresh_all_from_disk
from ingestion import ocr_jobs
from chatbot.memory import ConversationMemory, make_memory
from cache.semantic_cache import semantic_cache
from chatbot.export import to_markdown, to_pdf
from chatbot.normalizer import normalize
from chatbot.rewriter import rewrite
from chatbot.intent import detect_intent
from agents.safety import safety_check
from agents.worker import call_groq
from agents.validator import validate
from agents.loop import run_reflection_loop, build_labeled_context
from monitor.logger import log_pipeline, get_stats

# Counters live in RATE_LIMIT_STORAGE_URI (per-process memory by default; point
# it at Redis for one limit shared by all worker processes - see config.py).
limiter = Limiter(key_func=get_remote_address, default_limits=[], storage_uri=RATE_LIMIT_STORAGE_URI)

store = MultiDocStore()
memory = ConversationMemory()

# GET /suggestions is expensive — one LLM call plus up to 16 retrieval+rerank
# checks — but every browser reload and every upload/delete re-fetches it even
# when the loaded document set hasn't actually changed. Cache the last result
# keyed by a fingerprint of the loaded file IDs: any real doc-set change
# (upload, delete, clear-all, reload) changes the fingerprint and produces a
# natural cache miss, so there's no separate invalidation path to keep in
# sync across endpoints. Single slot, not an LRU map — this is a single-
# process, single-tenant app, so there's only ever one "current" doc set.
_suggestions_cache: dict = {"fingerprint": None, "result": None}


def _suggestions_fingerprint() -> str:
    return ",".join(sorted(store.files.keys()))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global store, memory

    # Apply the Postgres schema (idempotent)
    init_db()

    # Read the document-set version BEFORE loading the data, so a change
    # that lands mid-load is caught by the next sync tick, never missed.
    try:
        loaded_version = db_get_store_version()
    except Exception:
        loaded_version = 0

    # Load the store from Postgres — replaces the old store_state.pkl load.
    # No STORE_VERSION migration concern here: every call reconstructs a
    # fresh MultiDocStore from source data (documents + chunks), never
    # deserializes an old pickled instance, so there's nothing to be
    # version-mismatched with.
    try:
        store = load_store_from_postgres()
        print(f"[OK] Store loaded from Postgres: {store.total_chunks()} chunks across {len(store.files)} file(s)")
    except Exception as e:
        store = MultiDocStore()
        print(f"[Warn] Could not load store from Postgres ({e}). Starting empty.")

    # Multi-process: keep this worker's in-memory store in step with Postgres.
    # Each tick also lets this worker take over OCR jobs whose owner died.
    synchronizer.attach(
        store, loaded_version,
        on_change=semantic_cache.clear,   # cached answers may cite changed documents
        on_tick=lambda: ocr_jobs.resume_interrupted(store),
    )

    # Finish OCR jobs a previous run was killed in the middle of.
    try:
        ocr_jobs.resume_interrupted(store)
    except Exception as e:
        print(f"[Warn] Could not resume OCR jobs ({e}).")

    memory = make_memory()

    sync_task = (
        asyncio.create_task(synchronizer.run(STORE_SYNC_INTERVAL_S))
        if STORE_SYNC_ENABLED else None
    )
    # Fire-and-forget: the server takes requests immediately; a question that
    # arrives mid-warm-up waits on the models' own load lock (no double load).
    if WARM_UP_ON_START:
        asyncio.get_running_loop().run_in_executor(None, warm_up_models)

    try:
        yield
    finally:
        if sync_task is not None:
            sync_task.cancel()
            try:
                await sync_task
            except asyncio.CancelledError:
                pass


app = FastAPI(title="Archiva", version="5.0.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Shared helpers ────────────────────────────────────────────────────────────

# The model cites inline as "[Source: file.pdf, page 1]" (or 【...】).
_CITATION_RE = re.compile(r"[\[【]\s*Source:([^\]】]*)[\]】]", re.IGNORECASE)
# Models swap in look-alike characters when they copy a filename: non-breaking
# hyphens ("A4‑S1‑GST‑Invoice") and narrow/no-break spaces.
_DASH_LIKE_RE  = re.compile(r"[‐-―−]")
_SPACE_LIKE_RE = re.compile(r"[\s  ]+")


def _normalize_citation_text(text: str) -> str:
    return _SPACE_LIKE_RE.sub(" ", _DASH_LIKE_RE.sub("-", text)).lower()


def _cited_chunks(answer: str, chunks: list) -> list:
    """
    The chunks whose file the answer actually cites.

    Every chunk sent to the model used to be returned as a source, so an
    answer drawn from one invoice showed chips for two unrelated PDFs that
    merely sat in the context. An answer with no citation we can match keeps
    the full list, so a source is never silently lost.
    """
    citations = [_normalize_citation_text(c) for c in _CITATION_RE.findall(answer or "")]
    if not citations:
        return chunks
    cited = [
        c for c in chunks
        if any(_normalize_citation_text(c["metadata"]["filename"]) in cit for cit in citations)
    ]
    return cited or chunks


def _build_sources(chunks: list, answer: str = "") -> list:
    return [
        {
            "filename": c["metadata"]["filename"],
            "page":     c["metadata"]["page"],
            "text":     c["text"][:200],
            "score":    round(c["score"], 4),
        }
        for c in _cited_chunks(answer, chunks)
    ]


def _sample_for_suggestions(all_chunks: list, total: int = 30) -> list:
    """
    Select up to `total` chunks to seed the /suggestions prompt: balanced
    across files, and within each file spread evenly across its full length
    rather than just its first few chunks.

    The old version round-robinned files but walked each file's chunk list
    front-to-back, so for a small number of large documents it only ever
    sampled their opening chunks — a 40-page handbook's suggestions would
    skew entirely toward whatever's on page 1, never seeing its later
    sections. Picking evenly-spaced indices per file fixes that while
    keeping the same per-file balance.
    """
    by_file: dict = {}
    for c in all_chunks:
        fid = c["metadata"].get("file_id", "x")
        by_file.setdefault(fid, []).append(c)

    if not by_file:
        return []

    per_file = max(1, total // len(by_file))

    def _evenly_spaced(chunks: list, k: int) -> list:
        n = len(chunks)
        if n <= k:
            return chunks
        return [chunks[i * n // k] for i in range(k)]

    per_file_samples = [_evenly_spaced(chunks, per_file) for chunks in by_file.values()]

    # Round-robin interleave across files so no single document dominates
    # the sample just for having more chunks.
    interleaved = []
    iters = [iter(v) for v in per_file_samples]
    while iters:
        next_iters = []
        for it in iters:
            try:
                interleaved.append(next(it))
                next_iters.append(it)
            except StopIteration:
                pass
        iters = next_iters

    return interleaved[:total]


def _build_log_payload(request_id, query, safety, loop_result, intent, elapsed, flagged):
    return {
        "request_id":           request_id,
        "query":                query,
        "safety_decision":      safety["reason"],
        "retrieval_scores":     [c["score"] for c in loop_result.get("chunks", [])],
        "model_used":           loop_result.get("model_used", "none"),
        "intent":               intent,
        "latency_ms":           elapsed,
        "flagged":              flagged,
        "attempts":             loop_result.get("attempts", 1),
        "reflected":            loop_result.get("reflected", False),
        "reflection_reason":    loop_result.get("reflection_reason", "not_reflected"),
        "confidence":           loop_result.get("confidence", 1.0),
        "search_queries":       loop_result.get("search_queries", []),
        "retrieval_latency_ms": loop_result.get("retrieval_latency_ms", 0),
        "reranker_scores":      loop_result.get("reranker_scores", []),
        "failure_type":         loop_result.get("failure_type", "NONE"),
        "tokens_used":          loop_result.get("tokens_used", 0),
        "healing_action":       loop_result.get("healing_action", "NONE"),
    }


# ── Chat endpoints ────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
@limiter.limit("20/minute")
async def chat(
    request: Request,
    body: ChatRequest,
) -> ChatResponse:
    t_start    = time.time()
    request_id = str(uuid.uuid4())
    event_loop = asyncio.get_event_loop()

    raw_query = normalize(body.message)
    query     = rewrite(raw_query, memory, body.session_id)
    intent    = detect_intent(query)
    safety    = safety_check(query)

    if not safety["safe"]:
        elapsed = int((time.time() - t_start) * 1000)
        return ChatResponse(
            answer="Query not allowed. Please rephrase.",
            sources=[], intent=intent, model_used="none",
            latency_ms=elapsed, flagged=True,
        )

    if store.is_empty():
        elapsed = int((time.time() - t_start) * 1000)
        return ChatResponse(
            answer="No documents loaded yet. Upload a document first.",
            sources=[], intent=intent, model_used="none",
            latency_ms=elapsed, flagged=False,
        )

    try:
        loop_result = await asyncio.wait_for(
            event_loop.run_in_executor(None, run_reflection_loop, query, store, intent, None),
            timeout=CHAT_REQUEST_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        # Note: this bounds the HTTP response, not the underlying thread —
        # Python threads can't be forcibly killed, so the reflection loop
        # keeps running in the executor's thread pool until it naturally
        # finishes. This still achieves the goal (the caller isn't left
        # hanging, and a single stuck request can't hold this connection
        # open forever); it just isn't true cancellation.
        elapsed = int((time.time() - t_start) * 1000)
        return ChatResponse(
            answer="Request timed out. Please try again.",
            sources=[], intent=intent, model_used="none",
            latency_ms=elapsed, flagged=False,
        )

    answer     = loop_result["answer"]
    top_chunks = loop_result["chunks"]
    model_id   = loop_result["model_used"]

    validation = validate(answer, top_chunks)
    flagged    = validation["flagged"]
    sources    = [SourceRef(**s) for s in _build_sources(top_chunks, answer)]

    # Store the ORIGINAL query, not the rewritten/anchored one - otherwise
    # each follow-up's anchor text (which already embeds the prior question
    # and answer) would get baked into memory as "the question", and the
    # next follow-up would anchor to THAT, compounding wrapper text turn
    # over turn instead of resolving cleanly against what the user actually
    # asked each time.
    memory.add(body.session_id, raw_query, answer, sources, intent)
    elapsed = int((time.time() - t_start) * 1000)

    asyncio.create_task(_async_log_feedback(
        query,
        loop_result.get("failure_type", "NONE"),
        loop_result.get("healing_action", "NONE"),
        not flagged,
    ))
    asyncio.create_task(log_pipeline(
        _build_log_payload(request_id, query, safety, loop_result, intent, elapsed, flagged)
    ))

    return ChatResponse(
        answer=answer, sources=sources, intent=intent, model_used=model_id,
        latency_ms=elapsed, flagged=flagged,
        attempts=loop_result.get("attempts", 1),
        reflected=loop_result.get("reflected", False),
        reflection_reason=loop_result.get("reflection_reason", "not_reflected"),
        confidence=loop_result.get("confidence", 1.0),
        search_queries=loop_result.get("search_queries", []),
        failure_type=loop_result.get("failure_type"),
        retrieval_latency_ms=loop_result.get("retrieval_latency_ms", 0),
        reranker_scores=loop_result.get("reranker_scores", []),
        tokens_used=loop_result.get("tokens_used", 0),
    )


@app.post("/chat/stream")
@limiter.limit("20/minute")
async def chat_stream(
    request: Request,
    body: ChatRequest,
) -> StreamingResponse:
    t_start    = time.time()
    event_loop = asyncio.get_event_loop()

    raw_query = normalize(body.message)
    query     = rewrite(raw_query, memory, body.session_id)
    intent    = detect_intent(query)
    safety    = safety_check(query)

    async def _error_stream(msg: str):
        payload = json.dumps({
            "token": msg, "done": True, "sources": [], "intent": intent,
            "model_used": "none", "latency_ms": 0,
            "attempts": 1, "reflected": False,
            "reflection_reason": "error", "confidence": 0.0,
        })
        yield f"data: {payload}\n\n"

    if not safety["safe"]:
        return StreamingResponse(_error_stream("Query not allowed. Please rephrase."),
                                 media_type="text/event-stream")

    if store.is_empty():
        return StreamingResponse(_error_stream("No documents loaded. Upload a document first."),
                                 media_type="text/event-stream")

    try:
        loop_result = await asyncio.wait_for(
            event_loop.run_in_executor(None, run_reflection_loop, query, store, intent, None),
            timeout=CHAT_REQUEST_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        # See /chat's identical try/except for why this bounds the response,
        # not the underlying executor thread.
        return StreamingResponse(_error_stream("Request timed out. Please try again."),
                                 media_type="text/event-stream")

    answer     = loop_result["answer"]
    top_chunks = loop_result["chunks"]
    model_id   = loop_result["model_used"]
    sources    = _build_sources(top_chunks, answer)

    async def event_generator():
        # ── Stream the already-computed answer token-by-token ─────────────
        # Fix #1: The reflection loop already did retrieval + generation +
        # validation. We stream the pre-computed answer in word-sized chunks
        # so the UI feels live — no second Groq call needed.
        CHUNK_SIZE = 4   # characters per SSE token event (feels smooth)
        for i in range(0, len(answer), CHUNK_SIZE):
            token = answer[i : i + CHUNK_SIZE]
            yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"
            await asyncio.sleep(0)   # yield control so FastAPI can flush

        # ── Final metadata frame ───────────────────────────────────────────
        elapsed     = int((time.time() - t_start) * 1000)
        validation  = validate(answer, top_chunks)
        source_refs = [SourceRef(**s) for s in sources]
        # See /chat above: store the original query, not the rewritten one,
        # so anchors don't compound turn over turn.
        memory.add(body.session_id, raw_query, answer, source_refs, intent)

        asyncio.create_task(_async_log_feedback(
            query,
            loop_result.get("failure_type", "NONE"),
            loop_result.get("healing_action", "NONE"),
            not validation["flagged"],
        ))
        asyncio.create_task(log_pipeline(
            _build_log_payload(str(uuid.uuid4()), query, safety,
                               loop_result, intent, elapsed, validation["flagged"])
        ))

        final = json.dumps({
            "token": "", "done": True, "sources": sources,
            "intent": intent, "model_used": model_id, "latency_ms": elapsed,
            "flagged": validation["flagged"],
            "attempts":               loop_result.get("attempts", 1),
            "reflected":              loop_result.get("reflected", False),
            "reflection_reason":      loop_result.get("reflection_reason", "not_reflected"),
            "confidence":             round(loop_result.get("confidence", 1.0), 3),
            "search_queries":         loop_result.get("search_queries", []),
            "failure_type":           loop_result.get("failure_type"),
            "retrieval_latency_ms":   loop_result.get("retrieval_latency_ms", 0),
            "reranker_scores":        loop_result.get("reranker_scores", []),
            "tokens_used":            loop_result.get("tokens_used", 0),
        })
        yield f"data: {final}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


async def _async_log_feedback(query, failure_type, fix_applied, success):
    try:
        db_log_feedback(str(uuid.uuid4()), query, failure_type, fix_applied, success)
    except Exception:
        pass


# ── Upload / File management ──────────────────────────────────────────────────

@app.post("/upload", response_model=UploadResponse)
@limiter.limit("10/minute")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
) -> UploadResponse:
    from pathlib import Path
    filename = file.filename or "upload"
    ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    content = await file.read()

    # Reject oversized uploads before spending any parse/chunk/embed work
    # on a file we're going to refuse anyway. See config.py's
    # "Large-file guards" section for why this and MAX_INGEST_CHARS are
    # two separate bounds.
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large ({len(content):,} bytes). "
                f"Maximum allowed is {MAX_UPLOAD_BYTES:,} bytes."
            ),
        )

    content_hash = compute_hash(content)

    # Duplicate check
    if content_hash in store.file_hash_map:
        existing_id = store.file_hash_map[content_hash]
        return UploadResponse(filename=filename, file_id=existing_id,
                              chunk_count=0, status="duplicate",
                              message="File already exists in the store.")

    pending_dup = store.find_pending_by_hash(content_hash)
    if pending_dup is not None:
        if pending_dup["status"] == "processing":
            return UploadResponse(filename=filename, file_id=pending_dup["file_id"],
                                  chunk_count=0, status="duplicate",
                                  message="This file is already being processed (OCR).")
        # An earlier OCR attempt failed: drop it so this re-upload is a clean retry
        # (its row would otherwise hold the content hash and block the new one).
        store.remove_pending(pending_dup["file_id"])
        try:
            delete_file_from_postgres(pending_dup["file_id"])
        except Exception as exc:
            print(f"[WARN] Failed to clear failed OCR record {pending_dup['file_id']!r}: {exc}")

    try:
        pages = parse_file(content, filename)
    except Exception as exc:
        return UploadResponse(filename=filename, file_id="", chunk_count=0,
                              status="error", message=f"Parse error: {exc}")

    file_id   = str(uuid.uuid4())
    file_type = ext.lstrip(".")

    try:
        chunks = chunk_document(
            pages=pages, file_id=file_id, filename=filename, file_type=file_type,
        )
    except Exception as exc:
        return UploadResponse(filename=filename, file_id="", chunk_count=0,
                              status="error", message=f"Chunking error: {exc}")

    valid_chunks = [c for c in chunks if c.get("text", "").strip()]
    if not valid_chunks:
        # A PDF with no extractable text is a scan. OCR takes too long to run
        # inside this request, so queue it as a background job instead.
        if ext == ".pdf" and ocr_jobs.ocr_ready():
            return _queue_ocr_job(content, file_id, filename, file_type, content_hash)
        hint = (
            " This looks like a scanned PDF, but OCR is unavailable "
            "(install rapidocr-onnxruntime and pypdfium2, or set OCR_ENABLED=true)."
            if ext == ".pdf" else
            " If this is a scanned PDF/image, convert it to searchable PDF/text."
        )
        return UploadResponse(filename=filename, file_id="", chunk_count=0,
                              status="error",
                              message="No readable text extracted." + hint)
    chunks = valid_chunks


    result_id, add_status = store.add_file(
        file_id=file_id, filename=filename, content_hash=content_hash,
        chunks=chunks, file_type=file_type,
    )

    if add_status == "limit":
        return UploadResponse(filename=filename, file_id="", chunk_count=0,
                              status="limit", message="Store is full.")

    # Persist the raw bytes so /admin/reingestion-queue/process can later
    # re-parse + re-chunk this file. Non-fatal: the upload already succeeded
    # in-memory/in-store even if this write fails.
    try:
        save_uploaded_file(content, file_id, file_type, UPLOADED_DOCS_DIR)
    except Exception as exc:
        print(f"[WARN] Failed to persist uploaded file {filename!r} for reingestion: {exc}")

    try:
        sync_file_to_postgres(store, file_id)
    except Exception as exc:
        return UploadResponse(filename=filename, file_id=file_id, chunk_count=len(chunks),
                              status="error",
                              message=f"Indexed but failed to persist to database: {exc}")

    return UploadResponse(filename=filename, file_id=file_id, chunk_count=len(chunks),
                          status="ok", message=f"Successfully indexed {len(chunks)} chunks.")


def _queue_ocr_job(content: bytes, file_id: str, filename: str, file_type: str,
                   content_hash: str) -> UploadResponse:
    """
    Accept a scanned PDF as a pending document and start its OCR job.

    The raw bytes are persisted first: a restart mid-job resumes from that
    copy (ocr_jobs.resume_interrupted), so failing to save it fails the
    upload up front rather than promising work that could silently vanish.
    """
    try:
        save_uploaded_file(content, file_id, file_type, UPLOADED_DOCS_DIR)
    except Exception as exc:
        return UploadResponse(filename=filename, file_id="", chunk_count=0, status="error",
                              message=f"Could not save the file for OCR: {exc}")

    store.add_pending(file_id, filename, content_hash, file_type, message="Queued for OCR")
    try:
        register_pending_document(file_id, filename, file_type, content_hash)
        if not ocr_jobs.claim(file_id):   # take the lease so no other worker starts the same job
            raise RuntimeError("could not take the OCR lease")
    except Exception as exc:
        store.remove_pending(file_id)
        return UploadResponse(filename=filename, file_id="", chunk_count=0, status="error",
                              message=f"Could not record the document for OCR: {exc}")

    ocr_jobs.submit(store, file_id, content)
    return UploadResponse(
        filename=filename, file_id=file_id, chunk_count=0, status="processing",
        message="Scanned PDF detected - running OCR in the background. "
                "It becomes searchable when processing finishes.",
    )


@app.delete("/files/{file_id}", response_model=DeleteResponse)
@limiter.limit("20/minute")
async def delete_file(request: Request, file_id: str) -> DeleteResponse:
    doc = db_get_document(file_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="File not found.")

    # A document still being OCR'd lives in store.pending, not store.files;
    # removing it also cancels its job at the next page boundary.
    deleted = store.delete_file(file_id) or store.remove_pending(file_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"File '{file_id}' not found in store.")

    try:
        delete_file_from_postgres(file_id)
    except Exception as exc:
        return DeleteResponse(file_id=file_id, deleted=True,
                              message=f"Deleted from memory but failed to persist: {exc}")

    return DeleteResponse(file_id=file_id, deleted=True,
                          message="File and all its chunks removed successfully.")


@app.get("/docs-loaded", response_model=DocsLoadedResponse)
async def docs_loaded() -> DocsLoadedResponse:
    """Return all documents currently loaded in the store."""
    store_files = store.get_files()
    pending_files = [
        {
            "file_id":     f["file_id"],
            "filename":    f["filename"],
            "chunk_count": 0,
            "uploaded_at": f["uploaded_at"],
            "status":      f["status"],        # "processing" | "failed"
            "message":     f["message"],
            "progress":    f["progress"],      # [pages_done, pages_total] or null
        }
        for f in store.get_pending()
    ]
    if store_files or pending_files:
        files = [
            {
                "file_id":     f["file_id"],
                "filename":    f["filename"],
                "chunk_count": f["chunk_count"],
                "uploaded_at": f["uploaded_at"],
                "status":      "ready",
                "ocr":         f.get("ocr", False),   # read from a scan
            }
            for f in store_files if f.get("status") == "active"
        ]
    else:
        db_docs = db_list_all_documents()
        live_rows = [row for row in db_docs if not row["is_deleted"]]
        files = [
            {
                "file_id":     row["id"],
                "filename":    row["filename"],
                "chunk_count": row["chunk_count"],
                "uploaded_at": row["upload_time"],
                "status":      "ready",
                "ocr":         row["ocr"],
            }
            for row in live_rows if row["status"] == "ready"
        ]
        pending_files = [
            {
                "file_id":     row["id"],
                "filename":    row["filename"],
                "chunk_count": 0,
                "uploaded_at": row["upload_time"],
                "status":      row["status"],
                "message":     row["status_message"] or "",
                "progress":    None,
            }
            for row in live_rows if row["status"] != "ready"
        ]
    # Only searchable documents count toward the totals; pending ones are
    # listed (so the UI can show progress / failures) but not counted.
    return DocsLoadedResponse(
        files=files + pending_files,
        total_files=len(files),
        total_chunks=sum(f["chunk_count"] for f in files),
    )



@app.post("/reload")
@limiter.limit("5/minute")
async def reload_docs(request: Request) -> dict:
    """Re-index all files in test_docs/."""
    from pathlib import Path

    docs_dir = Path("test_docs")
    if not docs_dir.exists():
        docs_dir.mkdir(parents=True)
        return {"status": "no_docs", "message": "test_docs/ created — upload files first."}

    doc_files = sorted(
        p for p in docs_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    loaded = skipped = failed = 0
    log: list = []

    for filepath in doc_files:
        filename = filepath.name
        try:
            content = filepath.read_bytes()
        except Exception as exc:
            log.append({"file": filename, "status": "error", "msg": str(exc)}); failed += 1; continue

        if len(content) > MAX_UPLOAD_BYTES:
            log.append({"file": filename, "status": "error",
                       "msg": f"too large ({len(content):,} bytes, max {MAX_UPLOAD_BYTES:,})"})
            failed += 1; continue

        content_hash = compute_hash(content)
        if content_hash in store.file_hash_map:
            log.append({"file": filename, "status": "skipped", "msg": "duplicate"}); skipped += 1; continue

        try:
            pages = parse_file(content, filename)
        except Exception as exc:
            log.append({"file": filename, "status": "error", "msg": f"parse: {exc}"}); failed += 1; continue

        file_id   = str(uuid.uuid4())
        file_type = filepath.suffix.lower().lstrip(".")

        try:
            chunks = chunk_document(pages=pages, file_id=file_id,
                                    filename=filename, file_type=file_type)
        except Exception as exc:
            log.append({"file": filename, "status": "error", "msg": f"chunk: {exc}"}); failed += 1; continue

        if not chunks:
            log.append({"file": filename, "status": "error", "msg": "no chunks"}); failed += 1; continue

        _, add_status = store.add_file(file_id=file_id, filename=filename,
                                       content_hash=content_hash, chunks=chunks, file_type=file_type)
        if add_status == "ok":
            sync_file_to_postgres(store, file_id)
            log.append({"file": filename, "status": "ok", "chunks": len(chunks)}); loaded += 1
        elif add_status == "duplicate":
            log.append({"file": filename, "status": "skipped", "msg": "duplicate"}); skipped += 1
        else:
            log.append({"file": filename, "status": "limit"}); break

    return {"status": "done", "loaded": loaded, "skipped": skipped, "failed": failed,
            "total_files": len(store.files), "total_chunks": store.total_chunks(), "log": log}


# ── Dynamic Suggestions Agent ─────────────────────────────────────────────────

@app.get("/suggestions", tags=["suggestions"])
@limiter.limit("10/minute")
async def get_suggestions(request: Request) -> dict:
    """
    Analyses uploaded document chunks and generates structured topic cards.
    """
    import re as _re

    if store.is_empty():
        return {"topics": [], "generated": False,
                "message": "Upload documents to get suggestions."}

    fingerprint = _suggestions_fingerprint()
    if _suggestions_cache["fingerprint"] == fingerprint and _suggestions_cache["result"] is not None:
        return _suggestions_cache["result"]

    sample_chunks = _sample_for_suggestions(store.get_all_chunks(), total=30)

    sample_text = "\n\n".join(
        f"[{c['metadata']['filename']} p{c['metadata'].get('page',1)}]: {c['text'][:400]}"
        for c in sample_chunks
    )

    system_prompt = (
        "You are a document intelligence agent. Analyse the provided document excerpts "
        "and identify the most important, distinct themes or areas a user would want to explore. "
        "For each theme, generate a topic card.\n\n"
        "Rules:\n"
        "- Create EXACTLY 4 topic cards.\n"
        "- Each card must be SPECIFIC to the actual content of these documents — "
        "  do NOT use generic labels like 'Overview' or 'Summary'.\n"
        "- Each card has: label (2-4 words), icon (single emoji), color (hex), "
        "  and prompts (array of exactly 4 specific questions about this topic).\n"
        "- Questions must be answerable from the documents — be precise, not vague.\n"
        "- Return ONLY valid JSON (no markdown, no explanation):\n\n"
        '[\n'
        '  {\n'
        '    "label": "Topic Name",\n'
        '    "icon": "📊",\n'
        '    "color": "#c98f3f",\n'
        '    "prompts": ["question 1", "question 2", "question 3", "question 4"]\n'
        '  }\n'
        ']\n\n'
        "Use a variety of colours from: #c98f3f #7fae8a #c17a4a #8fb4c2 #bd6b5c #c9a24a #a68a64 #6f8f7a"
    )

    user_msg = f"Document excerpts:\n{sample_text}"

    def _call_suggestions(model_id: str) -> str:
        return call_groq(model_id, f"{system_prompt}\n\n{user_msg}", "")

    def _generate_topics_raw() -> str:
        # Try preferred Qwen model first; fall back to GROQ_STRONG if unavailable
        try:
            return _call_suggestions(GROQ_QWEN)
        except Exception as qwen_err:
            print(f"  [WARN]  [suggestions] {GROQ_QWEN} failed ({qwen_err}) -- falling back to {GROQ_STRONG}")
            return _call_suggestions(GROQ_STRONG)

    event_loop = asyncio.get_event_loop()

    try:
        # Same reasoning as the retrieval-gate below: call_groq() is a
        # blocking network call — run it off the event loop so one slow
        # /suggestions request can't stall every other concurrent request.
        raw = await event_loop.run_in_executor(None, _generate_topics_raw)

        raw = _re.sub(r"```(?:json)?|```", "", raw).strip()
        match = _re.search(r'\[.*\]', raw, _re.DOTALL)
        if not match:
            return {"topics": [], "generated": False, "message": "Could not parse topics."}

        topics = json.loads(match.group())

        # Deep/saturated so these are still legible as text on the light
        # "Modern Parchment" palette (frontend/src/styles.css) — the old
        # values were tuned for a dark background and read as washed-out
        # pastel there.
        ALLOWED_COLORS = {"#8b4a12","#3f7350","#9c5a28","#3d6b78",
                          "#a1483a","#8a5d14","#6b5638","#3d5c48"}
        COLOR_CYCLE = list(ALLOWED_COLORS)
        clean = []
        for i, t in enumerate(topics[:4]):
            if not isinstance(t, dict):
                continue
            label   = str(t.get("label", f"Topic {i+1}"))[:30]
            icon    = str(t.get("icon", "📄"))[:4]
            color   = t.get("color", COLOR_CYCLE[i % len(COLOR_CYCLE)])
            if color not in ALLOWED_COLORS:
                color = COLOR_CYCLE[i % len(COLOR_CYCLE)]
            prompts = [str(p) for p in t.get("prompts", []) if isinstance(p, str)][:4]
            if not prompts:
                continue
            clean.append({"label": label, "icon": icon, "color": color, "prompts": prompts})

        # ── Retrieval-gate: drop any prompt the store can't answer ───────────
        # Run a lightweight retrieve+rerank for each suggested question and
        # discard any whose best cross-encoder score is below the threshold.
        # This prevents the UI from surfacing a question that the LLM will
        # inevitably answer with "Not found in the document."
        #
        # Each check does real retrieval + cross-encoder inference (both
        # CPU-bound and synchronous) — up to 16 of them here (4 topics x 4
        # prompts). Run the whole batch in the executor, same as /chat does
        # for run_reflection_loop, so this doesn't stall the event loop for
        # every other concurrent request.
        from retrieval.search import hybrid_retrieve
        from retrieval.reranker import rerank as _rerank

        def _prompt_is_answerable(question: str) -> bool:
            try:
                candidates = hybrid_retrieve(question, store, top_k=SUGGESTION_CE_TOP_N)
                if not candidates:
                    return False
                ranked = _rerank(candidates, query=question, final_k=SUGGESTION_CE_TOP_N)
                if not ranked:
                    return False
                best_score = ranked[0].get("reranker_score", ranked[0].get("score", -99.0))
                return best_score >= SUGGESTION_MIN_SCORE
            except Exception:
                return True   # fail open — keep the prompt if check errors

        def _validate_topics() -> list:
            validated = []
            for topic in clean:
                good_prompts = [p for p in topic["prompts"] if _prompt_is_answerable(p)]
                if good_prompts:
                    validated.append({**topic, "prompts": good_prompts})
            return validated

        validated = await event_loop.run_in_executor(None, _validate_topics)

        result = {"topics": validated, "generated": True}
        # Only cache real successes — a transient LLM/parse failure above
        # returns early without reaching here, so it's naturally retried on
        # the next call instead of getting stuck cached.
        _suggestions_cache["fingerprint"] = fingerprint
        _suggestions_cache["result"] = result
        return result

    except Exception as exc:
        return {"topics": [], "generated": False, "message": str(exc)}


# ── System endpoints ──────────────────────────────────────────────────────────

@app.get("/stats", response_model=StatsResponse)
async def stats() -> StatsResponse:
    return StatsResponse(**get_stats())


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        docs_loaded=len(store.files),
        total_chunks=store.total_chunks(),
        model_fast=GROQ_FAST,
        model_strong=GROQ_STRONG,
        model_reasoning=GROQ_QWEN,
    )


@app.get("/conversations/{session_id}/export")
async def export_conversation(session_id: str, format: str = "markdown") -> Response:
    history = memory.get_history(session_id)
    if not history:
        raise HTTPException(status_code=404, detail="No conversation history found for this session.")

    if format == "markdown":
        content = to_markdown(session_id, history)
        return Response(
            content=content,
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="conversation_{session_id}.md"'},
        )

    if format == "pdf":
        content = to_pdf(session_id, history)
        return Response(
            content=content,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="conversation_{session_id}.pdf"'},
        )

    raise HTTPException(
        status_code=400,
        detail=f"Unsupported export format '{format}'. Supported: markdown, pdf",
    )


# ── Admin routes ──────────────────────────────────────────────────────────────

@app.get("/admin/documents", tags=["admin"])
async def admin_list_documents():
    rows = db_list_all_documents()
    return {"documents": [dict(r) for r in rows]}


@app.get("/admin/reingestion-queue", tags=["admin"])
async def admin_reingestion_queue(clear: bool = False):
    """
    Return all entries in the healer reingestion signal queue.

    Queueing is operator-driven, not automatic: nothing in the live
    reflection pipeline currently detects "this data is stale" from a
    query/answer/chunks alone (unlike the other failure types, staleness
    needs an external signal reflection doesn't have) — see
    agents/reflection.py's _map_failure_type() docstring. An operator (or
    a future signal source) queues an entry by writing OUTDATED_DATA as
    the failure_reason via agents/healer.py's REINGEST action.
    Pass ?clear=true to flush the queue after reading.
    """
    import json as _json
    from config import REINGESTION_QUEUE_PATH

    queue_path = REINGESTION_QUEUE_PATH
    entries = []
    if os.path.exists(queue_path):
        try:
            with open(queue_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(_json.loads(line))
                        except Exception:
                            pass
            if clear:
                open(queue_path, "w").close()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to read queue: {exc}")

    return {
        "count":   len(entries),
        "cleared": clear,
        "entries": entries,
    }


@app.post("/admin/reingestion-queue/process", tags=["admin"])
@limiter.limit("5/minute")
async def admin_process_reingestion_queue(request: Request) -> dict:
    """
    Consume the reingestion signal queue: re-parse + re-chunk every active
    document from its persisted upload (see ingestion/reingest.py), replace
    its chunks in the store, then clear the queue.

    Limitation: this re-runs today's parser/chunker against the ORIGINAL
    uploaded bytes — it refreshes chunking, not the underlying data. If a
    queued entry means the source document itself is out of date, someone
    still needs to upload a newer version; this just makes sure that once
    they do (or once parsing/chunking logic improves), a refresh is one
    call away instead of nothing at all.
    """
    entries = 0
    if os.path.exists(REINGESTION_QUEUE_PATH):
        with open(REINGESTION_QUEUE_PATH, "r", encoding="utf-8") as f:
            entries = sum(1 for line in f if line.strip())

    if entries == 0:
        return {"processed_queue_entries": 0, "refreshed": [], "skipped": [],
                "message": "Queue is empty — nothing to process."}

    summary = refresh_all_from_disk(store, UPLOADED_DOCS_DIR)

    # refresh_all_from_disk() already replaced each refreshed file's chunks
    # in the in-memory store; sync those new chunks to Postgres so the
    # refresh is durable, not just in-memory until the next restart.
    sync_errors = []
    for filename in summary["refreshed"]:
        matching_ids = [fid for fid, rec in store.files.items() if rec["filename"] == filename]
        for file_id in matching_ids:
            try:
                sync_file_to_postgres(store, file_id)
            except Exception as exc:
                sync_errors.append({"file": filename, "error": str(exc)})
    if sync_errors:
        summary["sync_errors"] = sync_errors

    open(REINGESTION_QUEUE_PATH, "w").close()  # clear queue after processing

    return {"processed_queue_entries": entries, **summary}



@app.get("/admin/stats", tags=["admin"])
async def admin_stats():
    sql_stats  = db_get_system_stats()
    live_stats = get_stats()
    # Both carry "total_queries": the in-memory one (since this process
    # started) wins the merge, so the all-time count of logged answers, the
    # denominator of success_rate, is also returned under its own name.
    return {**sql_stats, **live_stats,
            "answers_logged": sql_stats["total_queries"],
            "store_chunks": store.total_chunks(),
            "store_files":  len(store.files)}


@app.delete("/admin/documents/{doc_id}", tags=["admin"])
@limiter.limit("20/minute")
async def admin_delete_document(request: Request, doc_id: str):
    doc = db_get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    if not store.delete_file(doc_id):
        store.remove_pending(doc_id)   # cancels an in-flight OCR job
    try:
        delete_file_from_postgres(doc_id)
    except Exception as exc:
        print(f"[WARN] Failed to persist deletion of {doc_id!r}: {exc}")

    return {"deleted": True, "doc_id": doc_id}


@app.delete("/admin/bulk-delete/documents", tags=["admin"])
@limiter.limit("5/minute")
async def admin_delete_all_documents(request: Request):
    """Alias for DELETE /documents/clear-all — prefer that endpoint."""
    return await clear_all_documents(request)


@app.delete("/documents/clear-all", tags=["documents"])
@limiter.limit("5/minute")
async def clear_all_documents(request: Request):
    """Delete ALL documents from the store."""
    count = len(store.files) + len(store.pending)
    file_ids_in_store = list(store.files.keys()) + list(store.pending.keys())

    store.pending.clear()   # also cancels any in-flight OCR jobs
    store.files.clear()
    store.chunks.clear()
    store.file_hash_map.clear()
    store._chunk_hashes.clear()
    store._rebuild_index()

    for file_id in file_ids_in_store:
        try:
            delete_file_from_postgres(file_id)
        except Exception as exc:
            print(f"[WARN] Failed to persist deletion of {file_id!r}: {exc}")

    # Catch any documents that exist in Postgres but weren't in the
    # in-memory store (e.g. a prior sync failure) so clear-all is thorough.
    stray = [d for d in db_list_all_documents() if not d["is_deleted"]]
    for d in stray:
        try:
            delete_file_from_postgres(d["id"])
        except Exception as exc:
            print(f"[WARN] Failed to persist deletion of {d['id']!r}: {exc}")

    # Only documents that were live: the full row list also holds every
    # document removed earlier, which used to inflate this number.
    return {
        "deleted": True,
        "count": count + len(stray),
        "message": f"Cleared all documents and chunks. Ready for fresh document ingestion!"
    }

