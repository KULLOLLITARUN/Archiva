"""
main.py — Archiva FastAPI application (open / no-auth edition).

Auth fully removed. All endpoints are open — no login, register, or tokens.
"""

import asyncio
import json
import os
import pickle
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.requests import Request

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# ── Project imports ───────────────────────────────────────────────────────────
from config import (
    GROQ_FAST, GROQ_STRONG, GROQ_QWEN,
)
from database import (
    init_db,
    db_create_document, db_list_all_documents,
    db_get_document, db_soft_delete_document,
    db_log_feedback, db_get_system_stats,
)
from models.schemas import (
    ChatRequest, ChatResponse, SourceRef,
    DeleteResponse, DocsLoadedResponse, HealthResponse,
    StatsResponse, UploadResponse,
)
from retrieval.store import MultiDocStore
from ingestion.parser import parse_file, SUPPORTED_EXTENSIONS, compute_hash
from ingestion.chunker import chunk_document
from chatbot.memory import ConversationMemory
from chatbot.normalizer import normalize
from chatbot.rewriter import rewrite
from chatbot.intent import detect_intent
from agents.safety import safety_check
from agents.worker import call_groq_stream, build_prompt, call_groq
from agents.validator import validate
from agents.loop import run_reflection_loop, build_labeled_context
from monitor.logger import log_pipeline, get_stats

STORE_PKL = "store_state.pkl"
STORE_TMP = "store_state.tmp"

limiter = Limiter(key_func=get_remote_address, default_limits=[])

store = MultiDocStore()
memory = ConversationMemory()



@asynccontextmanager
async def lifespan(app: FastAPI):
    global store, memory

    # Initialise SQLite DB (idempotent)
    init_db()

    # Load persisted vector store
    try:
        with open(STORE_PKL, "rb") as f:
            store = pickle.load(f)
        loaded_version = getattr(store, "STORE_VERSION", "1.0")
        if loaded_version != MultiDocStore.CURRENT_VERSION:
            print(f"[Warn] Store version mismatch — starting fresh.")
            store = MultiDocStore()
        else:
            print(f"[OK] Store loaded: {store.total_chunks()} chunks across {len(store.files)} file(s)")
    except Exception as e:
        store = MultiDocStore()
        print(f"[Warn] No store found ({e}). Starting empty.")

    memory = ConversationMemory()
    yield


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

def _save_store() -> None:
    with open(STORE_TMP, "wb") as f:
        pickle.dump(store, f)
    os.replace(STORE_TMP, STORE_PKL)


def _build_sources(chunks: list) -> list:
    return [
        {
            "filename": c["metadata"]["filename"],
            "page":     c["metadata"]["page"],
            "text":     c["text"][:200],
            "score":    round(c["score"], 4),
        }
        for c in chunks
    ]


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

    query  = normalize(body.message)
    query  = rewrite(query, memory, body.session_id)
    intent = detect_intent(query)
    safety = safety_check(query)

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

    loop_result = await event_loop.run_in_executor(
        None, run_reflection_loop, query, store, intent, None,
    )

    answer     = loop_result["answer"]
    top_chunks = loop_result["chunks"]
    model_id   = loop_result["model_used"]

    validation = validate(answer, top_chunks)
    flagged    = validation["flagged"]
    sources    = [SourceRef(**s) for s in _build_sources(top_chunks)]

    memory.add(body.session_id, query, answer, sources, intent)
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

    query  = normalize(body.message)
    query  = rewrite(query, memory, body.session_id)
    intent = detect_intent(query)
    safety = safety_check(query)

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

    loop_result = await event_loop.run_in_executor(
        None, run_reflection_loop, query, store, intent, None,
    )

    answer     = loop_result["answer"]
    top_chunks = loop_result["chunks"]
    model_id   = loop_result["model_used"]
    sources    = _build_sources(top_chunks)

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
        memory.add(body.session_id, query, answer, source_refs, intent)

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

    content      = await file.read()
    content_hash = compute_hash(content)

    # Duplicate check
    if content_hash in store.file_hash_map:
        existing_id = store.file_hash_map[content_hash]
        return UploadResponse(filename=filename, file_id=existing_id,
                              chunk_count=0, status="duplicate",
                              message="File already exists in the store.")

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
        return UploadResponse(filename=filename, file_id="", chunk_count=0,
                              status="error",
                              message="No readable text extracted. If this is a scanned PDF/image, convert it to searchable PDF/text.")
    chunks = valid_chunks


    result_id, add_status = store.add_file(
        file_id=file_id, filename=filename, content_hash=content_hash,
        chunks=chunks, file_type=file_type,
    )

    if add_status == "limit":
        return UploadResponse(filename=filename, file_id="", chunk_count=0,
                              status="limit", message="Store is full.")

    db_create_document(file_id, filename, file_type, len(chunks))

    try:
        _save_store()
    except Exception as exc:
        return UploadResponse(filename=filename, file_id=file_id, chunk_count=len(chunks),
                              status="error",
                              message=f"Indexed but failed to persist to disk: {exc}")

    return UploadResponse(filename=filename, file_id=file_id, chunk_count=len(chunks),
                          status="ok", message=f"Successfully indexed {len(chunks)} chunks.")


@app.delete("/files/{file_id}", response_model=DeleteResponse)
async def delete_file(file_id: str) -> DeleteResponse:
    doc = db_get_document(file_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="File not found.")

    deleted = store.delete_file(file_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"File '{file_id}' not found in store.")

    db_soft_delete_document(file_id)

    try:
        _save_store()
    except Exception as exc:
        return DeleteResponse(file_id=file_id, deleted=True,
                              message=f"Deleted from memory but failed to persist: {exc}")

    return DeleteResponse(file_id=file_id, deleted=True,
                          message="File and all its chunks removed successfully.")


@app.get("/docs-loaded", response_model=DocsLoadedResponse)
async def docs_loaded() -> DocsLoadedResponse:
    """Return all documents currently loaded in the store."""
    store_files = store.get_files()
    if store_files:
        files = [
            {
                "filename":    f["filename"],
                "chunk_count": f["chunk_count"],
                "uploaded_at": f["uploaded_at"],
            }
            for f in store_files if f.get("status") == "active"
        ]
    else:
        db_docs = db_list_all_documents()
        files = [
            {
                "filename":    row["filename"],
                "chunk_count": row["chunk_count"],
                "uploaded_at": row["upload_time"],
            }
            for row in db_docs if not row["is_deleted"]
        ]
    return DocsLoadedResponse(
        files=files,
        total_files=len(files),
        total_chunks=sum(f["chunk_count"] for f in files),
    )



@app.post("/reload")
async def reload_docs() -> dict:
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
            db_create_document(file_id, filename, file_type, len(chunks))
            _save_store()
            log.append({"file": filename, "status": "ok", "chunks": len(chunks)}); loaded += 1
        elif add_status == "duplicate":
            log.append({"file": filename, "status": "skipped", "msg": "duplicate"}); skipped += 1
        else:
            log.append({"file": filename, "status": "limit"}); break

    return {"status": "done", "loaded": loaded, "skipped": skipped, "failed": failed,
            "total_files": len(store.files), "total_chunks": store.total_chunks(), "log": log}


# ── Dynamic Suggestions Agent ─────────────────────────────────────────────────

@app.get("/suggestions", tags=["suggestions"])
async def get_suggestions() -> dict:
    """
    Analyses uploaded document chunks and generates structured topic cards.
    """
    import re as _re

    if store.is_empty():
        return {"topics": [], "generated": False,
                "message": "Upload documents to get suggestions."}

    all_chunks = store.get_all_chunks()
    by_file: dict = {}
    for c in all_chunks:
        fid = c["metadata"].get("file_id", "x")
        by_file.setdefault(fid, []).append(c)
    interleaved = []
    iters = [iter(v) for v in by_file.values()]
    while iters:
        next_iters = []
        for it in iters:
            try:
                interleaved.append(next(it))
                next_iters.append(it)
            except StopIteration:
                pass
        iters = next_iters
    sample_chunks = interleaved[:30]

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
        '    "color": "#7c6fff",\n'
        '    "prompts": ["question 1", "question 2", "question 3", "question 4"]\n'
        '  }\n'
        ']\n\n'
        "Use a variety of colours from: #7c6fff #2dd4bf #fb923c #f472b6 #38bdf8 #a78bfa #fbbf24 #34d399"
    )

    user_msg = f"Document excerpts:\n{sample_text}"

    try:
        raw = call_groq(GROQ_QWEN, f"{system_prompt}\n\n{user_msg}", "")
        raw = _re.sub(r"```(?:json)?|```", "", raw).strip()
        match = _re.search(r'\[.*\]', raw, _re.DOTALL)
        if not match:
            return {"topics": [], "generated": False, "message": "Could not parse topics."}

        topics = json.loads(match.group())

        ALLOWED_COLORS = {"#7c6fff","#2dd4bf","#fb923c","#f472b6",
                          "#38bdf8","#a78bfa","#fbbf24","#34d399"}
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

        return {"topics": clean, "generated": True}

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


# ── Admin routes ──────────────────────────────────────────────────────────────

@app.get("/admin/documents", tags=["admin"])
async def admin_list_documents():
    rows = db_list_all_documents()
    return {"documents": [dict(r) for r in rows]}


@app.get("/admin/reingestion-queue", tags=["admin"])
async def admin_reingestion_queue(clear: bool = False):
    """
    Return all entries in the healer reingestion signal queue.
    These are queries where the system detected OUTDATED_DATA failure.
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




@app.get("/admin/stats", tags=["admin"])
async def admin_stats():
    sql_stats  = db_get_system_stats()
    live_stats = get_stats()
    return {**sql_stats, **live_stats,
            "store_chunks": store.total_chunks(),
            "store_files":  len(store.files)}


@app.delete("/admin/documents/{doc_id}", tags=["admin"])
async def admin_delete_document(doc_id: str):
    doc = db_get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    store.delete_file(doc_id)
    db_soft_delete_document(doc_id)
    try:
        _save_store()
    except Exception:
        pass

    return {"deleted": True, "doc_id": doc_id}


@app.delete("/admin/bulk-delete/documents", tags=["admin"])
async def admin_delete_all_documents():
    count = len(store.files)
    store.files.clear()
    store.chunks.clear()
    store.file_hash_map.clear()
    store._chunk_hashes.clear()
    store._rebuild_index()

    rows = db_list_all_documents()
    for d in rows:
        db_soft_delete_document(d["id"])

    try:
        _save_store()
    except Exception:
        pass

    return {"deleted": True, "count": max(count, len(rows))}


@app.delete("/documents/clear-all", tags=["documents"])
async def clear_all_documents():
    """Delete ALL documents from the store."""
    count = len(store.files)
    store.files.clear()
    store.chunks.clear()
    store.file_hash_map.clear()
    store._chunk_hashes.clear()
    store._rebuild_index()

    rows = db_list_all_documents()
    for d in rows:
        db_soft_delete_document(d["id"])

    try:
        _save_store()
    except Exception as exc:
        print(f"[WARN] Failed to save empty store: {exc}")

    return {
        "deleted": True,
        "count": max(count, len(rows)),
        "message": f"Cleared all documents and chunks. Ready for fresh document ingestion!"
    }

