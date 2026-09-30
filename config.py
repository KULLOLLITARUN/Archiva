import os
import re
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────

# Single key (legacy fallback)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Multi-key list for round-robin (Part 7).
# Format in .env: GROQ_API_KEYS=key1,key2,key3
# Falls back to the single GROQ_API_KEY if not set.
def _load_groq_keys() -> list:
    raw = os.getenv("GROQ_API_KEYS", "")
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    if not keys and GROQ_API_KEY:
        keys = [GROQ_API_KEY]
    return keys

GROQ_API_KEYS: list = _load_groq_keys()

# ── Models ────────────────────────────────────────────────────────────────────

GROQ_FAST   = os.getenv("GROQ_FAST",   "llama-3.1-8b-instant")
GROQ_STRONG = os.getenv("GROQ_STRONG", "llama-3.3-70b-versatile")
GROQ_QWEN   = os.getenv("GROQ_QWEN",   "llama-3.3-70b-versatile")  # override with Qwen if available on your plan

# Per-HTTP-call ceiling (connect+read) on the Groq client itself — bounds a
# single network call so a stalled connection can't hang indefinitely.
# The SDK's own max_retries is set to 0 wherever this client is built
# (llm/groq_manager.py): agents/worker.py already implements its own
# retry/key-rotation loop (_MAX_RETRIES), so leaving the SDK's default
# retries on top would silently multiply attempts (and worst-case latency)
# without worker.py's logic knowing about it.
GROQ_REQUEST_TIMEOUT_S = float(os.getenv("GROQ_REQUEST_TIMEOUT_S", 30.0))

# Ceiling on a single /chat or /chat/stream request's retrieve->generate->
# reflect->heal pipeline (run off the event loop via run_in_executor). This
# doesn't stop the underlying thread — Python threads can't be forcibly
# killed — but it bounds how long the HTTP response can be held open, so
# one stuck request can't hang a client (or, via slow-loris-style repeated
# calls, exhaust the executor's thread pool) forever.
CHAT_REQUEST_TIMEOUT_S = float(os.getenv("CHAT_REQUEST_TIMEOUT_S", 60.0))

# ── BM25 Retrieval ────────────────────────────────────────────────────────────

# Minimum BM25 score to return a result.
BM25_THRESHOLD = float(os.getenv("BM25_THRESHOLD", 0.1))

# ── Retrieval ─────────────────────────────────────────────────────────────────

TOP_K       = int(os.getenv("TOP_K", 5))
FINAL_K     = int(os.getenv("FINAL_K", 5))

# Hybrid retrieval (Part 2): per-source candidate counts before RRF fusion.
TOP_K_BM25  = int(os.getenv("TOP_K_BM25", 20))
TOP_K_DENSE = int(os.getenv("TOP_K_DENSE", 20))

# Weight for RRF k-constant (higher = more rank smoothing).
RRF_K = int(os.getenv("RRF_K", 60))

# Hybrid blend alpha: 0 = BM25-only, 1 = dense-only, 0.5 = balanced.
# Used to weight RRF scores from each retriever before fusion.
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", 0.5))

# ── Dense Retrieval (sentence-transformers) ────────────────────────────────────

# HuggingFace model for dense embeddings.
DENSE_MODEL = os.getenv("DENSE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Disk-cache directory for precomputed embeddings (diskcache).
EMBEDDING_CACHE_DIR = os.getenv("EMBEDDING_CACHE_DIR", "cache/embeddings")

# ── Cross-Encoder Reranking ────────────────────────────────────────────────────

CROSS_ENCODER_MODEL = os.getenv(
    "CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# Number of candidates fed into the cross-encoder before trimming to FINAL_K.
CROSS_ENCODER_TOP_N = int(os.getenv("CROSS_ENCODER_TOP_N", 20))

# ── Context Optimization (MMR) ────────────────────────────────────────────────

# Lambda for Maximal Marginal Relevance: 1.0 = pure relevance, 0.0 = pure diversity.
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", 0.7))

# Maximum tokens per chunk before compression.
MAX_CHUNK_TOKENS = int(os.getenv("MAX_CHUNK_TOKENS", 300))

# ── Storage ───────────────────────────────────────────────────────────────────

# Dense retrieval (retrieval/dense.py) is exact brute-force cosine similarity
# over every chunk in the store — no ANN index (FAISS/HNSW/etc). That's a
# deliberate choice at this cap: numpy's matrix @ query_vec is sub-millisecond
# at a few thousand chunks, and skipping an ANN index avoids graph tuning,
# tombstoning on delete, and a new native dependency for accuracy no one needs
# yet. If this cap is ever raised well past its current default (tens of
# thousands of chunks), THAT is the signal to revisit an ANN index — not
# before, and not preemptively.
MAX_TOTAL_CHUNKS    = int(os.getenv("MAX_TOTAL_CHUNKS", 5000))
MAX_CHUNKS_PER_FILE = int(os.getenv("MAX_CHUNKS_PER_FILE", 500))

# ── Large-file guards ─────────────────────────────────────────────────────────
# Two independent bounds so a huge file costs bounded work, not unbounded:
#
# 1. MAX_UPLOAD_BYTES rejects an absurdly large upload outright, in
#    main.py's /upload handler, before any parsing/chunking/embedding is
#    attempted on a file we're going to refuse anyway.
# 2. MAX_INGEST_CHARS bounds how much decoded text a single-page format
#    (.txt/.csv/.html — anything that isn't naturally paginated like a PDF)
#    will actually process. Chunking a whole page is not itself lazy
#    (parent/child building materializes full lists), so without this, one
#    enormous single-blob file could still do unbounded work even though it
#    passed the upload-size gate. PDFs don't need this: parse_pdf() yields
#    pages lazily, so chunk_document()'s existing MAX_CHUNKS_PER_FILE early
#    stop already skips extracting pages beyond the cap.
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 50 * 1024 * 1024))   # 50 MB
MAX_INGEST_CHARS = int(os.getenv("MAX_INGEST_CHARS", 2_000_000))         # ~2M chars

# ── OCR (scanned/image-only PDFs) ──────────────────────────────────────────────
# OCR runs as a background job (ingestion/ocr_jobs.py), never inline in
# /upload. Bounded like every other ingestion path: OCR_MAX_PAGES caps the
# work per file, and lines the engine isn't confident about are DROPPED
# rather than indexed (OCR_MIN_LINE_CONFIDENCE) - low-confidence text in
# the search index is worse than a gap, since it invites hallucination.
OCR_ENABLED             = os.getenv("OCR_ENABLED", "true").lower() in ("1", "true", "yes")
OCR_MAX_PAGES           = int(os.getenv("OCR_MAX_PAGES", 50))
OCR_RENDER_SCALE        = float(os.getenv("OCR_RENDER_SCALE", 2.0))       # 2.0 ~ 144 dpi
OCR_MIN_LINE_CONFIDENCE = float(os.getenv("OCR_MIN_LINE_CONFIDENCE", 0.6))
# A worker owning an OCR job renews its lease once per page; if it goes this
# long without renewing (crash, hang) another worker may take the job over.
OCR_LEASE_SECONDS       = float(os.getenv("OCR_LEASE_SECONDS", 120))

# ── Multi-process operation ────────────────────────────────────────────────────
# Each worker process keeps its own in-memory copy of the document store.
# Every STORE_SYNC_INTERVAL_S seconds it compares Postgres's store_version
# with the one it last loaded and reloads only if another worker changed the
# documents (db/store_sync.py's StoreSynchronizer). Costs one tiny query per
# tick when idle. Set STORE_SYNC_ENABLED=false to skip the background task
# in a strictly single-process deployment.
STORE_SYNC_ENABLED    = os.getenv("STORE_SYNC_ENABLED", "true").lower() in ("1", "true", "yes")
STORE_SYNC_INTERVAL_S = float(os.getenv("STORE_SYNC_INTERVAL_S", 2.0))

# Where slowapi keeps rate-limit counters. The default is per-process memory,
# so with N workers each one enforces the limit separately (effective limit
# is N x the stated one). Point this at a shared backend for a true global
# limit, e.g. redis://localhost:6379 (needs `pip install redis`).
RATE_LIMIT_STORAGE_URI = os.getenv("RATE_LIMIT_STORAGE_URI", "memory://")

# ── Chunking ──────────────────────────────────────────────────────────────────

CHUNK_SIZE    = 400
CHUNK_OVERLAP = 50

# ── Parent-Child Chunking ─────────────────────────────────────────────────────
# Parent chunks are large sections fed to the LLM as rich context.
# Child chunks are small units used for precise vector / BM25 search.
PARENT_CHUNK_SIZE = int(os.getenv("PARENT_CHUNK_SIZE", 1200))  # tokens
CHILD_CHUNK_SIZE  = int(os.getenv("CHILD_CHUNK_SIZE",  200))   # tokens (was 120, raised to reduce over-fragmentation)

# ── Context ───────────────────────────────────────────────────────────────────

MAX_CONTEXT_TOKENS = 6000

# ── Semantic Query Cache ──────────────────────────────────────────────────────
# Cosine similarity threshold for a query to be considered a cache hit.
# 0.97+ required to avoid false hits on structurally similar but topically
# different queries (e.g. "give summary of linkedin" vs "give summary of roadmap").
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", 0.97))
# Maximum number of (query, result) pairs to keep in memory.
SEMANTIC_CACHE_MAX_SIZE  = int(os.getenv("SEMANTIC_CACHE_MAX_SIZE", 200))

# ── Suggestions retrieval-gate ────────────────────────────────────────────────
# GET /suggestions runs a lightweight retrieve+rerank for each LLM-proposed
# prompt and drops any whose best cross-encoder score is below this threshold,
# so the UI never surfaces a question the store can't actually answer.
SUGGESTION_MIN_SCORE = float(os.getenv("SUGGESTION_MIN_SCORE", -5.0))
SUGGESTION_CE_TOP_N   = int(os.getenv("SUGGESTION_CE_TOP_N", 5))

# ── Router ────────────────────────────────────────────────────────────────────

ROUTER_LONG_QUERY = 20

# ── Memory persistence ────────────────────────────────────────────────────────

PERSIST_MEMORY = os.getenv("PERSIST_MEMORY", "true").lower() != "false"

# ── Observability ─────────────────────────────────────────────────────────────

FEEDBACK_LOG_PATH    = os.getenv("FEEDBACK_LOG_PATH", "logs/feedback.jsonl")
REINGESTION_QUEUE_PATH = os.getenv("REINGESTION_QUEUE_PATH", "logs/reingestion_queue.jsonl")

# ── Stopwords ─────────────────────────────────────────────────────────────────

STOPWORDS = {
    "the", "is", "and", "a", "to", "of", "in", "that",
    "it", "this", "was", "for", "on", "are", "with",
    "as", "at", "be", "by", "from", "or", "an", "but"
}

COMPLEX_KEYWORDS = [
    "compare", "difference", "analyze", "summarize",
    "explain why", "contrast", "evaluate", "between",
    "across", "versus", "vs", "pros and cons",
    "similarities", "differences"
]

FOLLOWUP_SIGNALS = [
    "it", "this", "that", "explain it", "tell me more",
    "what does it mean", "elaborate", "go deeper",
    "and what about", "more detail", "expand on",
    "what about", "continue", "and then"
]

BLOCK_PATTERNS = [
    r"ignore (previous|above|all) instructions",
    r"you are now",
    r"pretend you",
    r"jailbreak",
    r"bypass.*filter",
    r"disregard.*rules",
    r"act as if",
    r"forget.*instructions",
    r"new persona",
    r"override"
]

AMBIGUITY_TRIGGERS = [
    "ignore", "override", "bypass",
    "instructions", "forget", "pretend"
]

# ── Document-content injection screening ──────────────────────────────────────
# Retrieved chunk text is untrusted — it can come from any uploaded file, and
# an injected instruction embedded in a document ("ignore previous
# instructions...") would otherwise flow straight into the LLM context.
#
# Deliberately narrower than BLOCK_PATTERNS (query screening): bare
# single-word patterns like "override"/"bypass" are common in ordinary
# technical prose and would false-positive constantly at document length,
# where BLOCK_PATTERNS only ever sees one short user query at a time.
DOCUMENT_INJECTION_PATTERNS = [
    r"ignore (previous|above|all) instructions",
    r"disregard (all|the) (previous|above|prior) (instructions|rules)",
    r"you are now (a|an|the)",
    r"pretend you are",
    r"act as if you (are|were)",
    r"forget (all|your) (previous|prior) instructions",
    r"jailbreak",
    r"new persona",
    r"reveal (the|your) system prompt",
]

# ── Reingestion / uploaded-file persistence ───────────────────────────────────

# Directory where raw uploaded file bytes are persisted so the reingestion
# queue (agents/healer.py REINGEST action) can trigger a real re-parse +
# re-chunk pass instead of only logging a signal. See ingestion/reingest.py.
UPLOADED_DOCS_DIR = os.getenv("UPLOADED_DOCS_DIR", "uploaded_docs")

# ── Reflection loop ───────────────────────────────────────────────────────────

# Maximum retrieve→generate→reflect cycles per query.
MAX_REFLECTION_ATTEMPTS = int(os.getenv("MAX_REFLECTION_ATTEMPTS", 3))

# Minimum confidence score to accept without retry (0.0–1.0).
MIN_REFLECTION_CONFIDENCE = float(os.getenv("MIN_REFLECTION_CONFIDENCE", 0.4))

# Minimum overlap ratio between answer words and chunk words.
MIN_OVERLAP_RATIO = float(os.getenv("MIN_OVERLAP_RATIO", 0.15))

# Confidence below this threshold (but above MIN_REFLECTION_CONFIDENCE) triggers
# an optional LLM faithfulness judge as a second-pass check.
# Set to 0.0 to disable the judge entirely.
JUDGE_CONFIDENCE_THRESHOLD = float(os.getenv("JUDGE_CONFIDENCE_THRESHOLD", 0.7))

