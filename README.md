# Archiva — Self-Healing Agentic RAG for Your Documents

A self-hosted document Q&A system built with FastAPI + React. Answers questions
**strictly from your loaded documents** — hybrid retrieval, cross-encoder
reranking, and a self-healing reflection loop that retries and repairs its own
failures before ever returning an answer.

> **Open / no-auth edition.** Every endpoint is unauthenticated by design —
> this runs as a self-hosted, single-instance tool, not a multi-tenant
> service. See [Known Limitations](#known-limitations-deliberate-not-oversights)
> before deploying it anywhere other endpoints can reach.

CI: the full test suite (97 tests) runs on every push/PR via
`.github/workflows/tests.yml` — no API key required, every LLM call in the
suite is mocked.

---

## Architecture at a Glance

```
User Query
   │
   ▼
Normalizer → Context-Aware Rewriter → Intent Detector → Safety Layer
                                        (regex → ambiguity → LLM classifier,
                                         layer 3 only fires on ambiguous input)
   │
   ▼
Semantic Cache lookup ── hit (cosine ≥ 0.97) ──→ Return cached result  [NO LLM]
   │ miss
   ▼
Multi-Hop Decomposition ── query splits into N sub-questions ──→ run each
   │                                                              through this
   │ single question                                             whole pipeline
   ▼                                                              independently,
Hybrid Retrieval: BM25 + Dense (sentence-transformers) → RRF fusion          then merge
   │
   ▼
Score Gate ── below threshold ──→ "Not found in the document."  [NO LLM]
   │
   ▼
Cross-Encoder Reranking (query-aware)
   │
   ▼
Context Optimizer: injection screening → MMR diversification →
                    compression → parent-section expansion → citations
   │
   ▼
Router ── complex/long query ──→ strong model
       └── simple             ──→ fast model
   │
   ▼
LLM Call (Groq, multi-key rotation + retry/backoff)
   │
   ▼
Reflection (deterministic: overlap ratio, number-grounding, contradiction
            check — zero LLM calls) → structured failure_type
   │
   ├── passed, confident ──────────────────────────────→ Response
   │
   └── failed / low-confidence
        │
        ▼
   Self-Healing Loop (root_cause → healer)
     REWRITE_QUERY | INCREASE_TOP_K | STRICT_PROMPT | REINGEST
        │
        └── retry (up to MAX_REFLECTION_ATTEMPTS), optionally escalating
            to the strong model, optionally consulting an LLM faithfulness
            judge on borderline-confidence answers
   │
   ▼
Validator → Response + Sources
```

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- Node.js 18+
- A Groq API key → https://console.groq.com

### 2. Configure

```bash
cp .env.example .env
# Only GROQ_API_KEY is required; everything else has a sane default.
```

### 3. Install & Run

**Windows:**
```bat
.\start.bat
```

**Linux / macOS:**
```bash
bash start.sh
```

This installs dependencies, starts the FastAPI backend on port 8000, and the
React frontend (usually http://localhost:5173).

### 4. Add Documents

Either use the `/upload` endpoint (from the UI, or directly — see below), or
drop files into `test_docs/` and call `POST /reload`.

---

## Manual Setup (Step by Step)

```bash
# Install Python dependencies (includes pytest)
pip install -r requirements.txt

# Start backend
uvicorn main:app --reload
# → http://localhost:8000

# Start frontend (new terminal)
cd frontend
npm install
npm run dev
```

---

## Project Structure

```
rag_agentic/
│
├── main.py               ← FastAPI app: all HTTP endpoints
├── config.py              ← All settings, thresholds, and constants
├── database.py             ← SQLite schema (document metadata, feedback logs)
│
├── agents/                ← The reasoning layer
│   ├── loop.py               orchestrates retrieve→generate→reflect→heal
│   ├── decomposer.py          multi-hop query splitting
│   ├── reflection.py          deterministic answer-quality checks
│   ├── root_cause.py          failure_type → healing action mapping
│   ├── healer.py               applies the healing action
│   ├── judge.py                 optional LLM faithfulness check
│   ├── context_optimizer.py    MMR, compression, citations, injection screening
│   ├── router.py                picks fast vs strong model
│   ├── safety.py                 query-level prompt-injection screening
│   ├── worker.py                  Groq call + prompt construction
│   ├── query_rewriter.py           LLM-assisted retry query rewriting
│   ├── validator.py, state.py
│
├── retrieval/              ← Hybrid search
│   ├── store.py               in-memory chunk store + BM25 index
│   ├── search.py               BM25, RRF fusion, hybrid_retrieve()
│   ├── dense.py                 sentence-transformer embeddings + cache
│   └── reranker.py               cross-encoder reranking
│
├── ingestion/               ← Parsing and chunking
│   ├── parser.py               .txt/.pdf/.docx/.md/.csv/.html
│   ├── chunker.py               parent-child chunking, log-aware chunking
│   └── reingest.py               persisted-file reprocessing (REINGEST action)
│
├── cache/
│   └── semantic_cache.py     cosine-similarity query cache
│
├── chatbot/                ← Session-level query handling
│   ├── memory.py, normalizer.py, rewriter.py, intent.py
│
├── llm/
│   └── groq_manager.py       multi-key rotation, rate-limit handling
│
├── monitor/                ← Observability
│   ├── logger.py, feedback.py
│
├── models/
│   └── schemas.py            Pydantic request/response models
│
├── eval/                   ← Offline retrieval-quality regression harness
│   ├── run_eval.py, golden_queries.json, fixtures/
│
├── tests/                  ← 97 tests, unit + HTTP integration
├── .github/workflows/       ← CI
│
└── frontend/                ← React + Vite UI
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST   | `/chat` | Send a message, get an answer (full self-healing pipeline) |
| POST   | `/chat/stream` | Same, streamed via SSE |
| POST   | `/upload` | Upload a document (`.txt/.pdf/.docx/.md/.csv/.html`) |
| DELETE | `/files/{file_id}` | Remove a file and its chunks |
| GET    | `/docs-loaded` | List loaded files + chunk counts |
| POST   | `/reload` | Re-index everything in `test_docs/` |
| GET    | `/suggestions` | LLM-generated topic cards from loaded documents |
| GET    | `/stats` | Pipeline stats (model usage, latency, failure types) |
| GET    | `/health` | System status + configured models |
| GET    | `/admin/documents` | List all documents (including soft-deleted) |
| GET    | `/admin/reingestion-queue` | View the healer's REINGEST signal queue |
| POST   | `/admin/reingestion-queue/process` | Reprocess queued documents from persisted uploads, then clear the queue |
| GET    | `/admin/stats` | Combined SQL + live pipeline stats |
| DELETE | `/admin/documents/{doc_id}` | Admin document delete |
| DELETE | `/documents/clear-all` | Delete **all** documents and chunks |

### Example `/chat` request

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test-123", "message": "What is the termination clause?"}'
```

### Example response

```json
{
  "answer": "Either party may terminate with 30 days written notice... [Source: sample_contract.txt | Page 1]",
  "sources": [
    {
      "filename": "sample_contract.txt",
      "page": 1,
      "text": "Either party may terminate this Agreement...",
      "score": 0.0164
    }
  ],
  "intent": "qa",
  "model_used": "llama-3.3-70b-versatile",
  "latency_ms": 890,
  "flagged": false,
  "attempts": 1,
  "reflected": false,
  "confidence": 0.94
}
```

> **Note:** `score` is the fused RRF score from hybrid retrieval (small
> floats, ~0.005–0.05), not a bounded 0–1 similarity.

---

## Configuration

Edit `.env` to tune behaviour — see `config.py` for the full list with inline
explanations. The most commonly tuned:

| Variable | Default | Description |
|----------|---------|-------------|
| `HYBRID_ALPHA` | 0.5 | BM25 vs dense retrieval blend (0=BM25-only, 1=dense-only) |
| `FINAL_K` | 5 | Chunks sent to the LLM after reranking |
| `MAX_REFLECTION_ATTEMPTS` | 3 | Max retrieve→generate→reflect cycles per query |
| `MIN_REFLECTION_CONFIDENCE` | 0.4 | Confidence floor to accept an answer without retry |
| `SEMANTIC_CACHE_THRESHOLD` | 0.97 | Cosine similarity for a query cache hit |
| `MAX_TOTAL_CHUNKS` | 5000 | Hard cap on total stored chunks (see Known Limitations) |

---

## Testing

```bash
pip install -r requirements.txt   # includes pytest
pytest -v                          # 97 tests, ~8-10s, no API key needed
python eval/run_eval.py            # retrieval-quality report (BM25 + hybrid/dense)
```

`tests/` covers the deterministic core (reflection, healing, chunking, fusion,
caching, decomposition) as unit tests, plus HTTP-layer integration tests
against the real FastAPI app (`tests/test_api_integration.py`). None of it
needs a live Groq call — the suite mocks every LLM boundary.

---

## Models Used

| Role | Config Variable | Default | Provider |
|------|------------------|---------|----------|
| Fast worker (simple queries, query rewriting, decomposition check) | `GROQ_FAST` | `llama-3.1-8b-instant` | Groq |
| Strong worker (complex queries, healing escalation) | `GROQ_STRONG` | `llama-3.3-70b-versatile` | Groq |
| Safety classifier / suggestions / faithfulness judge | `GROQ_QWEN` | `llama-3.3-70b-versatile`* | Groq |
| Dense embeddings | `DENSE_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local |
| Cross-encoder reranker | `CROSS_ENCODER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local |
| Keyword retrieval | — | BM25 (`rank-bm25`) | Local / offline |

\* Override with an actual Qwen model ID in `.env` if available on your Groq plan.

---

## Known Limitations (Deliberate, Not Oversights)

| Limitation | Why | Revisit when |
|---|---|---|
| Dense retrieval is exact brute-force cosine, no ANN index | Sub-millisecond at the current chunk cap; avoids graph tuning/tombstoning and a new native dependency | `MAX_TOTAL_CHUNKS` is raised well past its current default (tens of thousands of chunks) |
| No OCR for scanned/image PDFs | Avoids a system-level binary dependency (Tesseract, not pip-installable) and silent low-confidence text polluting search/hallucination risk; upload returns an explicit error instead | Scanned documents become an actual input source — implement as a background job, not inline in `/upload` (a 50-page scan can take 30-60s) |
| Single-process, in-memory store (pickled to disk) | Simple, no external infra, fine for one instance | You need multiple worker processes or horizontal scaling — requires moving to a real vector DB / shared store |
| No auth on any endpoint, including destructive ones (`/documents/clear-all`) | Intentional — this is the "open/no-auth edition," see `main.py`'s module docstring | Never, unless the deployment model changes (e.g. public-facing) — then auth needs to be designed in, not bolted on |
| Tables in PDF/DOCX flatten to text | Keeps parsing dependency-light | You need to answer "sum this column"-style questions over tabular data — needs structured extraction, not text chunking |

---

## Roadmap

- [x] Hybrid retrieval — BM25 + dense embeddings + reciprocal rank fusion
- [x] Cross-encoder reranking, MMR diversification, parent-child chunking
- [x] Self-healing reflection loop with structured failure classification
- [x] Multi-hop query decomposition
- [x] Semantic query cache
- [x] Document-content prompt-injection screening
- [x] `.txt/.pdf/.docx/.md/.csv/.html` ingestion
- [x] Retrieval-quality eval harness + CI
- [ ] Real vector DB (Qdrant/pgvector) + ANN index
- [ ] Structured/tabular retrieval
- [ ] OCR (background job)
- [ ] Optional auth layer for non-local deployments
