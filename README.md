# Archiva — Self-Healing Agentic RAG for Your Documents

A self-hosted document Q&A system built with FastAPI + React. Answers questions
**strictly from your loaded documents** — hybrid retrieval, cross-encoder
reranking, and a self-healing reflection loop that retries and repairs its own
failures before ever returning an answer.

> **Open / no-auth edition.** Every endpoint is unauthenticated by design —
> this runs as a self-hosted, single-instance tool, not a multi-tenant
> service. See [Known Limitations](#known-limitations-deliberate-not-oversights)
> before deploying it anywhere other endpoints can reach.

CI: the full test suite (523 tests) runs on every push/PR via
`.github/workflows/tests.yml`, including a real Postgres service — no
Groq API key required, every LLM call in the suite is mocked.

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
Multi-Hop Decomposition ── query splits into N sub-questions ──→ run each
   │                                                              through the rest
   │ single question                                             of this pipeline
   ▼                                                              independently,
Semantic Cache lookup ── hit (cosine ≥ 0.97) ──→ Return cached result  [NO LLM]
   │ miss                                                          then merge
   ▼
Hybrid Retrieval: BM25 + Dense (sentence-transformers) → RRF fusion
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
     REWRITE_QUERY | INCREASE_TOP_K | STRICT_PROMPT | REINGEST*
        │
        └── retry (up to MAX_REFLECTION_ATTEMPTS), optionally escalating
            to the strong model, optionally consulting an LLM faithfulness
            judge on borderline-confidence answers

   * REINGEST is the one action reflection never triggers automatically —
     staleness isn't derivable from a query/answer/chunks the way the other
     three failure types are. It's reachable only by an operator (or a
     future signal source) writing OUTDATED_DATA as the failure_reason;
     see agents/reflection.py's _map_failure_type().
   │
   ▼
Validator → Response + Sources
```

---

## Quick Start

> **Setting this up on a new machine?** See [docs/SETUP.md](docs/SETUP.md) for a full
> step-by-step walkthrough — including creating the Postgres role/database
> and troubleshooting (forgotten password, port conflicts, etc.). The
> summary below assumes Postgres and `.env` are already in place.

### 1. Prerequisites

- Python 3.10+
- Node.js 18+
- A Groq API key → https://console.groq.com
- A local Postgres instance (document/chunk/feedback persistence — see
  [Database Setup](#database-setup) below)

### 2. Configure

```bash
cp .env.example .env
# GROQ_API_KEY and DATABASE_URL are both required; everything else has a
# sane default.
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
React frontend on http://localhost:3000.

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

## Database Setup

Archiva persists documents, chunks (+ embeddings), and feedback logs in
Postgres — no SQLite, no pickle file.

**Quickest:** `docker compose up -d` runs Postgres 16 with the role and
database below already created, plus an `archiva_test` database for the
Postgres tests. The connection strings are in the comment at the top of
`docker-compose.yml`.

**Native install:** use a dedicated role/database, not your Postgres
superuser:

```sql
CREATE ROLE archiva LOGIN PASSWORD 'choose_a_password';
CREATE DATABASE archiva OWNER archiva;
-- Optional: lets the Postgres tests run locally (they TRUNCATE, so they
-- only accept a database named *_test). Run as the superuser.
CREATE DATABASE archiva_test OWNER archiva;
```

Then set `DATABASE_URL` in `.env`:

```
DATABASE_URL=postgresql://archiva:choose_a_password@localhost:5432/archiva
```

That's it — `db/postgres.py`'s schema (`db/schema.sql`) is applied
automatically and idempotently every time the app starts (`init_db()` in
`main.py`'s lifespan, and at the top of `load_docs.py`). No separate
migration step, no manual `psql -f schema.sql` required, on this machine
or a fresh clone.

No pgvector, and that's deliberate, not a gap: embeddings are stored as a
plain array column and similarity search stays application-side
brute-force cosine — see `db/schema.sql`'s module docstring for the full
reasoning and the exact condition for revisiting it.

---

## Project Structure

```
rag_agentic/
│
├── main.py               ← FastAPI app: all HTTP endpoints
├── config.py              ← All settings, thresholds, and constants
│
├── db/                     ← Postgres persistence
│   ├── postgres.py            connection + schema application + data-access
│   ├── schema.sql              documents / chunks / sessions / version / OCR-lease tables
│   └── store_sync.py            loads/persists MultiDocStore; keeps workers in sync
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
│   ├── ocr.py                   OCR of scanned PDFs (RapidOCR)
│   ├── ocr_jobs.py               background OCR jobs: lifecycle, leases, recovery
│   ├── chunker.py               parent-child chunking, log-aware chunking
│   └── reingest.py               persisted-file reprocessing (REINGEST action)
│
├── cache/
│   └── semantic_cache.py     cosine-similarity query cache
│
├── chatbot/                ← Session-level query handling
│   ├── memory.py (Postgres-backed sessions), normalizer.py, rewriter.py, intent.py
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
├── eval/                   ← Offline quality regression harnesses
│   ├── run_eval.py             retrieval quality (no API key)
│   ├── run_answer_eval.py      end-to-end answer quality (live Groq calls)
│   ├── golden_queries.json, answer_baseline.json, fixtures/
│
├── tests/                  ← 523 tests, unit + HTTP integration + Postgres
├── .github/workflows/       ← CI (runs a Postgres service too)
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
| GET    | `/docs-loaded` | List loaded files + chunk counts (`ocr: true` for scans) |
| POST   | `/reload` | Re-index everything in `test_docs/` |
| GET    | `/suggestions` | LLM-generated topic cards from loaded documents |
| GET    | `/stats` | Pipeline stats (model usage, latency, failure types) |
| GET    | `/health` | System status + configured models |
| GET    | `/conversations` | Recent conversations, newest first (each keeps its last 10 turns) |
| GET    | `/conversations/{session_id}` | One conversation's turns, with each answer's saved checks |
| DELETE | `/conversations/{session_id}` | Delete a conversation |
| GET    | `/conversations/{session_id}/export` | Export as Markdown or PDF (`?format=pdf`) |
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
  "model_used": "openai/gpt-oss-120b",
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
| `OCR_ENABLED` | true | Turn background OCR of scanned PDFs on/off |
| `OCR_MAX_PAGES` | 50 | Pages OCR'd per file (beyond this, the rest is skipped) |
| `OCR_MIN_LINE_CONFIDENCE` | 0.6 | OCR lines below this confidence are dropped, not indexed |
| `STORE_SYNC_INTERVAL_S` | 2 | How often each worker checks Postgres for document changes made by other workers |
| `RATE_LIMIT_STORAGE_URI` | `memory://` | Rate-limit counter backend; use `redis://…` for one limit shared by all workers |

---

## Testing

```bash
pip install -r requirements.txt   # includes pytest
pytest -v                          # 523 tests, no API key needed
python eval/run_eval.py            # retrieval-quality report (BM25 + hybrid/dense)
```

### Answer-quality eval (live)

```bash
PYTHONIOENCODING=utf-8 python eval/run_answer_eval.py
```

Runs every case in the answer suite through the full pipeline against the
real Groq API and compares the result with `eval/answer_baseline.json`
(pass rate, tokens and latency per case, newly failing / passing cases).
It needs `GROQ_API_KEY`, so CI doesn't run it. It takes 10-15 minutes on
Groq's free tier (8000 tokens/min); short rate-limit waits are absorbed by
the worker's retry. `--only` / `--case` run a subset, and
`--save-baseline` refuses to save a run that was partial or hit provider
errors.

`tests/` covers the deterministic core (reflection, healing, chunking, fusion,
caching, decomposition) as unit tests, plus HTTP-layer integration tests
against the real FastAPI app (`tests/test_api_integration.py`) and direct
Postgres integration tests (`tests/test_postgres.py`). None of it needs a
live Groq call — the suite mocks every LLM boundary.

The Postgres-dependent tests (`test_postgres.py`, `test_api_integration.py`,
`test_api_ocr.py`, `test_coordination_db.py`) skip automatically — not fail —
when `DATABASE_URL` isn't reachable, so the rest of the suite still runs fine
without a database up. CI runs a real Postgres service, so nothing is
permanently skipped there.

**These tests `TRUNCATE` every table they touch**, so they refuse to run
against a database that looks like it holds real data: `DATABASE_URL` must
name a database ending in `_test`, be pinned to an isolated schema
(`?options=-csearch_path%3Dmy_test_schema`), or you must set
`ARCHIVA_ALLOW_DESTRUCTIVE_TESTS=1` (CI does, since its Postgres is a
throwaway container). Otherwise they skip with a message saying why.

---

## Models Used

| Role | Config Variable | Default | Provider |
|------|------------------|---------|----------|
| Fast worker (simple queries, query rewriting, decomposition, faithfulness judge) | `GROQ_FAST` | `openai/gpt-oss-20b` | Groq |
| Strong worker (complex queries, healing escalation) | `GROQ_STRONG` | `openai/gpt-oss-120b` | Groq |
| Safety classifier / suggestions | `GROQ_QWEN` | `qwen/qwen3.8-27b` | Groq |
| Dense embeddings | `DENSE_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local |
| Cross-encoder reranker | `CROSS_ENCODER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local |
| Keyword retrieval | — | BM25 (`rank-bm25`) | Local / offline |

Any chat model on your Groq plan works: reasoning models (gpt-oss, Qwen3)
get the request parameters they need automatically (see
`light_completion_params()` in `llm/groq_manager.py`). Groq retires models
from time to time; if requests fail with `model_not_found`, pick current
ones from https://console.groq.com/docs/models and set them in `.env`.

---

## Scanned PDFs (OCR)

A PDF with no extractable text is treated as a scan. `/upload` answers
immediately with `status: "processing"` and the document appears in the
Documents panel with live page progress (`OCR in progress · page 3/10`); it
becomes searchable when OCR finishes. It is not counted or retrievable
before then, and a failure (nothing readable, store full) is shown with its
reason instead of disappearing.

- **Engine:** [RapidOCR](https://github.com/RapidAI/RapidOCR) (ONNX, models bundled) + `pypdfium2` to render pages — both pip-installable, no system binary. If they are missing, scanned uploads fail with an explicit message, as before.
- **Quality policy:** lines below `OCR_MIN_LINE_CONFIDENCE` are dropped, not indexed.
- **Durable:** the raw upload is saved first and the job state lives in Postgres. If the server dies mid-OCR, a worker takes the job over once its lease (`OCR_LEASE_SECONDS`) lapses — it restarts from page 1, it does not resume mid-document.
- **Cancel:** deleting a processing document stops its job at the next page boundary, even if the job is running in a different worker.
- **Retry:** re-uploading a failed scan replaces the failed record.

---

## Multi-Hop Questions

`agents/decomposer.py` splits a question into sub-questions. When a later
step needs an earlier step's *answer*, the decomposer marks it with a `{N}`
placeholder:

> "Which vendor supplies Project Atlas, when does **that vendor's** contract renew, and is **it** before the audit?"
> → `Which vendor supplies Project Atlas?` → `What is the renewal date of {1}?` → `Is {2} before the audit date?`

Steps run in order. Each dependent step is first rewritten into a standalone
question from the real earlier answers (one small fast-model call, with a
mechanical fallback if it fails), then goes through the normal
retrieve→generate→reflect loop. If a step can't be answered, every step that
depends on it is **skipped with an explicit reason** rather than retrieving
for a question that still has an unresolved reference. Independent
sub-questions are unaffected, and the older single-pronoun anchoring
("…and what's **their** policy") still applies when there is no placeholder.

---

## Running multiple workers

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

Every worker keeps its own in-memory store for fast retrieval. What keeps
them consistent:

| State | How it is shared |
|---|---|
| Documents / chunks / embeddings | Every change bumps `store_version` in Postgres in the same transaction. Each worker checks that one number every `STORE_SYNC_INTERVAL_S` seconds and reloads only when it changed, so an upload on worker A is searchable on worker B within a couple of seconds. |
| Chat history | `chat_sessions` table (was `sessions.json`, which workers overwrote each other's turns in). An existing `sessions.json` is imported once and renamed `sessions.json.migrated`. |
| OCR jobs | A lease row per job: one owner at a time, renewed every page, taken over if it lapses. |
| Semantic cache | Cleared whenever the document set changes, locally or from another worker. |

**Not shared** (known, by design): rate-limit counters are per-worker unless
`RATE_LIMIT_STORAGE_URI` points at Redis, so the effective limit is N× the
stated one; and the live counters behind `/stats` are per-worker, so that
endpoint reflects only the worker that served the request (totals from
Postgres in `/admin/stats` are global). Workers must run on one host, or at
least share the `uploaded_docs/` directory.

---

## Known Limitations (Deliberate, Not Oversights)

| Limitation | Why | Revisit when |
|---|---|---|
| Dense retrieval is exact brute-force cosine, no ANN index (pgvector or otherwise) | Sub-millisecond at the current chunk cap; avoids graph tuning/tombstoning and a new native dependency (pgvector has no official Windows binary — see `db/schema.sql`) | `MAX_TOTAL_CHUNKS` is raised well past its current default (tens of thousands of chunks) |
| OCR covers scanned **PDFs** only (not standalone images), capped at `OCR_MAX_PAGES` pages, and low-confidence lines are dropped rather than indexed | Dropping beats indexing noise: garbled text that BM25/dense search happily retrieves invites hallucination. Standalone image upload wasn't needed yet | Image files become an input source, or scans need better accuracy (swap the engine in `ingestion/ocr.py`; it is one function) |
| Each worker holds its own full in-memory copy of the store; workers must share one host (or at least the `uploaded_docs/` directory) and one Postgres | In-memory retrieval stays sub-millisecond; workers stay consistent by polling a Postgres version counter (see [Running multiple workers](#running-multiple-workers)). Memory is multiplied by the worker count | Corpus or worker count makes N copies too expensive, or workers must span hosts — retrieval would then need to query Postgres per request (needs an ANN index) and uploads need shared storage |
| No auth on any endpoint, including destructive ones (`/documents/clear-all`) | Intentional — this is the "open/no-auth edition," see `main.py`'s module docstring | Never, unless the deployment model changes (e.g. public-facing) — then auth needs to be designed in, not bolted on |
| Multi-hop chains are capped at 4 steps, and a step can only depend on an *earlier* step's answer | Each hop is a full retrieve→generate→reflect pass, so cost and latency grow per hop, and the Groq token budget is tight | Real questions routinely need longer chains, or steps that depend on several earlier answers at once (would need a proper plan-and-execute graph) |

Tables in PDF/DOCX/CSV are extracted with real structure (not flattened
prose) — see [Tabular data](#tabular-data) below. Tables spanning
thousands of rows still get split at `PARENT_CHUNK_SIZE`, same as any
other large single-blob content (see `MAX_INGEST_CHARS` in config.py).

---

## Tabular Data

Tables in PDF (via `pdfplumber`), DOCX, and CSV are extracted as
structured `column=value` rows — not flattened into prose — and kept as
one retrievable unit (up to `PARENT_CHUNK_SIZE`) instead of being
fragmented across scattered chunks, so a retrieved table brings its
**whole** row set into the LLM's context, not just whichever row happened
to match the search query.

The hallucination-grounding check (`agents/reflection.py`) has a narrow,
bounded exception for this: a number in the answer that doesn't appear
verbatim in the source is still accepted if it exactly equals the sum of
a small set of numbers that DO appear (e.g. summing a costs column) —
without this, the system would refuse to answer "what's the total?"
questions even when the underlying arithmetic is fully grounded and
correct. A number with no such explanation is still rejected exactly as
before; this doesn't loosen the check for anything else.

This is retrieval + verified-arithmetic, not a computation engine —
there's no SQL/pandas execution layer. It relies on the LLM performing
the arithmetic itself with the full table in context, which works well
for tables that fit in context but won't scale to a table with thousands
of rows the model can't reason over directly.

---

## Roadmap

- [x] Hybrid retrieval — BM25 + dense embeddings + reciprocal rank fusion
- [x] Cross-encoder reranking, MMR diversification, parent-child chunking
- [x] Self-healing reflection loop with structured failure classification
- [x] Multi-hop query decomposition (independent + simple reference-anchored)
- [x] Semantic query cache
- [x] Document-content prompt-injection screening
- [x] `.txt/.pdf/.docx/.md/.csv/.html` ingestion
- [x] Structured/tabular extraction + retrieval, with verified-arithmetic grounding
- [x] Retrieval-quality eval harness + CI
- [x] Postgres persistence (documents, chunks + embeddings, feedback logs) — replaces SQLite + pickle
- [ ] ANN index (pgvector or otherwise) once chunk-count scale actually needs it
- [x] OCR for scanned PDFs, as a background job (RapidOCR; restart- and crash-safe)
- [ ] Optional auth layer for non-local deployments
- [x] Sequential multi-hop reasoning — dependent steps resolved from real earlier answers, failed prerequisites skip their dependents
- [x] Multi-process operation — version-synced in-memory stores, Postgres-backed chat sessions, leased OCR jobs
- [ ] Multi-host scaling (shared upload storage, shared rate-limit backend, aggregated live stats)
