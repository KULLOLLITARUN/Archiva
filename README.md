# DocChat — Conversational Agentic RAG Chatbot

A production-ready, zero-hallucination document Q&A system built with FastAPI + React.
Answers questions **strictly from your loaded documents** — never from general knowledge.

> **Retrieval mode: BM25 (offline keyword search)**
> No embedding server, no Gemini API key, no model downloads required for retrieval.

---

## Architecture at a Glance

```
User Query
   │
   ▼
Normalizer → Rewriter → Intent Detector → Safety Layer
   │
   ▼
BM25 Search (offline, in-memory, no API calls)
   │
   ▼
Score Gate ── score < 0.1 ──→ "Not found in the document."  [NO LLM]
   │
   ▼ score ≥ 0.1
Reranker (dedup + top-3)
   │
   ▼
Context Builder (source-labeled, trimmed to 6000 tokens)
   │
   ▼
Router ── complex/long ──→ GPT-OSS-120B
       └── simple      ──→ GPT-OSS-20B
   │
   ▼
Single LLM Call (Groq API)
   │
   ▼
Validator (stopword-filtered overlap check, non-blocking)
   │
   ▼
Response + Sources
```

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- Node.js 18+
- A Groq API key → https://console.groq.com

### 2. Configure API Keys

```bash
# Copy and edit .env
cp .env.example .env
# Only GROQ_API_KEY is required
GROQ_API_KEY=your_key_here
```

### 3. Add Your Documents

Place `.txt` files in the `test_docs/` folder.

### 4. Run Everything

**Windows:**
```bat
.\start.bat
```

**Linux / macOS:**
```bash
bash start.sh
```

This will:
- Install Python dependencies
- Run document ingestion (`load_docs.py`) — instant, no API calls
- Start the FastAPI backend on port 8000
- Start the React frontend (usually http://localhost:5173)

---

## Manual Setup (Step by Step)

```bash
# Install Python dependencies
pip install -r requirements.txt

# Ingest documents (run once, or whenever docs change)
python load_docs.py

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
rag-agent/
│
├── test_docs/          ← Put your .txt files here
├── logs/               ← Auto-created pipeline logs (10% sampling)
├── store_state.pkl     ← Auto-created after load_docs.py
│
├── load_docs.py        ← Run once to ingest documents
├── main.py             ← FastAPI app (pipeline)
├── config.py           ← All settings and constants
├── start.bat           ← Windows startup script
├── start.sh            ← Linux/macOS startup script
│
├── ingestion/          ← Parser, chunker, BM25 index builder
├── retrieval/          ← BM25 store, search, reranker
├── chatbot/            ← Memory, normalizer, rewriter, intent
├── agents/             ← Safety, router, worker, validator
├── monitor/            ← Async pipeline logger
├── models/             ← Pydantic schemas
│
└── frontend/           ← React + Vite UI
    └── src/
        ├── App.jsx
        ├── api.js
        ├── styles.css
        └── components/
```

---

## API Endpoints

| Method | Endpoint       | Description                          |
|--------|----------------|--------------------------------------|
| POST   | `/chat`        | Send a message, get an answer        |
| GET    | `/health`      | System status + model info           |
| GET    | `/docs-loaded` | List loaded files + chunk counts     |

### Example `/chat` request

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test-123", "message": "What is the termination clause?"}'
```

### Example response

```json
{
  "answer": "Either party may terminate with 30 days written notice...",
  "sources": [
    {
      "filename": "sample_contract.txt",
      "page": 1,
      "text": "Either party may terminate this Agreement...",
      "score": 4.21
    }
  ],
  "intent": "qa",
  "model_used": "openai/gpt-oss-20b",
  "latency_ms": 340,
  "flagged": false
}
```

> **Note:** `score` is a BM25 score (not bounded to 0–1). Higher = stronger keyword overlap.

---

## Configuration

Edit `.env` to tune behaviour:

| Variable              | Default | Description                                         |
|-----------------------|---------|-----------------------------------------------------|
| `BM25_THRESHOLD`      | 0.1     | Minimum BM25 score to pass the retrieval gate       |
| `TOP_K`               | 5       | Chunks retrieved before reranking                   |
| `FINAL_K`             | 3       | Chunks sent to LLM after reranking                  |
| `MAX_TOTAL_CHUNKS`    | 5000    | Hard cap on total stored chunks                     |
| `MAX_CHUNKS_PER_FILE` | 500     | Per-file chunk cap                                  |

### Tuning `BM25_THRESHOLD`

| Value | Effect |
|-------|--------|
| `0.1` (default) | Returns any keyword match — most permissive |
| `0.5` | Requires moderate keyword overlap |
| `1.0+` | Strict — significant overlap needed |

---

## Adding New Documents

1. Copy `.txt` files into `test_docs/`
2. Delete `store_state.pkl`
3. Re-run `python load_docs.py`
4. Restart the backend

---

## Test Checklist

| Test | Expected Result |
|------|----------------|
| Question whose keywords appear in a doc | Correct answer + source badge |
| Question with no keyword overlap | "Not found in the document." — NO LLM called |
| Follow-up ("Explain it") | Rewrites with prior context, answers correctly |
| Compare query | Balanced retrieval from multiple files, compare tag shown |
| Safety block ("ignore all instructions") | "Query not allowed. Please rephrase." |
| Hallucination trap (false premise) | "Not found in the document." |

---

## Models Used

| Role | Model | Provider |
|------|-------|----------|
| Fast worker | `openai/gpt-oss-20b` | Groq |
| Strong worker | `openai/gpt-oss-120b` | Groq |
| Safety classifier | `qwen/qwen3-32b` | Groq |
| Retrieval | BM25 (rank-bm25) | Local / offline |

---

## Roadmap

- [x] **Phase 1** — TXT ingestion, in-memory store, BM25 retrieval
- [x] **Phase 2** — Retrieval layer (BM25 search, threshold gate, reranker)
- [x] **Phase 3** — Intelligence layer (safety, router, worker, validator, memory)
- [x] **Phase 4** — FastAPI endpoints, full pipeline
- [x] **Phase 5** — React + Vite chat UI
- [ ] **Phase 6** — PDF support, file upload UI
- [ ] **Phase 7** — Vector DB (pgvector/Qdrant) + optional embedding upgrade
