# Archiva — Resume Highlights

Personal prep material, not project documentation — pull whatever fits
your resume's space/format.

---

## One-line summary (for a projects list / GitHub pinned repo description)

> **Archiva** — Self-hosted RAG document Q&A system (FastAPI + React) with
> hybrid retrieval, a self-healing answer-verification loop, and multi-hop
> query reasoning. 171 automated tests, CI with a live Postgres service.

---

## Ready-to-paste resume bullets

Pick 3–5 depending on space. Ordered roughly by how much they signal (system
design first, then correctness/testing rigor, then polish).

- Built a full-stack RAG (Retrieval-Augmented Generation) document Q&A
  system from scratch — FastAPI backend, React/Vite frontend, Postgres
  persistence, Groq-hosted LLMs — with hybrid BM25 + dense-embedding
  retrieval fused via Reciprocal Rank Fusion and cross-encoder reranking.

- Designed and implemented a **self-healing answer-verification loop**: a
  fully deterministic (zero-LLM-call) pipeline of quality checks — word-
  overlap grounding, hallucinated-number detection with bounded subset-sum
  arithmetic verification, contradiction detection — that classifies
  failures and automatically retries with a broader search, a stronger
  model, or a stricter prompt before ever returning an answer to the user.

- Implemented multi-hop query decomposition with reference resolution,
  splitting compound questions ("what's the policy, and what's *their*
  exception to it?") into independently-retrieved sub-questions while
  correctly anchoring pronoun references back to the prior sub-answer.

- Migrated persistence from SQLite + an in-memory pickle store to
  Postgres (documents, chunks, embeddings, feedback logs) with an
  idempotent, auto-applied schema — verified against real service
  restarts, not just unit tests — while deliberately scoping out an ANN
  index (pgvector) as unnecessary at the project's current chunk-count
  ceiling, documenting the exact condition under which to revisit it.

- Built a 171-test automated suite (pytest) covering deterministic core
  logic, HTTP-layer integration, and real Postgres integration, wired into
  GitHub Actions CI with a live Postgres service container — zero LLM API
  calls required in CI, every model boundary mocked.

- Added a document-content prompt-injection screening layer, structured/
  tabular extraction (PDF/DOCX/CSV tables kept as retrievable units instead
  of flattened prose) with verified-arithmetic grounding for "what's the
  total?"-style questions, and a retrieval-quality regression eval harness.

- Designed and shipped a full frontend visual identity from scratch — a
  WCAG-AA-contrast-verified light editorial color system, custom SVG brand
  mark, and Fraunces/Inter typography pairing — replacing a generic
  AI-template look with a deliberate, accessible design system.

---

## Deeper talking points (for the interview questions the bullets invite)

**"Tell me about a bug you found and fixed."**
Two good ones, both found via live dogfooding rather than unit tests
missing them:
- A contradiction-detection heuristic checked the source chunk for a
  negation phrase *near* the flagged keyword, but checked the generated
  answer for a negation phrase *anywhere in the whole text* — asymmetric.
  An honest, correctly-hedged answer ("X mentions Y but does not define
  Z") got flagged as contradicting the source purely because "does not"
  appeared somewhere unrelated. Fixed by making both sides symmetric
  (proximity-windowed), with regression tests reproducing the exact false
  positive and a genuine-contradiction true positive.
- A follow-up-detection heuristic fell back to "query is short → treat as
  a follow-up," which caught complete, unrelated new questions just for
  being short and anchored them to whatever the *previous* turn happened
  to be about — dragging retrieval off-topic and producing a cascading
  false "not found." Root-caused to the same shortcut-heuristic anti-
  pattern in two different files; fixed by requiring an actual reference
  signal, not a length proxy, matching a pattern the codebase already used
  correctly elsewhere.

**"How do you keep an LLM from hallucinating?"**
Retrieval grounding is necessary but not sufficient — a model can still
compute a number that doesn't appear verbatim in the source and be
correct (e.g., summing a cost column) or incorrect (fabricated). Built a
bounded subset-sum check: a number in the answer is accepted if it
exactly equals the sum of a small set of numbers that *do* appear in the
source, otherwise rejected. This avoids the two failure modes of a naive
"is this number in the source text" check — refusing correct arithmetic,
or accepting fabricated numbers that happen to coincide with unrelated
digits.

**"Why Postgres over a vector database?"**
Made a deliberate, documented call to skip pgvector/ANN indexing: at the
project's chunk-count ceiling, brute-force cosine similarity in Python is
sub-millisecond, and pgvector has no official Windows binary. Rather than
quietly working around that, the decision and its exact revisit condition
(chunk count raised well past current default) are written directly into
the schema file's own docstring — a scoping decision made deliberately,
not a limitation stumbled into.

**"How do you test something that calls an LLM?"**
The entire deterministic core (retrieval fusion, reflection/quality
checks, chunking, decomposition, caching) is pure Python with zero LLM
calls and is unit-tested directly. Every LLM boundary is mocked at the
integration-test layer, so the 171-test suite — including HTTP and real-
Postgres integration tests — runs with no API key and no network call,
which is also why CI doesn't need a Groq key configured as a secret.

---

## Stack, for a "Technologies" line

**Backend:** Python, FastAPI, Postgres, psycopg3, sentence-transformers,
rank-bm25, cross-encoder reranking (Hugging Face), Groq API (multi-key
rotation + backoff), pytest, GitHub Actions.
**Frontend:** React, Vite, hand-written CSS design system (no UI
framework), lucide-react, Playwright (used for visual verification, not
shipped).
**Architecture concepts demonstrated:** hybrid sparse+dense retrieval,
reciprocal rank fusion, cross-encoder reranking, MMR diversification,
parent-child chunking, deterministic self-healing/retry loops, multi-hop
query decomposition, prompt-injection screening, semantic caching.
