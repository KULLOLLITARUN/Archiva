# Archiva RAG Optimization & System Tracker

## Phase 1: Core System Optimization
- [x] Streamlining LLM latency — direct reflection loop output streaming
- [x] Vector store stabilization — fixed embedding pre-compute race condition
- [x] Dependency & auth cleanup — open-access single-tenant transition
- [x] Flexible model failovers & Groq API key rotation

## Phase 2: Ingestion & Verification
- [x] Multi-format document parser — PDF, DOCX, TXT support with layout/table extraction
- [x] Document store & SQLite database sync (`/docs-loaded` accuracy)
- [x] Direct store purging for fresh document re-chunking (`/documents/clear-all`)
- [x] SQLite schema backward-compatibility for document management

## Phase 3: UI & Telemetry Refinement
- [x] Clean message UI — telemetry metadata hidden from chat bubbles
- [x] Document status indicators — real-time chunk & document header counts
