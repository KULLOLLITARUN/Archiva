from pydantic import BaseModel, ConfigDict
from typing import List, Optional


class ChunkMetadata(BaseModel):
    file_id: str
    filename: str
    file_type: str
    page: int
    section: str
    chunk_index: int
    uploaded_at: str
    # Part 3: new metadata fields
    doc_id: Optional[str] = None
    date: Optional[str] = None
    content_hash: Optional[str] = None


class Chunk(BaseModel):
    chunk_id: str
    text: str
    # Fix #5: embedding field removed from chunk dicts (BM25 needs none).
    # Keeping it here would be misleading; removed from this schema too.
    metadata: ChunkMetadata


class FileRecord(BaseModel):
    file_id: str
    filename: str
    file_type: str
    hash: str
    chunk_count: int
    uploaded_at: str
    status: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


class SourceRef(BaseModel):
    filename: str
    page: int
    text: str
    score: float


class ChatResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    answer: str
    sources: List[SourceRef]
    intent: str
    model_used: str
    latency_ms: int
    flagged: bool = False
    # ── Reflection metadata (all have defaults → backward-compatible) ──────────
    attempts: int = 1
    reflected: bool = False
    reflection_reason: str = "not_reflected"
    confidence: float = 1.0
    search_queries: List[str] = []
    # ── Part 6: extended observability (all optional → backward-compatible) ────
    failure_type: Optional[str] = None
    retrieval_latency_ms: int = 0
    reranker_scores: List[float] = []
    tokens_used: int = 0


# ── Reflection result (used internally by the loop; not a response model) ─────
class ReflectionMeta(BaseModel):
    decision: str    # accept | retry_search | retry_model | refuse
    reason: str      # human-readable explanation
    confidence: float  # 0.0–1.0
    attempt: int     # which attempt produced this result (0-based)


class MemoryEntry(BaseModel):
    query: str
    answer: str
    sources: List[SourceRef]
    intent: str
    timestamp: str


class PipelineLog(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    request_id: str
    query: str
    safety_decision: str
    retrieval_scores: List[float]
    model_used: str
    intent: str
    latency_ms: int
    flagged: bool
    timestamp: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str
    docs_loaded: int
    total_chunks: int
    model_fast: str
    model_strong: str
    model_reasoning: str


class DocsLoadedResponse(BaseModel):
    files: List[dict]
    total_files: int
    total_chunks: int


# Fix #7: response schema for the /upload endpoint
class UploadResponse(BaseModel):
    filename: str
    file_id: str
    chunk_count: int
    status: str       # "ok" | "duplicate" | "limit" | "error"
    message: str


# Fix #7: response schema for DELETE /files/{file_id}
class DeleteResponse(BaseModel):
    file_id: str
    deleted: bool
    message: str


# Fix #17: response schema for GET /stats
class StatsResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    total_queries: int
    blocked_queries: int
    flagged_responses: int
    model_usage: dict
    intent_counts: dict
    avg_latency_ms: float
    latency_samples: int
    # Reflection loop breakdown
    reflection_stats: dict = {}
    # Part 6: extended observability
    failure_types: dict = {}
    healing_actions: dict = {}
    avg_retrieval_latency_ms: float = 0.0
    total_tokens_used: int = 0
