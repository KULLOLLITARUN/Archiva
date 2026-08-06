"""
retrieval/reranker.py — Cross-encoder reranker with graceful BM25 fallback.

Upgrade (Part 4):
  - CrossEncoderReranker: lazy-loads cross-encoder/ms-marco-MiniLM-L-6-v2.
  - Input: up to CROSS_ENCODER_TOP_N (20) candidates from hybrid retrieval.
  - Output: top FINAL_K (5) chunks, scored and logged.
  - deduplicate() retained as pre-filter before cross-encoder.
  - Fallback: if sentence-transformers unavailable or model fails to load,
    gracefully falls back to score-sort (original behaviour).
"""

import threading
from typing import List, Optional

from config import FINAL_K, CROSS_ENCODER_MODEL, CROSS_ENCODER_TOP_N

# ── Graceful import ────────────────────────────────────────────────────────────

try:
    from sentence_transformers import CrossEncoder as _CrossEncoder
    _CE_AVAILABLE = True
except ImportError:
    _CE_AVAILABLE = False
    _CrossEncoder = None   # type: ignore[misc,assignment]


# ── Deduplication (unchanged) ─────────────────────────────────────────────────

def deduplicate(chunks: List[dict]) -> List[dict]:
    """
    Remove duplicate chunks using hash of first 300 chars.
    Preserves original order (score-sorted).
    """
    seen = set()
    unique = []
    for chunk in chunks:
        key = hash(chunk["text"][:300].lower().strip())
        if key not in seen:
            seen.add(key)
            unique.append(chunk)
    return unique


# ── Cross-encoder singleton ────────────────────────────────────────────────────

class CrossEncoderReranker:
    """
    Lazy-loading cross-encoder reranker.

    Thread-safe: model is loaded once behind a lock on first call.
    Falls back to score-sort if the model cannot be loaded.
    """

    def __init__(self, model_name: str = CROSS_ENCODER_MODEL) -> None:
        self._model_name = model_name
        self._model: Optional[object] = None
        self._lock = threading.Lock()
        self._available = _CE_AVAILABLE

    def _load(self) -> None:
        if not self._available or self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                print(f"  [DL]  [reranker] Loading cross-encoder: {self._model_name}")
                self._model = _CrossEncoder(self._model_name)
                print("  [OK]  [reranker] Cross-encoder ready.")
            except Exception as exc:
                print(f"  [WARN]  [reranker] Cross-encoder load failed: {exc} — using score-sort fallback.")
                self._available = False

    def rerank(self, query: str, chunks: List[dict], top_k: int = FINAL_K) -> List[dict]:
        """
        Score (query, chunk_text) pairs with the cross-encoder and return
        the top *top_k* chunks sorted by cross-encoder score.

        Falls back to original BM25/RRF score-sort if model is unavailable.
        """
        if not chunks:
            return chunks

        # Always deduplicate first
        deduped = deduplicate(chunks)

        if not self._available:
            return deduped[:top_k]

        self._load()

        if self._model is None:
            # Model still not loaded after attempt — use fallback
            return deduped[:top_k]

        candidates = deduped[:CROSS_ENCODER_TOP_N]

        try:
            pairs  = [[query, c["text"]] for c in candidates]
            scores = self._model.predict(pairs)   # type: ignore[union-attr]

            for chunk, score in zip(candidates, scores):
                chunk["reranker_score"] = float(score)

            candidates.sort(key=lambda c: c["reranker_score"], reverse=True)

            # Log scores for observability
            score_summary = [
                round(c["reranker_score"], 3) for c in candidates[:top_k]
            ]
            print(f"  [STATS]  [reranker] Cross-encoder top scores: {score_summary}")

            return candidates[:top_k]

        except Exception as exc:
            print(f"  [WARN]  [reranker] Cross-encoder predict failed: {exc} — using score-sort fallback.")
            return deduped[:top_k]


# ── Module-level singleton ────────────────────────────────────────────────────

_reranker = CrossEncoderReranker()


# ── Public API ────────────────────────────────────────────────────────────────

def rerank(results: List[dict], query: str = "", final_k: int = FINAL_K) -> List[dict]:
    """
    Deduplicate then cross-encode rerank the top candidates.

    Args:
        results: Candidate chunks (from hybrid or BM25 retrieval).
        query:   The current search query (needed for cross-encoder scoring).
        final_k: Number of chunks to return.

    Returns:
        Top *final_k* chunks, scored (and "reranker_score" key added if CE ran).
    """
    return _reranker.rerank(query=query, chunks=results, top_k=final_k)
