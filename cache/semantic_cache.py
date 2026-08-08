"""
cache/semantic_cache.py — In-memory semantic query cache.

Production Upgrade Part 2 — Semantic Query Cache:
  - Uses the dense embedder (all-MiniLM-L6-v2) to embed incoming queries.
  - On lookup(): computes cosine similarity to all cached query embeddings.
    If max similarity >= SEMANTIC_CACHE_THRESHOLD → returns cached result.
  - On store(): adds the query embedding + result to the cache.
  - LRU eviction via OrderedDict: once MAX_SIZE is reached, oldest entry dropped.
  - Thread-safe via threading.Lock.
  - Resets on process restart (intentional — no disk persistence for v1).

Usage:
    from cache.semantic_cache import semantic_cache

    hit = semantic_cache.lookup(query)
    if hit:
        return hit
    result = run_pipeline(query)
    semantic_cache.store(query, result)
    return result
"""

import threading
from collections import OrderedDict
from typing import Optional

from config import SEMANTIC_CACHE_THRESHOLD, SEMANTIC_CACHE_MAX_SIZE


class SemanticCache:
    """
    Thread-safe in-memory semantic query cache backed by cosine similarity.
    """

    def __init__(
        self,
        threshold: float = SEMANTIC_CACHE_THRESHOLD,
        max_size:  int   = SEMANTIC_CACHE_MAX_SIZE,
    ) -> None:
        self._threshold = threshold
        self._max_size  = max_size
        self._lock      = threading.Lock()
        # OrderedDict[query_str, (embedding, result_dict)]
        self._store: OrderedDict = OrderedDict()
        self._hits   = 0
        self._misses = 0

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_embedder(self):
        """Lazy import to avoid circular dependency at module load."""
        try:
            from retrieval.dense import embedder
            return embedder
        except Exception:
            return None

    def _cosine(self, a, b) -> float:
        """Cosine similarity of two 1-D numpy arrays (both pre-normalised)."""
        try:
            import numpy as np
            return float(np.dot(a, b))
        except Exception:
            return 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def lookup(self, query: str) -> Optional[dict]:
        """
        Return the cached result if a semantically similar query was seen before.
        Returns None on a cache miss.
        """
        embedder = self._get_embedder()
        if embedder is None:
            return None

        query_vec = embedder.embed(query)
        if query_vec is None:
            return None

        with self._lock:
            best_sim = 0.0
            best_key = None

            for key, (cached_vec, _result) in self._store.items():
                sim = self._cosine(query_vec, cached_vec)
                if sim > best_sim:
                    best_sim = sim
                    best_key = key

            if best_sim >= self._threshold and best_key is not None:
                # Move to end (most-recently-used)
                self._store.move_to_end(best_key)
                self._hits += 1
                cached_result = self._store[best_key][1]
                print(
                    f"  [CACHE HIT]  [semantic] sim={best_sim:.3f} "
                    f"query='{query[:60]}'"
                )
                return cached_result

            self._misses += 1
            return None

    def store(self, query: str, result: dict) -> None:
        """
        Store a query→result pair.  Evicts the oldest entry when full.
        """
        embedder = self._get_embedder()
        if embedder is None:
            return

        query_vec = embedder.embed(query)
        if query_vec is None:
            return

        with self._lock:
            # Remove if already present (refresh position)
            if query in self._store:
                del self._store[query]

            # Evict oldest if at capacity
            if len(self._store) >= self._max_size:
                self._store.popitem(last=False)

            self._store[query] = (query_vec, result)

    def stats(self) -> dict:
        """Return hit/miss statistics for observability."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "hits":        self._hits,
                "misses":      self._misses,
                "hit_rate":    round(self._hits / total, 3) if total else 0.0,
                "cache_size":  len(self._store),
                "max_size":    self._max_size,
                "threshold":   self._threshold,
            }

    def clear(self) -> None:
        """Flush all cached entries (useful for testing or after re-ingestion)."""
        with self._lock:
            self._store.clear()
            self._hits   = 0
            self._misses = 0


# ── Module-level singleton ────────────────────────────────────────────────────

semantic_cache = SemanticCache()
