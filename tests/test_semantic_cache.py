"""Tests for cache/semantic_cache.py — cosine-similarity lookup, LRU
eviction, and hit/miss stats. The dense embedder is faked out so these
tests don't need to load sentence-transformers."""

import numpy as np

from cache.semantic_cache import SemanticCache


class FakeEmbedder:
    """Maps known query strings to fixed unit vectors; unknown queries
    get their own unique orthogonal-ish vector so they never accidentally
    collide with a cached entry."""

    def __init__(self, vectors: dict):
        self._vectors = vectors

    def embed(self, text: str):
        if text in self._vectors:
            return self._vectors[text]
        # Deterministic "unrelated" vector for anything not pre-registered.
        return np.array([0.0, 1.0, 0.0])


def _cache_with(vectors: dict, threshold: float = 0.97, max_size: int = 200) -> SemanticCache:
    cache = SemanticCache(threshold=threshold, max_size=max_size)
    cache._get_embedder = lambda: FakeEmbedder(vectors)
    return cache


def test_lookup_misses_on_empty_cache():
    cache = _cache_with({"q1": np.array([1.0, 0.0, 0.0])})
    assert cache.lookup("q1") is None
    assert cache.stats()["misses"] == 1


def test_store_then_lookup_near_duplicate_query_hits():
    original_vec = np.array([1.0, 0.0, 0.0])
    # cosine similarity with original_vec ≈ 0.995 — above the 0.97 threshold.
    near_dup_vec = np.array([0.995, 0.0998, 0.0])

    cache = _cache_with({
        "give summary of roadmap": original_vec,
        "give me a summary of the roadmap": near_dup_vec,
    })
    cache.store("give summary of roadmap", {"answer": "the roadmap says X"})

    hit = cache.lookup("give me a summary of the roadmap")
    assert hit == {"answer": "the roadmap says X"}
    assert cache.stats()["hits"] == 1


def test_dissimilar_query_is_a_miss_even_with_entries_cached():
    original_vec = np.array([1.0, 0.0, 0.0])
    unrelated_vec = np.array([0.0, 1.0, 0.0])  # cosine similarity = 0.0

    cache = _cache_with({
        "give summary of linkedin": original_vec,
        "give summary of roadmap": unrelated_vec,
    })
    cache.store("give summary of linkedin", {"answer": "linkedin summary"})

    assert cache.lookup("give summary of roadmap") is None
    assert cache.stats()["misses"] == 1


def test_lru_eviction_drops_oldest_entry_at_capacity():
    # Give each query a distinct, non-similar (orthogonal) vector.
    vectors = {
        "q0": np.array([1.0, 0.0, 0.0]),
        "q1": np.array([0.0, 1.0, 0.0]),
        "q2": np.array([0.0, 0.0, 1.0]),
    }
    cache = _cache_with(vectors, max_size=2)

    cache.store("q0", {"answer": "a0"})
    cache.store("q1", {"answer": "a1"})
    cache.store("q2", {"answer": "a2"})  # evicts q0 (oldest, capacity=2)

    assert cache.stats()["cache_size"] == 2
    assert cache.lookup("q0") is None
    assert cache.lookup("q1") == {"answer": "a1"}
    assert cache.lookup("q2") == {"answer": "a2"}


def test_stats_hit_rate_computation():
    vec = np.array([1.0, 0.0, 0.0])
    cache = _cache_with({"q": vec})
    cache.store("q", {"answer": "a"})

    cache.lookup("q")   # hit
    cache.lookup("q")   # hit

    stats = cache.stats()
    assert stats["hits"] == 2
    assert stats["hit_rate"] == 1.0
