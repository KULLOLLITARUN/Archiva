"""Tests for retrieval/dense.py — embedding cache behavior and the
graceful-unavailable fallback paths.

Deliberately avoids loading the real sentence-transformer model (no other
test in this suite loads the ML models either): SentenceTransformerEmbedder
instances here have `_model`/`_cache` set directly so caching logic is
verified without any network/disk model load."""

import numpy as np

import retrieval.dense as dense
from retrieval.dense import SentenceTransformerEmbedder, dense_search, precompute_embeddings


class _FakeModel:
    def __init__(self):
        self.encode_calls = []

    def encode(self, text_or_texts, normalize_embeddings=True):
        self.encode_calls.append(text_or_texts)
        if isinstance(text_or_texts, list):
            return np.array([[1.0, 0.0] for _ in text_or_texts])
        return np.array([1.0, 0.0])


class _FakeCache:
    def __init__(self):
        self._store = {}

    def get(self, key):
        return self._store.get(key)

    def set(self, key, value):
        self._store[key] = value


def _wired_embedder():
    """A SentenceTransformerEmbedder with a fake model/cache already
    'loaded', so .embed()/.embed_batch() skip real model loading."""
    e = SentenceTransformerEmbedder()
    e._model = _FakeModel()
    e._cache = _FakeCache()
    return e


# ── embed() cache behavior ─────────────────────────────────────────────────────

def test_embed_returns_none_when_dense_unavailable(monkeypatch):
    monkeypatch.setattr(dense, "_DENSE_AVAILABLE", False)
    e = SentenceTransformerEmbedder()
    assert e.embed("some text") is None


def test_embed_calls_model_on_cache_miss_and_populates_cache():
    e = _wired_embedder()
    vec = e.embed("hello")
    assert list(vec) == [1.0, 0.0]
    assert e._model.encode_calls == ["hello"]
    assert e._cache.get("hello") == [1.0, 0.0]


def test_embed_uses_cache_on_second_call_without_calling_model_again():
    e = _wired_embedder()
    e.embed("hello")
    e.embed("hello")
    assert len(e._model.encode_calls) == 1  # second call was a cache hit


# ── embed_batch() ──────────────────────────────────────────────────────────────

def test_embed_batch_returns_none_when_dense_unavailable(monkeypatch):
    monkeypatch.setattr(dense, "_DENSE_AVAILABLE", False)
    e = SentenceTransformerEmbedder()
    assert e.embed_batch(["a", "b"]) is None


def test_embed_batch_returns_none_for_empty_input():
    e = _wired_embedder()
    assert e.embed_batch([]) is None


def test_embed_batch_only_encodes_uncached_texts():
    e = _wired_embedder()
    e._cache.set("cached", [0.5, 0.5])

    result = e.embed_batch(["cached", "uncached"])
    assert result.shape == (2, 2)
    # Only the uncached text should have gone through the model
    assert e._model.encode_calls == [["uncached"]]


# ── precompute_embeddings() ─────────────────────────────────────────────────────

def test_precompute_embeddings_noop_when_dense_unavailable(monkeypatch):
    monkeypatch.setattr(dense, "_DENSE_AVAILABLE", False)
    chunks = [{"text": "a"}, {"text": "b"}]
    precompute_embeddings(chunks)
    assert "_embedding" not in chunks[0]


def test_precompute_embeddings_skips_already_embedded_chunks():
    # No chunks need embedding -> returns before touching the model at all,
    # so this is safe regardless of whether the real model is installed.
    chunks = [{"text": "a", "_embedding": np.array([1.0, 0.0])}]
    precompute_embeddings(chunks)  # should not raise / not touch network
    assert list(chunks[0]["_embedding"]) == [1.0, 0.0]


# ── dense_search() ──────────────────────────────────────────────────────────────

def test_dense_search_returns_empty_when_dense_unavailable(monkeypatch):
    monkeypatch.setattr(dense, "_DENSE_AVAILABLE", False)
    assert dense_search("query", store=object()) == []
