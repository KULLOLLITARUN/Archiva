"""Tests for retrieval/warmup.py: warming must prime both models, and must
never raise (a failed warm-up only means the first real query pays the cost)."""

import retrieval.dense as dense
import retrieval.reranker as reranker
from retrieval.warmup import warm_up_models


def test_warm_up_primes_both_models(monkeypatch):
    seen = {"embed": [], "rerank": []}
    monkeypatch.setattr(dense.embedder, "embed", lambda text: seen["embed"].append(text))
    monkeypatch.setattr(reranker, "rerank",
                        lambda chunks, query="", final_k=1: seen["rerank"].append((chunks, query, final_k)) or chunks)

    elapsed = warm_up_models()

    assert len(seen["embed"]) == 1
    assert len(seen["rerank"]) == 1
    chunks, query, final_k = seen["rerank"][0]
    assert chunks[0]["text"] and query and final_k == 1     # a real scoring pass, not a no-op
    assert elapsed >= 0


def test_embedding_failure_does_not_stop_the_reranker_warm_up(monkeypatch):
    reranked = []

    def boom(text):
        raise RuntimeError("model download failed")

    monkeypatch.setattr(dense.embedder, "embed", boom)
    monkeypatch.setattr(reranker, "rerank", lambda *a, **k: reranked.append(1))

    warm_up_models()            # must not raise

    assert reranked == [1]


def test_reranker_failure_is_swallowed(monkeypatch):
    monkeypatch.setattr(dense.embedder, "embed", lambda text: None)

    def boom(*a, **k):
        raise RuntimeError("cross-encoder missing")

    monkeypatch.setattr(reranker, "rerank", boom)
    warm_up_models()            # must not raise


def test_tests_do_not_warm_models_on_app_start():
    # tests/conftest.py turns warm-up off so TestClient starts stay fast.
    import config
    assert config.WARM_UP_ON_START is False
