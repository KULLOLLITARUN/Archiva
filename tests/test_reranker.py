"""Tests for retrieval/reranker.py — deduplication and cross-encoder
reranking with graceful score-sort fallback.

Deliberately avoids loading the real cross-encoder model (no other test
in this suite loads the ML models either): CrossEncoderReranker instances
here have their `_model`/`_available` set directly so behavior is
verified without any network/disk model load."""

from retrieval.reranker import CrossEncoderReranker, deduplicate


def _chunk(text, score=1.0):
    return {"text": text, "score": score}


# ── deduplicate ──────────────────────────────────────────────────────────────

def test_deduplicate_removes_chunks_with_same_leading_text():
    chunks = [
        _chunk("The termination clause states that either party may end..."),
        _chunk("The termination clause states that either party may end..."),
        _chunk("A completely different paragraph about pricing."),
    ]
    result = deduplicate(chunks)
    assert len(result) == 2


def test_deduplicate_preserves_order():
    chunks = [_chunk("first"), _chunk("second"), _chunk("third")]
    result = deduplicate(chunks)
    assert [c["text"] for c in result] == ["first", "second", "third"]


def test_deduplicate_is_case_and_whitespace_insensitive():
    chunks = [_chunk("  Hello World  "), _chunk("hello world")]
    assert len(deduplicate(chunks)) == 1


# ── CrossEncoderReranker fallback behavior (no model loaded) ─────────────────

def test_rerank_returns_empty_for_empty_input():
    reranker = CrossEncoderReranker()
    assert reranker.rerank(query="q", chunks=[], top_k=5) == []


def test_rerank_falls_back_to_score_sort_when_unavailable():
    reranker = CrossEncoderReranker()
    reranker._available = False  # simulate sentence-transformers missing/failed to load

    chunks = [_chunk("a"), _chunk("b"), _chunk("c")]
    result = reranker.rerank(query="q", chunks=chunks, top_k=2)
    assert result == chunks[:2]


def test_rerank_uses_model_scores_when_available():
    reranker = CrossEncoderReranker()
    reranker._available = True

    class _FakeModel:
        def predict(self, pairs):
            # Score higher for chunks whose text contains "match"
            return [1.0 if "match" in text else 0.0 for _q, text in pairs]

    reranker._model = _FakeModel()  # skip _load(); already "loaded"

    chunks = [_chunk("no relevance here"), _chunk("this is a match")]
    result = reranker.rerank(query="q", chunks=chunks, top_k=2)
    assert result[0]["text"] == "this is a match"
    assert result[0]["reranker_score"] == 1.0


def test_rerank_falls_back_on_model_predict_exception():
    reranker = CrossEncoderReranker()
    reranker._available = True

    class _BoomModel:
        def predict(self, pairs):
            raise RuntimeError("model crashed")

    reranker._model = _BoomModel()

    chunks = [_chunk("a"), _chunk("b")]
    result = reranker.rerank(query="q", chunks=chunks, top_k=2)
    assert result == chunks[:2]
