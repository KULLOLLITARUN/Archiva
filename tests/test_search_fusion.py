"""Tests for retrieval/search.py — RRF fusion, thresholding, and query
expansion. These are pure functions with no store/model dependency."""

from retrieval.search import apply_threshold, expand_query, reciprocal_rank_fusion


def _chunk(chunk_id, score=0.0, **extra):
    return {"chunk_id": chunk_id, "score": score, **extra}


def test_apply_threshold_filters_below_cutoff():
    results = [_chunk("a", 0.5), _chunk("b", 0.05), _chunk("c", 0.2)]
    kept = apply_threshold(results, threshold=0.1)
    assert [c["chunk_id"] for c in kept] == ["a", "c"]


def test_rrf_boosts_chunks_ranked_highly_in_both_lists():
    bm25  = [_chunk("x"), _chunk("shared"), _chunk("y")]
    dense = [_chunk("shared"), _chunk("z"), _chunk("x")]

    fused = reciprocal_rank_fusion(bm25, dense, k=60)
    fused_ids = [c["chunk_id"] for c in fused]

    # "shared" appears near the top of both lists, so it should fuse to rank 1.
    assert fused_ids[0] == "shared"
    assert set(fused_ids) == {"x", "shared", "y", "z"}


def test_rrf_includes_chunks_present_in_only_one_list():
    bm25  = [_chunk("only_bm25")]
    dense = [_chunk("only_dense")]

    fused = reciprocal_rank_fusion(bm25, dense)
    assert {c["chunk_id"] for c in fused} == {"only_bm25", "only_dense"}


def test_rrf_scores_are_monotonically_decreasing():
    bm25  = [_chunk("a"), _chunk("b"), _chunk("c")]
    dense = []
    fused = reciprocal_rank_fusion(bm25, dense)
    scores = [c["score"] for c in fused]
    assert scores == sorted(scores, reverse=True)


def test_expand_query_adds_synonyms_for_short_technical_queries():
    expanded = expand_query("error")
    assert expanded.startswith("error ")
    assert "exception" in expanded


def test_expand_query_leaves_long_queries_unchanged():
    query = "what is the total number of errors reported across all regions last quarter"
    assert expand_query(query) == query


def test_expand_query_leaves_unmapped_short_queries_unchanged():
    assert expand_query("hello world") == "hello world"
