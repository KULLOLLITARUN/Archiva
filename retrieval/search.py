"""
retrieval/search.py — Hybrid BM25 + dense retrieval with RRF fusion.

Upgrade (Part 2):
  - reciprocal_rank_fusion(): merges two ranked lists using RRF.
  - hybrid_retrieve(): metadata-filtered BM25 + dense → RRF fusion.
  - Existing search(), apply_threshold(), balanced_retrieval() retained
    unchanged for backward compatibility.

Fix #4:  _tokenize() imported from ingestion.tokenizer.
Fix #18: expand_query() for short technical queries.
"""

from typing import Dict, List, Optional, Tuple

from config import BM25_THRESHOLD, TOP_K, TOP_K_BM25, TOP_K_DENSE, RRF_K
from ingestion.tokenizer import _tokenize
from retrieval.store import MultiDocStore


# ── Query expansion (Fix #18) ─────────────────────────────────────────────────

_EXPANSION_MAP = {
    "error":    "exception failure fault crash",
    "timeout":  "timed out deadline exceeded connection refused",
    "fail":     "failed failure error exception",
    "failed":   "fail failure error exception",
    "crash":    "crashed fatal exception terminated killed",
    "crashed":  "crash fatal exception terminated killed",
    "slow":     "latency delay performance degraded",
    "down":     "unavailable unreachable offline failed",
    "memory":   "ram heap oom out of memory leak",
    "disk":     "storage io filesystem",
}

_EXPANSION_TOKEN_THRESHOLD = 4


def expand_query(query: str) -> str:
    tokens = _tokenize(query)
    if len(tokens) > _EXPANSION_TOKEN_THRESHOLD:
        return query
    extra: List[str] = []
    for token in tokens:
        if token in _EXPANSION_MAP:
            extra.append(_EXPANSION_MAP[token])
    if not extra:
        return query
    return query + " " + " ".join(extra)


# ── Core BM25 search (unchanged) ─────────────────────────────────────────────

def search(
    query: str,
    store: MultiDocStore,
    file_ids: Optional[List[str]] = None,
    top_k: int = TOP_K,
) -> List[dict]:
    """
    BM25 keyword search across the full chunk pool.
    Retained unchanged for backward compatibility with the reflection loop.
    """
    bm25 = store.get_bm25_index()
    if bm25 is None:
        return []

    pool = store.get_all_chunks()

    if file_ids:
        pool = [c for c in pool if c["metadata"]["file_id"] in file_ids]
        if not pool:
            return []
        from ingestion.embedder import build_bm25_index
        bm25 = build_bm25_index(pool)

    expanded  = expand_query(query)
    tokens    = _tokenize(expanded)
    raw_scores = bm25.get_scores(tokens)

    scored = [
        {**chunk, "score": float(score)}
        for chunk, score in zip(pool, raw_scores)
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def apply_threshold(results: List[dict], threshold: float = BM25_THRESHOLD) -> List[dict]:
    """Filter results whose BM25 score falls below *threshold*."""
    return [r for r in results if r["score"] >= threshold]


def balanced_retrieval(
    query: str,
    store: MultiDocStore,
    top_per_file: int = 2,
) -> List[dict]:
    """
    For compare intent: top chunks from each file, filtered by BM25_THRESHOLD.
    Retained unchanged for backward compatibility.
    """
    from ingestion.embedder import build_bm25_index

    expanded = expand_query(query)
    tokens   = _tokenize(expanded)
    combined = []

    for file_id, _file_record in store.files.items():
        file_chunks = store.get_file_chunks(file_id)
        if not file_chunks:
            continue

        local_bm25  = build_bm25_index(file_chunks)
        raw_scores  = local_bm25.get_scores(tokens)

        scored = [
            {**chunk, "score": float(score)}
            for chunk, score in zip(file_chunks, raw_scores)
        ]

        max_score = max((s["score"] for s in scored), default=0.0)
        if max_score < BM25_THRESHOLD:
            continue

        scored.sort(key=lambda x: x["score"], reverse=True)
        combined.extend(scored[:top_per_file])

    combined.sort(key=lambda x: x["score"], reverse=True)
    return combined


# ── RRF Fusion (Part 2.2) ─────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    list_a: List[dict],
    list_b: List[dict],
    k: int = RRF_K,
) -> List[dict]:
    """
    Merge two ranked result lists using Reciprocal Rank Fusion.

    RRF score for chunk c: Σ 1 / (k + rank_i(c))  for each list i.
    Chunks are identified by their chunk_id.  A chunk present in only
    one list still gets a score from that list alone.

    Args:
        list_a: First ranked list (e.g. BM25 results, sorted desc by score).
        list_b: Second ranked list (e.g. dense results, sorted desc by score).
        k:      Rank smoothing constant (default 60 per the RRF paper).

    Returns:
        A merged list sorted by RRF score (descending), with "score" replaced
        by the fused RRF value.
    """
    # Map chunk_id → chunk dict (prefer list_a for metadata)
    chunks_by_id: Dict[str, dict] = {}
    rrf_scores:   Dict[str, float] = {}

    def _accumulate(ranked_list: List[dict]) -> None:
        for rank, chunk in enumerate(ranked_list, start=1):
            cid = chunk.get("chunk_id", "")
            if not cid:
                continue
            chunks_by_id[cid] = chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)

    _accumulate(list_a)
    _accumulate(list_b)

    fused = [
        {**chunks_by_id[cid], "score": round(rrf_scores[cid], 6)}
        for cid in rrf_scores
    ]
    fused.sort(key=lambda x: x["score"], reverse=True)
    return fused


# ── Hybrid Retrieve (Part 2.1) ────────────────────────────────────────────────

def hybrid_retrieve(
    query: str,
    store: MultiDocStore,
    top_k: int = TOP_K,
    file_ids: Optional[List[str]] = None,
    metadata_filter: Optional[dict] = None,
) -> List[dict]:
    """
    Hybrid BM25 + dense retrieval with RRF fusion.

    Pipeline:
      1. (Optional) metadata pre-filter — narrows the search space before scoring.
      2. BM25 search (TOP_K_BM25 candidates).
      3. Dense vector search (TOP_K_DENSE candidates).
      4. RRF fusion → top `top_k` results.

    Falls back gracefully:
      - If dense retrieval is unavailable, returns BM25 results directly.
      - If BM25 returns nothing, returns dense results.
      - BM25_THRESHOLD is NOT applied here — the loop's apply_threshold() is
        called separately so callers can choose.

    Args:
        query:           Raw user query.
        store:           MultiDocStore instance.
        top_k:           Final number of results to return.
        file_ids:        Restrict to these file IDs (pre-filter).
        metadata_filter: Dict of metadata key/value pairs for additional filtering.
                         Example: {"file_type": "pdf"} or {"section": "Page 3"}.

    Returns:
        Fused and ranked list of chunk dicts with "score" = RRF score.
    """
    # 1. Metadata pre-filter — build the effective file_ids scope
    effective_file_ids = list(file_ids) if file_ids else None

    if metadata_filter:
        all_chunks = store.get_all_chunks()
        filtered_ids = {
            c["metadata"]["file_id"]
            for c in all_chunks
            if all(
                c["metadata"].get(k) == v
                for k, v in metadata_filter.items()
            )
        }
        if filtered_ids:
            if effective_file_ids:
                effective_file_ids = [f for f in effective_file_ids if f in filtered_ids]
            else:
                effective_file_ids = list(filtered_ids)

    # 2. BM25 search
    bm25_results = search(
        query=query,
        store=store,
        file_ids=effective_file_ids,
        top_k=TOP_K_BM25,
    )

    # 3. Dense search (lazy import avoids circular dependency at module load)
    from retrieval.dense import dense_search
    dense_results = dense_search(
        query=query,
        store=store,
        top_k=TOP_K_DENSE,
        file_ids=effective_file_ids,
    )

    # 4. Fusion
    if not bm25_results and not dense_results:
        # Final hardening fallback: try expand_query on BM25 alone with no threshold
        expanded_query = expand_query(query)
        if expanded_query != query:
            bm25_results = search(
                query=expanded_query,
                store=store,
                file_ids=effective_file_ids,
                top_k=TOP_K_BM25,
            )
        if not bm25_results:
            return []

    if not dense_results:
        # Dense unavailable — return BM25 results directly
        return bm25_results[:top_k]

    if not bm25_results:
        return dense_results[:top_k]

    fused = reciprocal_rank_fusion(bm25_results, dense_results)
    return fused[:top_k]
