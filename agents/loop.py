"""
agents/loop.py — Self-healing reflection-loop orchestrator.

Upgrade (Part 1.4 — self-healing loop):
  - Uses AgentState to carry all mutable state between attempts.
  - reflection now returns failure_type + valid; root_cause maps it to action;
    healer applies the fix before the next attempt.
  - hybrid_retrieve() (BM25 + dense + RRF) replaces BM25-only search().
  - Cross-encoder rerank() receives the query for accurate scoring.
  - optimize_context() applies MMR + compression + citations post-reranking.
  - Confidence early-exit: if confidence > MIN_REFLECTION_CONFIDENCE, accept.
  - Observability: emits retrieval_latency_ms and reranker_scores to result.

Backward compat:
  - run_reflection_loop() signature and return dict keys unchanged.
  - build_labeled_context() kept for /chat/stream in main.py.
"""

import time
from collections import defaultdict
from typing import Dict, List, Optional

from config import (
    BM25_THRESHOLD,
    FINAL_K,
    GROQ_STRONG,
    MAX_CONTEXT_TOKENS,
    MAX_REFLECTION_ATTEMPTS,
    MIN_REFLECTION_CONFIDENCE,
)
from agents.state import AgentState
from agents.root_cause import analyze_failure
from agents.healer import apply_healing
from agents.context_optimizer import optimize_context
from retrieval.search import apply_threshold, balanced_retrieval, hybrid_retrieve
from retrieval.reranker import rerank
from agents.router import route
from agents.worker import build_prompt, call_groq
from agents.reflection import reflect, should_force_strong_model
from agents.query_rewriter import rewrite_for_retry

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_ATTEMPTS: int = MAX_REFLECTION_ATTEMPTS

_NOT_FOUND_ANSWER = "Not found in the document."


# ── Context builder (self-contained; used by main.py /chat/stream) ────────────

def build_labeled_context(chunks: List[dict]) -> str:
    """
    Inject source labels into context and trim to MAX_CONTEXT_TOKENS words.
    Self-contained copy — do NOT import from main.py (would be circular).
    """
    groups: Dict[str, List[dict]] = defaultdict(list)
    for chunk in chunks:
        groups[chunk["metadata"]["filename"]].append(chunk)

    for fname in groups:
        groups[fname].sort(
            key=lambda c: (c["metadata"]["page"], c["metadata"]["chunk_index"])
        )

    labeled_parts: List[str] = []
    for fname, file_chunks in groups.items():
        for chunk in file_chunks:
            page = chunk["metadata"]["page"]
            text = chunk["text"]
            labeled_parts.append(f"[Source: {fname} | Page {page}]\n{text}\n")

    context = "\n".join(labeled_parts)
    words = context.split()
    return " ".join(words[:MAX_CONTEXT_TOKENS])


# ── "No result" fast-path ─────────────────────────────────────────────────────

def _not_found_result(
    attempt: int,
    reflected: bool,
    reason: str,
    search_queries: List[str],
    failure_type: str = "RETRIEVAL_FAILURE",
) -> Dict:
    return {
        "answer":            _NOT_FOUND_ANSWER,
        "chunks":            [],
        "model_used":        "none",
        "attempts":          attempt + 1,
        "reflected":         reflected,
        "reflection_reason": reason,
        "confidence":        0.0,
        "search_queries":    search_queries,
        "failure_type":      failure_type,
        "retrieval_latency_ms": 0,
        "reranker_scores":   [],
        "tokens_used":       0,
    }


# ── Single-attempt logic ──────────────────────────────────────────────────────

def _single_attempt(
    state: AgentState,
    store,
    intent: str,
    force_strong: bool,
    force_model: Optional[str],
) -> Optional[Dict]:
    """
    Run one retrieve → rerank → optimize → generate → reflect cycle.

    Returns None if no results pass the score gate.
    Returns the full result dict (with "_decision" key) otherwise.
    """
    current_query = state.active_query()

    # 1. Hybrid retrieval (BM25 + dense + RRF)
    t_ret = time.monotonic()
    if intent == "compare":
        # For compare intent use balanced BM25, then fuse with dense via RRF
        bm25_results = balanced_retrieval(current_query, store)
        from retrieval.dense import dense_search
        dense_results = dense_search(current_query, store, top_k=20)
        from retrieval.search import reciprocal_rank_fusion
        results = reciprocal_rank_fusion(bm25_results, dense_results)
    else:
        results = hybrid_retrieve(current_query, store, top_k=state.top_k)
    retrieval_latency_ms = int((time.monotonic() - t_ret) * 1000)

    # 2. Score gate (BM25_THRESHOLD on the RRF score; RRF scores are small
    #    floats ~0.005–0.05, so we use a reduced threshold for the fused score)
    filtered = apply_threshold(results, BM25_THRESHOLD * 0.1)
    if not filtered:
        # Try pure BM25 as fallback before giving up this attempt
        from retrieval.search import search
        bm25_only = search(current_query, store, top_k=state.top_k)
        filtered = apply_threshold(bm25_only, BM25_THRESHOLD)
        if not filtered:
            return None

    # 3. Cross-encoder rerank (query needed for CE scoring)
    top_chunks = rerank(filtered, query=current_query, final_k=FINAL_K)

    # 4. Context optimization — MMR + compression + citations
    optimized = optimize_context(top_chunks, query=current_query, top_k=FINAL_K)

    # 5. Build context string
    context = build_labeled_context(optimized) if optimized else ""

    # 6. Route — respect force_strong and force_model flags
    top_score = top_chunks[0]["score"] if top_chunks else 0.0
    if force_strong:
        model_id = GROQ_STRONG
    elif force_model:
        model_id = force_model
    else:
        model_id = route(current_query, top_score)

    # 7. Generate — pass prompt_mode from state (may be "strict")
    prompt = build_prompt(current_query, context, intent, prompt_mode=state.prompt_mode)
    answer = call_groq(model_id, prompt, current_query)

    # Collect reranker scores for observability
    reranker_scores = [
        round(c.get("reranker_score", c.get("score", 0.0)), 4)
        for c in top_chunks
    ]

    # 8. Reflect — now returns failure_type + valid
    decision = reflect(current_query, answer, optimized, state.attempt, model_id)

    return {
        "answer":               answer,
        "chunks":               top_chunks,
        "model_used":           model_id,
        "attempts":             state.attempt + 1,
        "reflected":            state.attempt > 0,
        "reflection_reason":    decision["reason"],
        "confidence":           decision.get("confidence", 0.5),
        "search_queries":       [],              # filled by caller
        "failure_type":         decision.get("failure_type", "UNKNOWN"),
        "retrieval_latency_ms": retrieval_latency_ms,
        "reranker_scores":      reranker_scores,
        "tokens_used":          len(answer.split()) * 4 // 3,  # rough estimate
        # Internal
        "_decision":            decision,
    }


# ── Public orchestrator ───────────────────────────────────────────────────────

def run_reflection_loop(
    query: str,
    store,
    intent: str,
    force_model: Optional[str] = None,
) -> Dict:
    """
    Self-healing retrieve → generate → reflect loop (up to MAX_ATTEMPTS times).

    Args:
        query:        Normalised/rewritten user query.
        store:        MultiDocStore instance.
        intent:       Detected intent (qa / explain / summarize / compare).
        force_model:  Override router model for every attempt.

    Returns a result dict with keys:
        answer, chunks, model_used, attempts, reflected, reflection_reason,
        confidence, search_queries, failure_type, retrieval_latency_ms,
        reranker_scores, tokens_used
    """
    state = AgentState(
        original_query = query,
        max_attempts   = MAX_ATTEMPTS,
        top_k          = 5,
    )
    state.search_queries = [query]

    force_strong: bool     = False
    best_result: Optional[Dict] = None

    try:
        for attempt in range(MAX_ATTEMPTS):
            state.attempt = attempt

            # ── One full attempt ──────────────────────────────────────────────
            result = _single_attempt(state, store, intent, force_strong, force_model)

            # ── No results from retrieval ─────────────────────────────────────
            if result is None:
                if attempt < MAX_ATTEMPTS - 1:
                    # Apply healing: REWRITE_QUERY via LLM rewriter
                    state.failure_type   = "RETRIEVAL_FAILURE"
                    state.failure_reason = "no_results"
                    action = analyze_failure(state)
                    apply_healing(state, action)
                    print(
                        f"  [RETRY]  [loop] No results (attempt {attempt + 1}) "
                        f"→ {action} → {state.active_query()!r}"
                    )
                    continue

                return _not_found_result(
                    attempt,
                    reflected=attempt > 0,
                    reason="no_results_after_retry",
                    search_queries=state.search_queries,
                )

            # ── Attach search_queries ─────────────────────────────────────────
            result["search_queries"] = list(state.search_queries)
            best_result = result

            decision: Dict = result.pop("_decision")

            # ── Early exit on high confidence ─────────────────────────────────
            if decision["valid"] or decision.get("confidence", 0.0) >= MIN_REFLECTION_CONFIDENCE:
                return best_result

            # ── Healing cycle ─────────────────────────────────────────────────
            state.failure_type   = decision.get("failure_type", "UNKNOWN")
            state.failure_reason = decision.get("reason", "")
            state.answer         = result["answer"]

            action = analyze_failure(state)
            apply_healing(state, action)

            # Legacy decision routing still works alongside healing
            if decision["decision"] == "accept":
                return best_result

            if decision["decision"] == "refuse":
                best_result["answer"] = _NOT_FOUND_ANSWER
                best_result["chunks"] = []
                return best_result

            if decision["decision"] in ("retry_search", "retry_model"):
                if decision["decision"] == "retry_model" and \
                        should_force_strong_model(decision, result["model_used"]):
                    force_strong = True
                print(
                    f"  [RETRY]  [loop] {decision['decision']} (attempt {attempt + 1}) "
                    f"— {decision['reason']}"
                )

        # ── Max attempts exhausted ────────────────────────────────────────────
        if best_result:
            best_result["reflected"] = True
            best_result["reflection_reason"] = (
                best_result["reflection_reason"] + "_max_attempts_reached"
            )
            return best_result

    except Exception as exc:
        print(f"  [ERR]  [loop] Unexpected error: {exc}")
        if best_result:
            return best_result

    return _not_found_result(
        attempt=MAX_ATTEMPTS - 1,
        reflected=True,
        reason="max_attempts_no_result",
        search_queries=state.search_queries,
    )
