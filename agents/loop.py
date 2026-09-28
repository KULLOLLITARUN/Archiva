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

Multi-hop query decomposition:
  - run_reflection_loop() first checks whether the query is plausibly
    multiple distinct sub-questions (agents/decomposer.py). If so, each
    sub-question runs its own full retrieve->generate->reflect pass
    (recursively, with decomposition disabled) and the results are merged
    into one answer — see _maybe_decompose() / _run_decomposed().
"""

import time
from collections import defaultdict
from typing import Dict, List, Optional

from config import (
    BM25_THRESHOLD,
    CROSS_ENCODER_TOP_N,
    FINAL_K,
    GROQ_STRONG,
    JUDGE_CONFIDENCE_THRESHOLD,
    MAX_CONTEXT_TOKENS,
    MAX_REFLECTION_ATTEMPTS,
    MIN_REFLECTION_CONFIDENCE,
)
from cache.semantic_cache import semantic_cache
from agents.decomposer import anchor_to_prior_answer, decompose_query, should_decompose
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
from agents.judge import judge_faithfulness

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
    if intent in ("compare", "meta"):
        # For compare/meta intent use balanced BM25, then fuse with dense via RRF
        bm25_results = balanced_retrieval(current_query, store)
        from retrieval.dense import dense_search
        dense_results = dense_search(current_query, store, top_k=20)
        from retrieval.search import reciprocal_rank_fusion
        results = reciprocal_rank_fusion(bm25_results, dense_results)
    else:
        results = hybrid_retrieve(current_query, store, top_k=CROSS_ENCODER_TOP_N)
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

    # 3b. Parent-context expansion: replace chunk text with rich parent section
    for chunk in top_chunks:
        parent_text = chunk["metadata"].get("parent_text", "")
        if parent_text:
            chunk["_context_text"] = parent_text
        else:
            chunk["_context_text"] = chunk["text"]

    print(
        f"  [RETRIEVAL]  [loop] candidates={len(filtered)} "
        f"reranked_to={len(top_chunks)} "
        f"intent={intent}"
    )

    # 4. Context optimization — MMR + compression + citations
    optimized = optimize_context(top_chunks, query=current_query, top_k=FINAL_K)

    # 5. Build context — two-stage diversity filter:
    #    a) Deduplicate by parent_id (same section can't appear twice)
    #    b) Cap at MAX_CHUNKS_PER_DOC per file (stops one doc dominating context)
    MAX_CHUNKS_PER_DOC = 2
    if optimized:
        seen_parents: set = set()
        doc_counts: dict  = {}
        unique_by_parent: list = []

        for chunk in optimized:
            pid   = chunk["metadata"].get("parent_id") or chunk.get("chunk_id", "")
            fname = chunk["metadata"].get("filename", "")

            if pid in seen_parents:
                continue
            if doc_counts.get(fname, 0) >= MAX_CHUNKS_PER_DOC:
                continue

            seen_parents.add(pid)
            doc_counts[fname] = doc_counts.get(fname, 0) + 1
            unique_by_parent.append(chunk)

        context_chunks = unique_by_parent if unique_by_parent else optimized

        groups: dict = {}
        for chunk in context_chunks:
            fname = chunk["metadata"]["filename"]
            groups.setdefault(fname, []).append(chunk)
        for fname in groups:
            groups[fname].sort(
                key=lambda c: (c["metadata"]["page"], c["metadata"]["chunk_index"])
            )
        labeled_parts = []
        for fname, file_chunks in groups.items():
            for chunk in file_chunks:
                page = chunk["metadata"]["page"]
                ctx  = chunk.get("_context_text", chunk["text"])
                labeled_parts.append(f"[Source: {fname} | Page {page}]\n{ctx}\n")
        raw_context = "\n".join(labeled_parts)
        words = raw_context.split()
        context = " ".join(words[:MAX_CONTEXT_TOKENS])
        print(
            f"  [CONTEXT]  [loop] unique_parents={len(unique_by_parent)} "
            f"docs={list(doc_counts.keys())} total_words={len(words)}"
        )
    else:
        context = ""


    # 6. Route — respect force_strong and force_model flags
    top_score = top_chunks[0]["score"] if top_chunks else 0.0
    if force_strong:
        model_id = GROQ_STRONG
    elif force_model:
        model_id = force_model
    else:
        model_id = route(current_query, top_score)

    # 7. Generate — pass prompt_mode from state and files_summary
    files_summary = store.get_files_summary() if hasattr(store, "get_files_summary") else ""
    prompt = build_prompt(
        current_query,
        context,
        intent,
        prompt_mode=state.prompt_mode,
        files_summary=files_summary,
    )
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


# ── Multi-hop query decomposition ─────────────────────────────────────────────

def _maybe_decompose(query: str, allow_decompose: bool) -> Optional[List[str]]:
    """
    Return sub-questions to run separately, or None to run *query* as one.

    Pure decision function — no store/retrieval dependency — so the branch
    logic is unit-testable without a real store. `allow_decompose=False`
    (used on the recursive per-sub-question call) short-circuits before
    should_decompose() even runs, which is what prevents infinite recursion.
    """
    if not allow_decompose or not should_decompose(query):
        return None
    sub_queries = decompose_query(query)
    if len(sub_queries) <= 1:
        return None
    return sub_queries


def _merge_sub_results(sub_results: List[tuple]) -> Dict:
    """
    Merge per-sub-question result dicts (each shaped like a normal
    run_reflection_loop() return value) into one combined result.

    Confidence is the MIN across sub-answers (an answer is only as
    trustworthy as its weakest part); latency/tokens are summed (total
    work done); chunks are unioned and deduplicated by chunk_id.
    """
    answer_parts: List[str] = []
    all_chunks: List[dict] = []
    seen_chunk_ids: set = set()
    all_search_queries: List[str] = []
    all_reranker_scores: List[float] = []
    models_used: List[str] = []
    failure_types: List[str] = []
    healing_actions: List[str] = []

    max_attempts = 0
    any_reflected = False
    total_retrieval_ms = 0
    total_tokens = 0
    min_confidence = 1.0

    for sub_query, result in sub_results:
        answer_parts.append(f"**{sub_query}**\n{result['answer']}")

        for chunk in result.get("chunks", []):
            cid = chunk.get("chunk_id", "")
            if cid and cid in seen_chunk_ids:
                continue
            if cid:
                seen_chunk_ids.add(cid)
            all_chunks.append(chunk)

        all_search_queries.extend(result.get("search_queries", []))
        all_reranker_scores.extend(result.get("reranker_scores", []))

        model = result.get("model_used", "none")
        if model not in models_used:
            models_used.append(model)
        failure_types.append(result.get("failure_type", "UNKNOWN"))
        healing_actions.append(result.get("healing_action", "NONE"))

        max_attempts = max(max_attempts, result.get("attempts", 1))
        any_reflected = any_reflected or result.get("reflected", False)
        total_retrieval_ms += result.get("retrieval_latency_ms", 0)
        total_tokens += result.get("tokens_used", 0)
        min_confidence = min(min_confidence, result.get("confidence", 0.0))

    # UNKNOWN/NONE mean "no failure" in the per-attempt schema — surface a
    # real failure_type only if at least one sub-answer actually had one.
    distinct_failures = sorted({f for f in failure_types if f not in ("UNKNOWN", "NONE")})
    distinct_healing_actions = sorted({a for a in healing_actions if a != "NONE"})

    return {
        "answer":               "\n\n".join(answer_parts),
        "chunks":               all_chunks,
        "model_used":           "+".join(models_used) if models_used else "none",
        "attempts":             max_attempts,
        "reflected":            any_reflected,
        "reflection_reason":    f"decomposed_{len(sub_results)}_subquestions",
        "confidence":           round(min_confidence, 3),
        "search_queries":       all_search_queries,
        "failure_type":         distinct_failures[0] if distinct_failures else "UNKNOWN",
        # Mirrors model_used's "+".join() above for the same reason: a
        # decomposed query can trigger different healing actions across its
        # sub-questions, and monitor/logger.py's bucketed stat can't represent
        # a composite - it'll fall into "NONE" for a multi-action case just
        # like model_used falls into "none" for multi-model merges. The raw
        # value is still preserved here for anyone reading the log payload
        # directly instead of the bucketed /stats summary.
        "healing_action":       "+".join(distinct_healing_actions) if distinct_healing_actions else "NONE",
        "retrieval_latency_ms": total_retrieval_ms,
        "reranker_scores":      all_reranker_scores,
        "tokens_used":          total_tokens,
    }


def _run_decomposed(
    sub_queries: List[str],
    store,
    intent: str,
    force_model: Optional[str],
) -> Dict:
    """
    Run each sub-question through its own full loop, then merge.

    Sequential, not parallel: sub-question N is anchored to sub-question
    N-1's answer when it contains an unresolved pronoun (see
    agents/decomposer.py's anchor_to_prior_answer()), so a dependent chain
    like "who manages the roadmap team, and what's THEIR vacation policy"
    resolves "their" via the prior answer instead of retrieving for it
    literally. The merged result's per-question header still shows the
    ORIGINAL sub-question text, not the anchored version — the anchor is
    an internal retrieval/generation aid, not user-facing phrasing.
    """
    sub_results = []
    prior_answer = ""
    for sub_query in sub_queries:
        resolved_query = anchor_to_prior_answer(sub_query, prior_answer)
        result = run_reflection_loop(resolved_query, store, intent, force_model, _allow_decompose=False)
        sub_results.append((sub_query, result))
        prior_answer = result.get("answer", "")
    return _merge_sub_results(sub_results)


# ── Public orchestrator ───────────────────────────────────────────────────────

def run_reflection_loop(
    query: str,
    store,
    intent: str,
    force_model: Optional[str] = None,
    _allow_decompose: bool = True,
) -> Dict:
    """
    Self-healing retrieve → generate → reflect loop (up to MAX_ATTEMPTS times).

    Args:
        query:            Normalised/rewritten user query.
        store:            MultiDocStore instance.
        intent:           Detected intent (qa / explain / summarize / compare).
        force_model:      Override router model for every attempt.
        _allow_decompose: Internal — set False on the recursive per-
                           sub-question call to prevent infinite recursion.
                           Callers outside this module should never pass this.

    Returns a result dict with keys:
        answer, chunks, model_used, attempts, reflected, reflection_reason,
        confidence, search_queries, failure_type, retrieval_latency_ms,
        reranker_scores, tokens_used
    """
    # ── Multi-hop decomposition (checked before anything else — a merged
    #    result is built from independent sub-question loops, not this one) ──
    sub_queries = _maybe_decompose(query, _allow_decompose)
    if sub_queries:
        print(f"  [DECOMPOSE]  [loop] Split into {len(sub_queries)} sub-question(s): {sub_queries}")
        return _run_decomposed(sub_queries, store, intent, force_model)

    state = AgentState(
        original_query = query,
        max_attempts   = MAX_ATTEMPTS,
        top_k          = 5,
    )
    state.search_queries = [query]

    # ── Semantic cache lookup ─────────────────────────────────────────────────
    cached = semantic_cache.lookup(query)
    if cached is not None:
        return cached

    force_strong: bool      = False
    best_result: Optional[Dict] = None

    try:
        for attempt in range(MAX_ATTEMPTS):
            state.attempt = attempt

            # ── One full attempt ──────────────────────────────────────────────
            result = _single_attempt(state, store, intent, force_strong, force_model)

            # ── No results from retrieval ─────────────────────────────────────
            if result is None:
                if attempt < MAX_ATTEMPTS - 1:
                    state.failure_type   = "RETRIEVAL_FAILURE"
                    state.failure_reason = "no_results"
                    action = analyze_failure(state)
                    apply_healing(state, action)
                    state.last_healing_action = action
                    print(
                        f"  [RETRY]  [loop] No results (attempt {attempt + 1}) "
                        f"-> {action} -> {state.active_query()!r}"
                    )
                    continue

                result = _not_found_result(
                    attempt,
                    reflected=attempt > 0,
                    reason="no_results_after_retry",
                    search_queries=state.search_queries,
                )
                result["healing_action"] = state.last_healing_action
                return result

            # ── Attach search_queries ─────────────────────────────────────────
            result["search_queries"] = list(state.search_queries)
            best_result = result

            decision: Dict = result.pop("_decision")

            # ── Hard refuse: answer explicitly says not found ─────────────────
            if decision["reason"] in (
                "explicit_not_found",
                "no_chunks_retrieved",
                "answer_too_short_max_attempts",
                "ungrounded_numbers_strong_model_failed",
            ):
                best_result["answer"] = _NOT_FOUND_ANSWER
                best_result["chunks"] = []
                best_result["healing_action"] = state.last_healing_action
                return best_result

            # ── Early exit: reflection says valid + sufficient confidence ─────
            confidence = decision.get("confidence", 0.0)
            if decision["valid"] and confidence >= MIN_REFLECTION_CONFIDENCE:
                # Optional LLM judge for uncertain-but-passing answers (attempt 0 only)
                if (
                    attempt == 0
                    and JUDGE_CONFIDENCE_THRESHOLD > 0.0
                    and confidence < JUDGE_CONFIDENCE_THRESHOLD
                ):
                    verdict = judge_faithfulness(state.active_query(), result["answer"], result["chunks"])
                    if not verdict["faithful"]:
                        print(f"  [JUDGE]  [loop] Judge flagged unfaithful — escalating to healing")
                        state.failure_type   = "HALLUCINATION"
                        state.failure_reason = verdict["reason"]
                        state.answer         = result["answer"]
                        action = analyze_failure(state)
                        apply_healing(state, action)
                        state.last_healing_action = action
                        if action == "STRICT_PROMPT" and should_force_strong_model(decision, result["model_used"]):
                            force_strong = True
                        continue
                # Store successful result in semantic cache
                best_result["healing_action"] = state.last_healing_action
                semantic_cache.store(query, best_result)
                return best_result

            # ── Last attempt: return best we have ────────────────────────────
            if attempt >= MAX_ATTEMPTS - 1:
                break

            # ── Healing cycle (only runs when there are attempts remaining) ───
            state.failure_type   = decision.get("failure_type", "UNKNOWN")
            state.failure_reason = decision.get("reason", "")
            state.answer         = result["answer"]

            action = analyze_failure(state)
            apply_healing(state, action)
            state.last_healing_action = action

            # Escalate to strong model when healer returns STRICT_PROMPT
            # (covers HALLUCINATION and FORMAT_ERROR failure types)
            if action == "STRICT_PROMPT" and should_force_strong_model(decision, result["model_used"]):
                force_strong = True

            print(
                f"  [RETRY]  [loop] attempt {attempt + 1} "
                f"failure_type={state.failure_type!r} -> action={action!r}"
            )

        # ── Max attempts exhausted ────────────────────────────────────────────
        if best_result:
            best_result["reflected"] = True
            best_result["reflection_reason"] = (
                best_result["reflection_reason"] + "_max_attempts_reached"
            )
            best_result["healing_action"] = state.last_healing_action
            return best_result

    except Exception as exc:
        print(f"  [ERR]  [loop] Unexpected error: {exc}")
        if best_result:
            best_result["healing_action"] = state.last_healing_action
            semantic_cache.store(query, best_result)
            return best_result

    result = _not_found_result(
        attempt=MAX_ATTEMPTS - 1,
        reflected=True,
        reason="max_attempts_no_result",
        search_queries=state.search_queries,
    )
    result["healing_action"] = state.last_healing_action
    return result

