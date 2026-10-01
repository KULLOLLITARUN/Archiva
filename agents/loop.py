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
    MAX_CHUNKS_PER_DOC,
    MAX_CONTEXT_TOKENS,
    MAX_REFLECTION_ATTEMPTS,
    MIN_REFLECTION_CONFIDENCE,
    NOT_FOUND_RECHECK_MIN_CE_SCORE,
)
from cache.semantic_cache import semantic_cache
from agents.decomposer import (
    anchor_to_prior_answer,
    decompose_query,
    dependencies_of,
    display_query,
    resolve_dependent_query,
    should_decompose,
    step_heading,
)
from agents.state import AgentState
from agents.root_cause import analyze_failure
from agents.healer import apply_healing
from agents.context_optimizer import optimize_context
from retrieval.search import apply_threshold, balanced_retrieval, hybrid_retrieve
from retrieval.reranker import rerank
from agents.router import route
from agents.worker import SERVICE_UNAVAILABLE_ANSWER, build_prompt, call_groq
from agents.reflection import reflect, should_force_strong_model
from agents.query_rewriter import rewrite_for_retry
from agents.judge import judge_faithfulness

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_ATTEMPTS: int = MAX_REFLECTION_ATTEMPTS

_NOT_FOUND_ANSWER = "Not found in the document."

# Shown when the LLM provider is down or out of quota. Starts with the worker's
# own wording so anything that already recognises it (the answer eval, logs)
# still does, then says plainly that this is not a statement about the documents.
_PROVIDER_UNAVAILABLE_ANSWER = (
    "Service temporarily unavailable: the AI provider could not be reached or its "
    "rate limit / daily quota was reached. This is not an answer about your "
    "documents - please try again in a few minutes."
)


def _is_provider_unavailable(answer: str) -> bool:
    return (answer or "").strip().startswith("Service temporarily unavailable")


def _provider_unavailable_result(attempt: int, search_queries: List[str]) -> Dict:
    result = _not_found_result(
        attempt, reflected=attempt > 0, reason="provider_unavailable",
        search_queries=search_queries, failure_type="PROVIDER_UNAVAILABLE",
    )
    result["answer"] = _PROVIDER_UNAVAILABLE_ANSWER
    return result


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


# ── Context chunk selection ───────────────────────────────────────────────────

def _select_context_chunks(
    chunks: List[dict],
    max_per_doc: int = MAX_CHUNKS_PER_DOC,
    limit: int = FINAL_K,
) -> List[dict]:
    """
    Choose which retrieved chunks go into the LLM context (input is in rank order).

      1. A parent section is only included once (child chunks of the same
         parent would repeat the same text).
      2. Diversity first: at most *max_per_doc* chunks per document, so one
         document doesn't crowd out the others.
      3. Then backfill: slots still free (up to *limit*) are filled, in rank
         order, with chunks step 2 skipped. The per-document cap is a
         preference, never a reason to leave room unused - it used to drop
         the one chunk naming the thing asked about whenever a single
         document supplied the whole top-5.
    """
    seen_parents: set = set()
    per_doc: dict = {}
    picked: List[dict] = []
    skipped: List[dict] = []

    for chunk in chunks:
        pid   = chunk["metadata"].get("parent_id") or chunk.get("chunk_id", "")
        fname = chunk["metadata"].get("filename", "")
        if pid in seen_parents:
            continue
        seen_parents.add(pid)
        if per_doc.get(fname, 0) >= max_per_doc:
            skipped.append(chunk)
            continue
        per_doc[fname] = per_doc.get(fname, 0) + 1
        picked.append(chunk)

    for chunk in skipped:
        if len(picked) >= limit:
            break
        picked.append(chunk)
    return picked


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

    # 5. Build context - see _select_context_chunks (parent dedupe, per-document
    #    diversity preference, then backfill to FINAL_K).
    if optimized:
        unique_by_parent = _select_context_chunks(optimized)
        doc_counts: dict = {}
        for chunk in unique_by_parent:
            fname = chunk["metadata"].get("filename", "")
            doc_counts[fname] = doc_counts.get(fname, 0) + 1

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


def _should_recheck_not_found(
    result: Dict,
    already_rechecked: bool,
    force_model: Optional[str],
    attempt: int,
) -> bool:
    """
    True if a "not found" answer deserves one more try with the strong model:
    retrieval found strongly relevant text (best cross-encoder score at or
    above NOT_FOUND_RECHECK_MIN_CE_SCORE), we haven't already re-asked, the
    strong model wasn't the one that just said it, nobody pinned the model,
    and an attempt is still available.
    """
    if already_rechecked or force_model or attempt >= MAX_ATTEMPTS - 1:
        return False
    if result.get("model_used") == GROQ_STRONG:
        return False
    scores = result.get("reranker_scores") or []
    return bool(scores) and max(scores) >= NOT_FOUND_RECHECK_MIN_CE_SCORE


def _run_decomposed(
    sub_queries: List[str],
    store,
    intent: str,
    force_model: Optional[str],
    original_query: Optional[str] = None,
) -> Dict:
    """
    Run each sub-question through its own full loop, then merge.

    Sequential, not parallel, because later steps can depend on earlier
    answers in two ways (see agents/decomposer.py):

      * "{N}" placeholders — a genuine dependency chain. The step is first
        rewritten into a standalone question from the real answers it
        depends on (resolve_dependent_query). If any prerequisite step could
        not be answered, the step is SKIPPED with an explicit reason instead
        of retrieving for a question that still has an unresolved reference
        — and that skip propagates down the rest of the chain.
      * A loose pronoun ("their", "it") with no placeholder — anchored to
        the previous answer as before (anchor_to_prior_answer).

    The merged result's per-question header shows user-facing text, not the
    internal retrieval query: original wording for plain/anchored steps, the
    cleanly rewritten standalone question for placeholder steps (or "the
    answer to step N" if the rewrite fell back to pasting answers in).
    """
    sub_results = []
    prior_answer = ""
    answers: Dict[int, str] = {}      # 1-based step -> its answer
    questions: Dict[int, str] = {}    # 1-based step -> resolved question text
    unanswered: set = set()           # 1-based steps with no usable answer

    for step, sub_query in enumerate(sub_queries, start=1):
        deps = dependencies_of(sub_query)
        blocked_by = sorted(d for d in deps if d in unanswered)
        header = display_query(sub_query) if deps else sub_query

        if blocked_by:
            print(f"  [DECOMPOSE]  [loop] Skipping step {step}: depends on unanswered step(s) {blocked_by}")
            result = _not_found_result(
                attempt=0,
                reflected=False,
                reason="dependency_unresolved",
                search_queries=[],
            )
            result["answer"] = (
                "Could not answer this step because it depends on step "
                + ", ".join(str(d) for d in blocked_by)
                + ", which was not found in the documents."
            )
            unanswered.add(step)
            sub_results.append((header, result))
            questions[step] = header
            prior_answer = ""
            continue

        if deps:
            resolved_query = resolve_dependent_query(sub_query, answers, questions, original_query)
            header = step_heading(sub_query, resolved_query, answers)
            print(f"  [DECOMPOSE]  [loop] Step {step} resolved: {resolved_query!r}")
        else:
            resolved_query = anchor_to_prior_answer(sub_query, prior_answer)

        result = run_reflection_loop(resolved_query, store, intent, force_model, _allow_decompose=False)
        if _is_provider_unavailable(result.get("answer", "")):
            # No point asking the remaining steps: the provider is down for all of them.
            print(f"  [ERR]  [loop] Provider unavailable at step {step} - abandoning the chain")
            return result
        sub_results.append((header, result))

        answer = result.get("answer", "")
        answers[step] = answer
        questions[step] = resolved_query if deps else sub_query
        if not answer or answer.strip() == _NOT_FOUND_ANSWER:
            unanswered.add(step)
        prior_answer = answer

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
        return _run_decomposed(sub_queries, store, intent, force_model, original_query=query)

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
    rechecked_not_found     = False   # at most one strong-model second look per query
    best_result: Optional[Dict] = None

    try:
        for attempt in range(MAX_ATTEMPTS):
            state.attempt = attempt

            # ── One full attempt ──────────────────────────────────────────────
            result = _single_attempt(state, store, intent, force_strong, force_model)

            # ── Provider outage / quota: stop, and say so ─────────────────────
            # Retrying just burns more of an exhausted quota, and the old path
            # (too-short answer -> retries -> hard refuse) ended in "Not found
            # in the document", telling the user the documents lack an answer
            # when the real problem was the AI service. Never cached.
            if result is not None and result.get("answer") == SERVICE_UNAVAILABLE_ANSWER:
                print("  [ERR]  [loop] LLM provider unavailable - stopping without retries")
                unavailable = _provider_unavailable_result(attempt, list(state.search_queries))
                unavailable["healing_action"] = state.last_healing_action
                return unavailable

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

            # ── "Not found" despite strong retrieval: second look, strong model ─
            if (
                decision["reason"] == "explicit_not_found"
                and _should_recheck_not_found(
                    result, rechecked_not_found, force_model, attempt,
                )
            ):
                rechecked_not_found = True
                force_strong = True
                print(
                    f"  [RECHECK]  [loop] Model said not-found but retrieval is strong "
                    f"(top CE {max(result['reranker_scores']):.2f}) -> re-asking {GROQ_STRONG}"
                )
                continue

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

