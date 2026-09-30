"""Tests for the multi-hop decomposition wiring in agents/loop.py:
_maybe_decompose (the recursion-safe branch decision) and _merge_sub_results
(combining per-sub-question results). Both are pure — no store/LLM needed."""

import agents.loop as loop


# ── _maybe_decompose ────────────────────────────────────────────────────────────

def test_maybe_decompose_returns_none_when_disallowed(monkeypatch):
    calls = []
    monkeypatch.setattr(loop, "should_decompose", lambda q: True)
    monkeypatch.setattr(loop, "decompose_query", lambda q: calls.append(q) or ["a?", "b?"])

    assert loop._maybe_decompose("a long multi part question and another one", allow_decompose=False) is None
    assert calls == []  # decompose_query must never be called when disallowed


def test_maybe_decompose_returns_none_when_prefilter_says_no(monkeypatch):
    monkeypatch.setattr(loop, "should_decompose", lambda q: False)
    monkeypatch.setattr(loop, "decompose_query", lambda q: (_ for _ in ()).throw(AssertionError("should not be called")))

    assert loop._maybe_decompose("short query", allow_decompose=True) is None


def test_maybe_decompose_returns_none_when_llm_says_single_question(monkeypatch):
    monkeypatch.setattr(loop, "should_decompose", lambda q: True)
    monkeypatch.setattr(loop, "decompose_query", lambda q: [q])

    assert loop._maybe_decompose("compare x and y across regions in detail", allow_decompose=True) is None


def test_maybe_decompose_returns_subquestions_when_llm_splits(monkeypatch):
    monkeypatch.setattr(loop, "should_decompose", lambda q: True)
    monkeypatch.setattr(loop, "decompose_query", lambda q: ["sub A?", "sub B?"])

    result = loop._maybe_decompose("question A and question B, both long enough", allow_decompose=True)
    assert result == ["sub A?", "sub B?"]


# ── _merge_sub_results ──────────────────────────────────────────────────────────

def _result(answer, chunk_ids, confidence, attempts=1, reflected=False,
            failure_type="UNKNOWN", model_used="model-a", retrieval_ms=10, tokens=5):
    return {
        "answer": answer,
        "chunks": [{"chunk_id": cid, "text": f"text for {cid}"} for cid in chunk_ids],
        "model_used": model_used,
        "attempts": attempts,
        "reflected": reflected,
        "reflection_reason": "passed_all_checks",
        "confidence": confidence,
        "search_queries": [answer],
        "failure_type": failure_type,
        "retrieval_latency_ms": retrieval_ms,
        "reranker_scores": [0.9],
        "tokens_used": tokens,
    }


def test_merge_combines_answers_with_subquestion_headers():
    sub_results = [
        ("How many vacation days?", _result("20 days.", ["c1"], confidence=0.9)),
        ("How fast must oncall respond?", _result("15 minutes.", ["c2"], confidence=0.8)),
    ]
    merged = loop._merge_sub_results(sub_results)

    assert "**How many vacation days?**" in merged["answer"]
    assert "20 days." in merged["answer"]
    assert "**How fast must oncall respond?**" in merged["answer"]
    assert "15 minutes." in merged["answer"]


def test_merge_confidence_is_the_minimum_across_subanswers():
    sub_results = [
        ("q1", _result("a1", ["c1"], confidence=0.9)),
        ("q2", _result("a2", ["c2"], confidence=0.35)),
    ]
    merged = loop._merge_sub_results(sub_results)
    assert merged["confidence"] == 0.35


def test_merge_sums_retrieval_latency_and_tokens():
    sub_results = [
        ("q1", _result("a1", ["c1"], confidence=0.9, retrieval_ms=30, tokens=10)),
        ("q2", _result("a2", ["c2"], confidence=0.9, retrieval_ms=45, tokens=20)),
    ]
    merged = loop._merge_sub_results(sub_results)
    assert merged["retrieval_latency_ms"] == 75
    assert merged["tokens_used"] == 30


def test_merge_dedupes_chunks_by_chunk_id():
    sub_results = [
        ("q1", _result("a1", ["c1", "c2"], confidence=0.9)),
        ("q2", _result("a2", ["c2", "c3"], confidence=0.9)),
    ]
    merged = loop._merge_sub_results(sub_results)
    chunk_ids = [c["chunk_id"] for c in merged["chunks"]]
    assert chunk_ids == ["c1", "c2", "c3"]


def test_merge_reflection_reason_reports_subquestion_count():
    sub_results = [
        ("q1", _result("a1", ["c1"], confidence=0.9)),
        ("q2", _result("a2", ["c2"], confidence=0.9)),
        ("q3", _result("a3", ["c3"], confidence=0.9)),
    ]
    merged = loop._merge_sub_results(sub_results)
    assert merged["reflection_reason"] == "decomposed_3_subquestions"


def test_merge_attempts_is_the_max_across_subanswers():
    sub_results = [
        ("q1", _result("a1", ["c1"], confidence=0.9, attempts=1)),
        ("q2", _result("a2", ["c2"], confidence=0.9, attempts=3)),
    ]
    merged = loop._merge_sub_results(sub_results)
    assert merged["attempts"] == 3


def test_merge_failure_type_surfaces_a_real_failure_over_unknown():
    sub_results = [
        ("q1", _result("a1", ["c1"], confidence=0.9, failure_type="UNKNOWN")),
        ("q2", _result("a2", ["c2"], confidence=0.3, failure_type="HALLUCINATION")),
    ]
    merged = loop._merge_sub_results(sub_results)
    assert merged["failure_type"] == "HALLUCINATION"


def test_merge_failure_type_is_unknown_when_all_subanswers_succeeded():
    sub_results = [
        ("q1", _result("a1", ["c1"], confidence=0.9, failure_type="UNKNOWN")),
        ("q2", _result("a2", ["c2"], confidence=0.9, failure_type="UNKNOWN")),
    ]
    merged = loop._merge_sub_results(sub_results)
    assert merged["failure_type"] == "UNKNOWN"


def test_merge_models_used_lists_distinct_models():
    sub_results = [
        ("q1", _result("a1", ["c1"], confidence=0.9, model_used="fast-model")),
        ("q2", _result("a2", ["c2"], confidence=0.9, model_used="strong-model")),
    ]
    merged = loop._merge_sub_results(sub_results)
    assert merged["model_used"] == "fast-model+strong-model"


# ── _run_decomposed (integration of the two pieces above) ──────────────────────

def test_run_decomposed_calls_loop_per_subquestion_and_merges(monkeypatch):
    calls = []

    def fake_run_reflection_loop(query, store, intent, force_model, _allow_decompose=True):
        calls.append((query, _allow_decompose))
        return _result(f"answer for {query}", [f"chunk-{query}"], confidence=0.9)

    monkeypatch.setattr(loop, "run_reflection_loop", fake_run_reflection_loop)

    merged = loop._run_decomposed(["sub A?", "sub B?"], store=None, intent="qa", force_model=None)

    assert calls == [("sub A?", False), ("sub B?", False)]
    assert "answer for sub A?" in merged["answer"]
    assert "answer for sub B?" in merged["answer"]
    assert merged["reflection_reason"] == "decomposed_2_subquestions"


def test_run_decomposed_anchors_dependent_subquestion_to_prior_answer(monkeypatch):
    calls = []

    def fake_run_reflection_loop(query, store, intent, force_model, _allow_decompose=True):
        calls.append(query)
        if "roadmap team" in query:
            return _result("Priya Shah leads the roadmap team.", ["c1"], confidence=0.9)
        return _result("Team answer", ["c2"], confidence=0.9)

    monkeypatch.setattr(loop, "run_reflection_loop", fake_run_reflection_loop)

    loop._run_decomposed(
        ["Who manages the roadmap team?", "What is their vacation policy?"],
        store=None, intent="qa", force_model=None,
    )

    assert calls[0] == "Who manages the roadmap team?"
    # Second call must be anchored to the first call's answer, since it
    # contains "their" (a reference word) - not the literal sub-question.
    assert calls[1] == (
        "What is their vacation policy? "
        "(referring to: Priya Shah leads the roadmap team.)"
    )


def test_run_decomposed_header_shows_original_subquestion_not_anchored_text(monkeypatch):
    def fake_run_reflection_loop(query, store, intent, force_model, _allow_decompose=True):
        return _result("Priya Shah leads it." if "manages" in query else "20 days.", ["c1"], confidence=0.9)

    monkeypatch.setattr(loop, "run_reflection_loop", fake_run_reflection_loop)

    merged = loop._run_decomposed(
        ["Who manages the roadmap team?", "What is their vacation policy?"],
        store=None, intent="qa", force_model=None,
    )

    # The anchor text ("(referring to: ...)") is an internal retrieval aid
    # and must never leak into the user-facing answer headers.
    assert "**What is their vacation policy?**" in merged["answer"]
    assert "(referring to:" not in merged["answer"]


def test_run_decomposed_independent_subquestions_are_not_anchored(monkeypatch):
    calls = []

    def fake_run_reflection_loop(query, store, intent, force_model, _allow_decompose=True):
        calls.append(query)
        return _result("some answer", ["c1"], confidence=0.9)

    monkeypatch.setattr(loop, "run_reflection_loop", fake_run_reflection_loop)

    loop._run_decomposed(
        ["What is the vacation policy?", "What is planned for Q3 2026?"],
        store=None, intent="qa", force_model=None,
    )

    # Neither sub-question contains a reference word, so both must run
    # exactly as written - no anchoring noise added.
    assert calls == ["What is the vacation policy?", "What is planned for Q3 2026?"]


# ── Dependent chains ({N} placeholders) ────────────────────────────────────────

def test_run_decomposed_resolves_placeholder_step_from_prior_answer(monkeypatch):
    calls = []

    def fake_run_reflection_loop(query, store, intent, force_model, _allow_decompose=True):
        calls.append(query)
        if "supplies" in query:
            return _result("Acme Corp supplies Project Atlas.", ["c1"], confidence=0.9)
        return _result("March 3.", ["c2"], confidence=0.9)

    monkeypatch.setattr(loop, "run_reflection_loop", fake_run_reflection_loop)
    seen = {}

    def fake_resolve(sub_query, answers, questions):
        seen.update(sub_query=sub_query, answers=dict(answers), questions=dict(questions))
        return "What is the renewal date of Acme Corp?"

    monkeypatch.setattr(loop, "resolve_dependent_query", fake_resolve)

    merged = loop._run_decomposed(
        ["Which vendor supplies Project Atlas?", "What is the renewal date of {1}?"],
        store=None, intent="qa", force_model=None,
    )

    assert calls == ["Which vendor supplies Project Atlas?", "What is the renewal date of Acme Corp?"]
    assert seen["answers"] == {1: "Acme Corp supplies Project Atlas."}
    # Header is user-facing text, never the raw {1} placeholder.
    assert "**What is the renewal date of the answer to step 1?**" in merged["answer"]
    assert "{1}" not in merged["answer"]


def test_run_decomposed_skips_dependent_steps_when_prerequisite_not_found(monkeypatch):
    calls = []

    def fake_run_reflection_loop(query, store, intent, force_model, _allow_decompose=True):
        calls.append(query)
        return _result(loop._NOT_FOUND_ANSWER, [], confidence=0.0)

    monkeypatch.setattr(loop, "run_reflection_loop", fake_run_reflection_loop)
    monkeypatch.setattr(loop, "resolve_dependent_query",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not resolve")))

    merged = loop._run_decomposed(
        ["Which vendor supplies Atlas?", "Renewal date of {1}?", "Is {2} before the audit?"],
        store=None, intent="qa", force_model=None,
    )

    # Only step 1 ever hits retrieval; 2 is skipped, and 3 is skipped because
    # step 2 (which it depends on) was itself skipped.
    assert calls == ["Which vendor supplies Atlas?"]
    assert "depends on step 1" in merged["answer"]
    assert "depends on step 2" in merged["answer"]


def test_run_decomposed_independent_step_still_runs_after_failed_unrelated_step(monkeypatch):
    calls = []

    def fake_run_reflection_loop(query, store, intent, force_model, _allow_decompose=True):
        calls.append(query)
        answer = loop._NOT_FOUND_ANSWER if "Atlas" in query else "20 days."
        return _result(answer, [], confidence=0.5)

    monkeypatch.setattr(loop, "run_reflection_loop", fake_run_reflection_loop)

    loop._run_decomposed(
        ["Which vendor supplies Atlas?", "What is the vacation policy?"],
        store=None, intent="qa", force_model=None,
    )
    assert calls == ["Which vendor supplies Atlas?", "What is the vacation policy?"]


# ── Top-level run_reflection_loop wiring ────────────────────────────────────────

def test_run_reflection_loop_routes_to_decomposition_when_split(monkeypatch):
    monkeypatch.setattr(loop, "should_decompose", lambda q: True)
    monkeypatch.setattr(loop, "decompose_query", lambda q: ["sub A?", "sub B?"])

    sentinel = {"answer": "merged", "reflection_reason": "decomposed_2_subquestions"}
    recorded = {}

    def fake_run_decomposed(sub_queries, store, intent, force_model):
        recorded["sub_queries"] = sub_queries
        return sentinel

    monkeypatch.setattr(loop, "_run_decomposed", fake_run_decomposed)

    result = loop.run_reflection_loop("q1 and q2, long enough to pass the prefilter", store=None, intent="qa")

    assert result is sentinel
    assert recorded["sub_queries"] == ["sub A?", "sub B?"]


def test_run_reflection_loop_does_not_decompose_on_recursive_call(monkeypatch):
    monkeypatch.setattr(loop, "should_decompose", lambda q: True)

    def _fail_if_called(q):
        raise AssertionError("decompose_query must not run when _allow_decompose=False")
    monkeypatch.setattr(loop, "decompose_query", _fail_if_called)

    def _fail_if_called_2(*a, **kw):
        raise AssertionError("_run_decomposed must not run when _allow_decompose=False")
    monkeypatch.setattr(loop, "_run_decomposed", _fail_if_called_2)

    # semantic_cache.lookup returning a cached result lets us observe that
    # execution proceeded past the decomposition check without needing a
    # real store for the rest of the pipeline.
    monkeypatch.setattr(loop.semantic_cache, "lookup", lambda q: {"answer": "cached"})

    result = loop.run_reflection_loop(
        "q1 and q2, long enough to pass the prefilter", store=None, intent="qa",
        _allow_decompose=False,
    )
    assert result == {"answer": "cached"}
