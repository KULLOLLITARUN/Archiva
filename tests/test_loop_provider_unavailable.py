"""
Regression tests: an LLM provider outage must be reported as an outage.

Found by the answer-quality eval: when Groq's daily token quota ran out,
call_groq() returned "Service temporarily unavailable...", the loop scored
that as a too-short answer, retried (burning more of the exhausted quota),
and finally replaced it with "Not found in the document." - telling the user
the documents didn't contain an answer when the AI service was down.
"""

import agents.loop as loop
from agents.worker import SERVICE_UNAVAILABLE_ANSWER


def _attempt_result(answer, reason="answer_too_short"):
    return {
        "answer": answer, "chunks": [{"chunk_id": "c1", "text": "t"}], "model_used": "m",
        "attempts": 1, "reflected": False, "reflection_reason": reason, "confidence": 0.0,
        "search_queries": [], "failure_type": "RETRIEVAL_FAILURE", "retrieval_latency_ms": 0,
        "reranker_scores": [6.0], "tokens_used": 0,
        "_decision": {"valid": False, "confidence": 0.0, "reason": reason,
                      "decision": "retry_search", "failure_type": "RETRIEVAL_FAILURE"},
    }


def _patch_common(monkeypatch):
    stored = []
    monkeypatch.setattr(loop, "should_decompose", lambda q: False)
    monkeypatch.setattr(loop.semantic_cache, "lookup", lambda q: None)
    monkeypatch.setattr(loop.semantic_cache, "store", lambda q, r: stored.append(q))
    return stored


def test_outage_is_reported_as_an_outage_not_as_not_found(monkeypatch):
    stored = _patch_common(monkeypatch)
    calls = []

    def fake_single_attempt(state, store, intent, force_strong, force_model):
        calls.append(state.attempt)
        return _attempt_result(SERVICE_UNAVAILABLE_ANSWER)

    monkeypatch.setattr(loop, "_single_attempt", fake_single_attempt)
    result = loop.run_reflection_loop("what zone?", store=None, intent="qa")

    assert "Not found in the document" not in result["answer"]
    assert result["answer"].startswith("Service temporarily unavailable")
    assert "not an answer about your documents" in result["answer"]
    assert result["failure_type"] == "PROVIDER_UNAVAILABLE"
    assert result["reflection_reason"] == "provider_unavailable"
    assert calls == [0]            # no retries against a dead / exhausted provider
    assert stored == []            # an outage is never cached as the answer


def test_outage_on_a_retry_also_stops_immediately(monkeypatch):
    _patch_common(monkeypatch)
    answers = [_attempt_result("too short"), _attempt_result(SERVICE_UNAVAILABLE_ANSWER)]
    calls = []

    def fake_single_attempt(state, store, intent, force_strong, force_model):
        calls.append(state.attempt)
        return answers.pop(0)

    monkeypatch.setattr(loop, "_single_attempt", fake_single_attempt)
    monkeypatch.setattr(loop, "analyze_failure", lambda state: "REWRITE_QUERY")
    monkeypatch.setattr(loop, "apply_healing", lambda state, action: None)

    result = loop.run_reflection_loop("what zone?", store=None, intent="qa")
    assert result["failure_type"] == "PROVIDER_UNAVAILABLE"
    assert calls == [0, 1]


def test_chained_question_stops_at_the_first_outage(monkeypatch):
    unavailable = {"answer": loop._PROVIDER_UNAVAILABLE_ANSWER, "failure_type": "PROVIDER_UNAVAILABLE",
                   "reflection_reason": "provider_unavailable", "chunks": []}
    calls = []

    def fake_run_reflection_loop(query, store, intent, force_model, _allow_decompose=True):
        calls.append(query)
        return unavailable

    monkeypatch.setattr(loop, "run_reflection_loop", fake_run_reflection_loop)
    result = loop._run_decomposed(["Which disk?", "What zone must {1} be in?", "What else?"],
                                  store=None, intent="qa", force_model=None)

    assert calls == ["Which disk?"]                 # later steps never hit the dead provider
    assert result["failure_type"] == "PROVIDER_UNAVAILABLE"
    assert "Not found" not in result["answer"]


def test_eval_grader_recognises_the_outage_message():
    from eval.answer_grading import is_not_found, is_unavailable
    assert is_unavailable(loop._PROVIDER_UNAVAILABLE_ANSWER)
    assert not is_not_found(loop._PROVIDER_UNAVAILABLE_ANSWER)


def test_normal_short_answer_still_goes_through_healing(monkeypatch):
    # Only the exact provider-failure text short-circuits; ordinary bad answers still retry.
    _patch_common(monkeypatch)
    calls = []

    def fake_single_attempt(state, store, intent, force_strong, force_model):
        calls.append(state.attempt)
        return _attempt_result("too short")

    monkeypatch.setattr(loop, "_single_attempt", fake_single_attempt)
    monkeypatch.setattr(loop, "analyze_failure", lambda state: "REWRITE_QUERY")
    monkeypatch.setattr(loop, "apply_healing", lambda state, action: None)
    loop.run_reflection_loop("what zone?", store=None, intent="qa")
    assert len(calls) == loop.MAX_ATTEMPTS
