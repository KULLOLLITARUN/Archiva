"""
Tests for the "not found despite strong retrieval" second look in
agents/loop.py.

Found live: for "What region and availability zone must Disk-Clone-1 be in?"
the top retrieved chunk said "The managed disk must be created in Central
India, Zone 1", yet the fast model answered "Not found in the document." and
the loop treated that as final. The loop now re-asks the strong model once
when retrieval was clearly strong, and still refuses fast when it wasn't.
"""

import agents.loop as loop
from config import GROQ_STRONG

FAST = "fast-model"
NOT_FOUND_DECISION = {"valid": False, "confidence": 0.0, "reason": "explicit_not_found",
                      "decision": "refuse", "failure_type": "RETRIEVAL_FAILURE"}
GOOD_DECISION = {"valid": True, "confidence": 0.9, "reason": "passed_all_checks",
                 "decision": "accept", "failure_type": "UNKNOWN"}


def _result(answer, decision, scores, model=FAST):
    return {
        "answer": answer, "chunks": [{"chunk_id": "c1", "text": "t"}], "model_used": model,
        "attempts": 1, "reflected": False, "reflection_reason": decision["reason"],
        "confidence": decision["confidence"], "search_queries": [], "failure_type": decision["failure_type"],
        "retrieval_latency_ms": 0, "reranker_scores": scores, "tokens_used": 0,
        "_decision": dict(decision),
    }


def _run(monkeypatch, responses, force_model=None):
    """Drive run_reflection_loop with scripted _single_attempt results; return (result, calls)."""
    monkeypatch.setattr(loop, "should_decompose", lambda q: False)
    monkeypatch.setattr(loop.semantic_cache, "lookup", lambda q: None)
    monkeypatch.setattr(loop.semantic_cache, "store", lambda q, r: None)
    calls = []
    queue = list(responses)

    def fake_single_attempt(state, store, intent, force_strong, force_model_arg):
        calls.append({"attempt": state.attempt, "force_strong": force_strong})
        return queue.pop(0)

    monkeypatch.setattr(loop, "_single_attempt", fake_single_attempt)
    result = loop.run_reflection_loop("a question", store=None, intent="qa", force_model=force_model)
    return result, calls


def test_strong_retrieval_plus_not_found_retries_once_with_the_strong_model(monkeypatch):
    result, calls = _run(monkeypatch, [
        _result("Not found in the document.", NOT_FOUND_DECISION, [6.4, 2.1]),
        _result("Central India, Zone 1.", GOOD_DECISION, [6.4, 2.1], model=GROQ_STRONG),
    ])
    assert result["answer"] == "Central India, Zone 1."
    assert [c["force_strong"] for c in calls] == [False, True]


def test_weak_retrieval_plus_not_found_refuses_immediately(monkeypatch):
    # Genuinely unanswerable questions score about -10 on the cross-encoder.
    result, calls = _run(monkeypatch, [
        _result("Not found in the document.", NOT_FOUND_DECISION, [-10.5, -11.0]),
    ])
    assert result["answer"] == "Not found in the document."
    assert len(calls) == 1


def test_a_lexically_similar_but_absent_topic_is_not_rechecked(monkeypatch):
    # "Dev-Clone-7" scored 1.46 live: shares words with real text, answer isn't there.
    result, calls = _run(monkeypatch, [
        _result("Not found in the document.", NOT_FOUND_DECISION, [1.46, 1.01]),
    ])
    assert result["answer"] == "Not found in the document." and len(calls) == 1


def test_recheck_happens_at_most_once(monkeypatch):
    result, calls = _run(monkeypatch, [
        _result("Not found in the document.", NOT_FOUND_DECISION, [6.0]),
        _result("Not found in the document.", NOT_FOUND_DECISION, [6.0], model=GROQ_STRONG),
    ])
    assert result["answer"] == "Not found in the document."
    assert len(calls) == 2


def test_no_recheck_when_the_strong_model_already_said_not_found(monkeypatch):
    result, calls = _run(monkeypatch, [
        _result("Not found in the document.", NOT_FOUND_DECISION, [6.0], model=GROQ_STRONG),
    ])
    assert result["answer"] == "Not found in the document." and len(calls) == 1


def test_no_recheck_when_the_caller_pinned_a_model(monkeypatch):
    result, calls = _run(monkeypatch, [
        _result("Not found in the document.", NOT_FOUND_DECISION, [6.0]),
    ], force_model="pinned-model")
    assert len(calls) == 1


def test_a_normal_good_answer_is_unaffected(monkeypatch):
    result, calls = _run(monkeypatch, [_result("A fine answer.", GOOD_DECISION, [6.0])])
    assert result["answer"] == "A fine answer." and len(calls) == 1


def test_threshold_boundary(monkeypatch):
    monkeypatch.setattr(loop, "NOT_FOUND_RECHECK_MIN_CE_SCORE", 3.0)
    at = _result("x", NOT_FOUND_DECISION, [3.0])
    below = _result("x", NOT_FOUND_DECISION, [2.99])
    assert loop._should_recheck_not_found(at, False, None, 0) is True
    assert loop._should_recheck_not_found(below, False, None, 0) is False


def test_no_recheck_without_scores_or_without_attempts_left(monkeypatch):
    assert loop._should_recheck_not_found(_result("x", NOT_FOUND_DECISION, []), False, None, 0) is False
    strong = _result("x", NOT_FOUND_DECISION, [6.0])
    assert loop._should_recheck_not_found(strong, False, None, loop.MAX_ATTEMPTS - 1) is False
