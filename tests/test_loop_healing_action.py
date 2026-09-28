"""Regression tests for healing_action propagation in agents/loop.py.

Bug: apply_healing() computed and applied a real action (REWRITE_QUERY,
INCREASE_TOP_K, STRICT_PROMPT) on every retry, but nothing ever copied that
action onto the result dict main.py logs — monitor/logger.py's
healing_actions stat always read "NONE" regardless of how much healing
actually happened. Fixed via AgentState.last_healing_action.
"""

import agents.loop as loop
from agents.state import AgentState


def test_agent_state_defaults_healing_action_to_none():
    state = AgentState()
    assert state.last_healing_action == "NONE"


def test_merge_sub_results_surfaces_a_real_healing_action_over_none():
    sub_results = [
        ("sub A?", {"answer": "a", "healing_action": "NONE", "chunks": []}),
        ("sub B?", {"answer": "b", "healing_action": "REWRITE_QUERY", "chunks": []}),
    ]
    merged = loop._merge_sub_results(sub_results)
    assert merged["healing_action"] == "REWRITE_QUERY"


def test_merge_sub_results_defaults_to_none_when_no_subanswer_healed():
    sub_results = [
        ("sub A?", {"answer": "a", "healing_action": "NONE", "chunks": []}),
        ("sub B?", {"answer": "b", "healing_action": "NONE", "chunks": []}),
    ]
    merged = loop._merge_sub_results(sub_results)
    assert merged["healing_action"] == "NONE"


def test_run_reflection_loop_surfaces_healing_action_after_a_retry(monkeypatch):
    """First attempt fails the score gate (None from _single_attempt),
    forcing a healing cycle; second attempt succeeds. The final result must
    carry the action that was actually applied, not the "NONE" default."""
    monkeypatch.setattr(loop, "should_decompose", lambda q: False)
    monkeypatch.setattr(loop.semantic_cache, "lookup", lambda q: None)
    monkeypatch.setattr(loop.semantic_cache, "store", lambda q, r: None)

    attempts = {"n": 0}

    def fake_single_attempt(state, store, intent, force_strong, force_model):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return None  # triggers the RETRIEVAL_FAILURE healing branch
        return {
            "answer": "Found it.",
            "chunks": [],
            "model_used": "test-model",
            "attempts": state.attempt + 1,
            "reflected": True,
            "reflection_reason": "ok",
            "confidence": 1.0,
            "search_queries": [],
            "failure_type": "NONE",
            "retrieval_latency_ms": 0,
            "reranker_scores": [],
            "tokens_used": 0,
            "_decision": {"valid": True, "confidence": 1.0, "reason": "ok"},
        }

    monkeypatch.setattr(loop, "_single_attempt", fake_single_attempt)
    monkeypatch.setattr(loop, "analyze_failure", lambda state: "REWRITE_QUERY")
    monkeypatch.setattr(loop, "apply_healing", lambda state, action: None)

    result = loop.run_reflection_loop("what is x?", store=object(), intent="qa")

    assert result["healing_action"] == "REWRITE_QUERY"
