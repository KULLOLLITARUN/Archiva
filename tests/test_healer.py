"""Tests for agents/healer.py — verifies each healing action mutates
AgentState the way agents/loop.py depends on for its next attempt."""

import json

import agents.healer as healer
import config
from agents.state import AgentState


def test_increase_top_k_steps_up_and_caps_at_max():
    state = AgentState(original_query="q", top_k=5)
    healer.apply_healing(state, "INCREASE_TOP_K")
    assert state.top_k == 10

    state.top_k = 18
    healer.apply_healing(state, "INCREASE_TOP_K")
    assert state.top_k == 20  # capped at _TOP_K_MAX

    healer.apply_healing(state, "INCREASE_TOP_K")
    assert state.top_k == 20  # stays capped


def test_strict_prompt_switches_prompt_mode():
    state = AgentState(original_query="q", prompt_mode="normal")
    healer.apply_healing(state, "STRICT_PROMPT")
    assert state.prompt_mode == "strict"


def test_unknown_action_is_a_no_op():
    state = AgentState(original_query="q", top_k=5, prompt_mode="normal")
    healer.apply_healing(state, "NOT_A_REAL_ACTION")
    assert state.top_k == 5
    assert state.prompt_mode == "normal"


def test_rewrite_query_updates_state_and_search_queries(monkeypatch):
    monkeypatch.setattr(healer, "rewrite_for_retry", lambda **kwargs: "better keyword query")

    state = AgentState(original_query="original question", search_queries=["original question"])
    healer.apply_healing(state, "REWRITE_QUERY")

    assert state.rewritten_query == "better keyword query"
    assert state.active_query() == "better keyword query"
    assert state.search_queries == ["original question", "better keyword query"]


def test_reingest_appends_record_to_queue_file(tmp_path, monkeypatch):
    queue_path = tmp_path / "reingestion_queue.jsonl"
    monkeypatch.setattr(config, "REINGESTION_QUEUE_PATH", str(queue_path))

    state = AgentState(original_query="stale data question", failure_reason="OUTDATED_DATA")
    healer.apply_healing(state, "REINGEST")

    lines = queue_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["query"] == "stale data question"
    assert record["reason"] == "OUTDATED_DATA"
