"""Tests for agents/root_cause.py — failure_type -> healing action mapping."""

import pytest

from agents.root_cause import analyze_failure
from agents.state import AgentState


@pytest.mark.parametrize("failure_type,expected_action", [
    ("RETRIEVAL_FAILURE",    "REWRITE_QUERY"),
    ("INSUFFICIENT_CONTEXT", "INCREASE_TOP_K"),
    ("HALLUCINATION",        "STRICT_PROMPT"),
    ("OUTDATED_DATA",        "REINGEST"),
    ("FORMAT_ERROR",         "STRICT_PROMPT"),
    ("SOME_UNMAPPED_TYPE",   "NONE"),
])
def test_analyze_failure_maps_to_expected_action(failure_type, expected_action):
    state = AgentState(original_query="q", failure_type=failure_type)
    assert analyze_failure(state) == expected_action


def test_analyze_failure_with_no_failure_type_is_none():
    state = AgentState(original_query="q", failure_type=None)
    assert analyze_failure(state) == "NONE"
