"""Tests for agents/router.py — deterministic fast/strong model routing
(no LLM, no Qwen; RULE 21)."""

from config import GROQ_FAST, GROQ_STRONG
from agents.router import route


def test_complex_keyword_routes_to_strong_regardless_of_score():
    assert route("please compare these two documents", top_score=10.0) == GROQ_STRONG


def test_long_query_routes_to_strong_regardless_of_score():
    query = " ".join(["word"] * 25)  # > ROUTER_LONG_QUERY (20) words
    assert route(query, top_score=10.0) == GROQ_STRONG


def test_low_score_routes_to_strong():
    assert route("what is the notice period", top_score=1.0) == GROQ_STRONG


def test_high_score_short_simple_query_routes_to_fast():
    assert route("what is the notice period", top_score=5.0) == GROQ_FAST


def test_score_exactly_at_threshold_routes_to_fast():
    # route() uses `< 3.0` for strong, so exactly 3.0 should be fast
    assert route("what is the notice period", top_score=3.0) == GROQ_FAST
