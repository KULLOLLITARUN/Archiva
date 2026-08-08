"""Tests for agents/reflection.py — the deterministic answer-quality checks
that drive the self-healing loop's retry/refuse decisions."""

from agents.reflection import reflect, should_force_strong_model
from config import GROQ_STRONG

WEAK_MODEL = "llama-3.1-8b-instant"

CHUNKS = [
    {"text": (
        "The system report shows 1000 users and 50 % growth this quarter "
        "across all regions, driven mostly by the mobile platform rollout."
    )}
]


def test_explicit_not_found_is_refused():
    decision = reflect("what is x", "Not found in the document.", CHUNKS, attempt=0, model_used=WEAK_MODEL)
    assert decision["decision"] == "refuse"
    assert decision["reason"] == "explicit_not_found"
    assert decision["valid"] is False


def test_short_answer_retries_before_max_attempts():
    decision = reflect("q", "Too short.", CHUNKS, attempt=0, model_used=WEAK_MODEL)
    assert decision["decision"] == "retry_search"
    assert decision["reason"] == "answer_too_short"


def test_short_answer_refuses_at_max_attempts():
    decision = reflect("q", "Too short.", CHUNKS, attempt=2, model_used=WEAK_MODEL)
    assert decision["decision"] == "refuse"
    assert decision["reason"] == "answer_too_short_max_attempts"


def test_no_chunks_is_refused():
    decision = reflect(
        "q", "This is a sufficiently long answer with plenty of words in it.",
        [], attempt=0, model_used=WEAK_MODEL,
    )
    assert decision["decision"] == "refuse"
    assert decision["reason"] == "no_chunks_retrieved"


def test_low_overlap_retries_search_then_model():
    unrelated_answer = "Bananas are yellow fruit that grow on tall tropical trees in warm climates."
    early = reflect("q", unrelated_answer, CHUNKS, attempt=0, model_used=WEAK_MODEL)
    assert early["decision"] == "retry_search"
    assert early["reason"] == "low_overlap"

    late = reflect("q", unrelated_answer, CHUNKS, attempt=2, model_used=WEAK_MODEL)
    assert late["decision"] == "retry_model"
    assert late["reason"] == "low_overlap_retry_model"


def test_grounded_answer_with_comma_and_percent_formatting_is_accepted():
    # Answer reformats the source numbers ("1000" -> "1,000", "50 %" -> "50%")
    # — this should NOT be treated as a hallucination.
    answer = (
        "The report shows 1,000 users and 50% growth this quarter across "
        "all regions, driven mostly by the mobile platform rollout."
    )
    decision = reflect("growth report", answer, CHUNKS, attempt=0, model_used=WEAK_MODEL)
    assert decision["decision"] == "accept"
    assert decision["valid"] is True


def test_ungrounded_number_retries_model_on_weak_model():
    answer = "The report shows 92345 users and huge growth this quarter across all regions."
    decision = reflect("growth report", answer, CHUNKS, attempt=0, model_used=WEAK_MODEL)
    assert decision["decision"] == "retry_model"
    assert decision["reason"] == "ungrounded_numbers"
    assert decision["failure_type"] == "HALLUCINATION"


def test_ungrounded_number_refuses_when_strong_model_already_used():
    answer = "The report shows 92345 users and huge growth this quarter across all regions."
    decision = reflect("growth report", answer, CHUNKS, attempt=1, model_used=GROQ_STRONG)
    assert decision["decision"] == "refuse"
    assert decision["reason"] == "ungrounded_numbers_strong_model_failed"


def test_should_force_strong_model_only_on_retry_model_with_weak_model():
    retry_decision = {"decision": "retry_model"}
    assert should_force_strong_model(retry_decision, WEAK_MODEL) is True
    assert should_force_strong_model(retry_decision, GROQ_STRONG) is False

    refuse_decision = {"decision": "refuse"}
    assert should_force_strong_model(refuse_decision, WEAK_MODEL) is False
