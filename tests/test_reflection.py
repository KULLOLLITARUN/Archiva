"""Tests for agents/reflection.py — the deterministic answer-quality checks
that drive the self-healing loop's retry/refuse decisions."""

from agents.reflection import _has_contradiction, _numbers_are_grounded, reflect, should_force_strong_model
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


# ── _numbers_are_grounded: verified-sum exception (table aggregation) ──────────

TABLE_CHUNK_TEXT = (
    "Quarter=Q1, Revenue=120000, Costs=80000\n"
    "Quarter=Q2, Revenue=135000, Costs=85000\n"
    "Quarter=Q3, Revenue=150000, Costs=90000\n"
    "Quarter=Q4, Revenue=160000, Costs=95000"
)


def test_correct_column_sum_is_accepted_as_grounded():
    answer = "The total costs across all four quarters were 350000."
    assert _numbers_are_grounded(answer, TABLE_CHUNK_TEXT) is True


def test_correct_two_number_sum_is_accepted_as_grounded():
    answer = "Q1 and Q2 revenue combined is 255000."
    assert _numbers_are_grounded(answer, TABLE_CHUNK_TEXT) is True


def test_incorrect_sum_is_still_rejected():
    # Off-by-one from the real total (350000) - must NOT be waved through
    # just because it's "close" to a real sum. Exact match only.
    answer = "The total costs across all four quarters were 350001."
    assert _numbers_are_grounded(answer, TABLE_CHUNK_TEXT) is False


def test_fabricated_unrelated_number_is_still_rejected():
    # The core anti-hallucination protection must still catch a number
    # that has no relationship to anything in the source at all.
    answer = "The total costs across all four quarters were 92345."
    assert _numbers_are_grounded(answer, TABLE_CHUNK_TEXT) is False


def test_dates_are_not_treated_as_summable_values():
    chunk = "The report was filed on 2024-01-01 and covers Q1 revenue of 120000."
    # "20240101" (if the hyphens were stripped) must not silently become a
    # candidate for subset-sum arithmetic - dates aren't table values.
    answer = "The total was 20240101."
    assert _numbers_are_grounded(answer, chunk) is False


def test_reflect_accepts_a_verified_table_sum_end_to_end():
    chunks = [{"text": TABLE_CHUNK_TEXT}]
    answer = (
        "To find the total costs, add each quarter: "
        "80000 + 85000 + 90000 + 95000 = 350000."
    )
    decision = reflect("What are the total costs?", answer, chunks, attempt=0, model_used=WEAK_MODEL)
    assert decision["decision"] == "accept"
    assert decision["valid"] is True


# ── _has_contradiction ────────────────────────────────────────────────────────
# Regression coverage for a real bug found live: an honest, well-hedged answer
# ("...but does not provide a formal definition...") got flagged as
# contradicting the source, because the old check only asked "does the ANSWER
# contain a negation phrase ANYWHERE" with no proximity requirement to the
# keyword - unlike the chunk-side check, which correctly required the
# negation to be near the keyword. A topic word that's simply the document's
# subject (and so appears "positively" many times in the chunk) would then
# false-positive against an unrelated negation elsewhere in the answer.

AI_ENGINEER_CHUNK = (
    "An AI engineer designs, builds, and deploys machine learning systems "
    "in production. The role of an AI engineer includes data pipeline work, "
    "model training, and MLOps. Many companies now hire an AI engineer to "
    "bridge research and deployment."
)


def test_honest_hedge_with_unrelated_negation_is_not_a_contradiction():
    # "does not" is far from "engineer" and negates something else entirely -
    # this must NOT be flagged, even though "engineer" appears positively
    # many times in the chunk.
    answer = (
        "An AI engineer builds and deploys machine learning systems in "
        "production, working across the data pipeline and MLOps. The "
        "document does not provide a single formal dictionary definition "
        "of the term."
    )
    assert _has_contradiction("what is ai engineer", answer, AI_ENGINEER_CHUNK) is False


def test_reflect_accepts_honest_hedge_end_to_end():
    # Full pipeline reproduction of the live bug: this used to come back
    # "possible_contradiction" / retry_model instead of "accept".
    chunks = [{"text": AI_ENGINEER_CHUNK}]
    answer = (
        "An AI engineer builds and deploys machine learning systems in "
        "production, working across the data pipeline and MLOps. The "
        "document does not provide a single formal dictionary definition "
        "of the term."
    )
    decision = reflect("what is ai engineer", answer, chunks, attempt=0, model_used=WEAK_MODEL)
    assert decision["reason"] != "possible_contradiction"
    assert decision["decision"] == "accept"


def test_genuine_contradiction_is_still_caught():
    # The keyword IS negated close-by in the answer, and the chunk positively
    # asserts the same keyword elsewhere - a real contradiction, must still
    # be flagged.
    chunk = "The warranty covers accidental damage for the first two years of ownership."
    answer = "There is no warranty coverage for accidental damage on this product."
    assert _has_contradiction("warranty accidental damage", answer, chunk) is True


def test_no_negation_in_answer_is_never_a_contradiction():
    chunk = "The warranty covers accidental damage for two years."
    answer = "The warranty covers accidental damage for two years, per the policy."
    assert _has_contradiction("warranty accidental damage", answer, chunk) is False
