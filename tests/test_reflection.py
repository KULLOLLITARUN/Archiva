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


def test_short_answer_with_a_fact_from_the_chunks_is_accepted():
    # Under _MIN_ANSWER_WORDS, but "1000" and "users" come from the chunks, not the question.
    decision = reflect("How many people use the system?", "1000 users.", CHUNKS, attempt=0, model_used=WEAK_MODEL)
    assert decision["decision"] == "accept"


def test_short_answer_that_only_echoes_the_question_still_retries():
    decision = reflect("What drove the growth?", "The growth.", CHUNKS, attempt=0, model_used=WEAK_MODEL)
    assert decision["reason"] == "answer_too_short"


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


# Regression: found live on an Azure VM cloning document. The query asked why
# one method is "safer"; the (correct) answer said the "specialized VM approach
# is safer because it never modifies ...". "never" negates "modifies", but the
# old check saw it within 50 characters of the query keyword "specialized" and,
# since the source asserts "specialized VM" positively, flagged a correct answer
# as a contradiction (retry, then a "possible contradiction / 0% confident" badge).

CLONING_CHUNK = (
    "Method: snapshot -> managed disk -> specialized vm (safe, no sysprep). "
    "The previous approach ran sysprep and generalized the source vm, which wiped "
    "the user profile. Generalized capture is used only for clean templates."
)


def test_negation_that_belongs_to_another_verb_is_not_a_contradiction():
    answer = (
        "The Snapshot → Managed Disk → Specialized VM approach is safer "
        "because it never modifies or touches the original VM."
    )
    query = "Why is the Snapshot Managed Disk Specialized VM method safer than Generalized Capture?"
    assert _has_contradiction(query, answer, CLONING_CHUNK) is False


def test_answer_negating_what_the_source_also_negates_is_not_a_contradiction():
    # The source says "no sysprep" for the new method (and "ran sysprep" for the
    # old one). An answer saying the new method does not run sysprep agrees with it.
    answer = "The specialized approach does not run sysprep, so the user profile is preserved."
    assert _has_contradiction("does the specialized method use sysprep", answer, CLONING_CHUNK) is False


def test_old_vs_new_comparison_with_many_negations_is_not_flagged():
    answer = (
        "It does not run Sysprep, so the user profile is preserved. "
        "The Start/Stop buttons remain active. It avoids the Compute Gallery. "
        "Changes in one clone do not affect the others."
    )
    query = "Why is the specialized VM method better than generalized capture for cloning VMs?"
    assert _has_contradiction(query, answer, CLONING_CHUNK) is False


def test_negation_directly_before_keyword_is_still_a_contradiction():
    chunk = "The service offers weekend support for all premium customers."
    answer = "The service does not offer weekend support."
    assert _has_contradiction("weekend support", answer, chunk) is True


def test_there_is_no_phrase_a_few_words_before_keyword_is_still_caught():
    chunk = "The warranty covers accidental damage for the first two years."
    answer = "There is no coverage for accidental damage."
    assert _has_contradiction("accidental damage", answer, chunk) is True


def test_not_mentioned_after_keyword_is_still_caught():
    chunk = "Refunds are available within thirty days of purchase."
    answer = "Refunds are not mentioned anywhere in the policy."
    assert _has_contradiction("refunds", answer, chunk) is True


def test_negation_does_not_cross_a_clause_boundary():
    chunk = "The warranty covers accidental damage for two years."
    answer = "The plan never expires early; warranty terms apply to accidental damage."
    assert _has_contradiction("warranty accidental damage", answer, chunk) is False


def test_reflect_accepts_the_cloning_answer_end_to_end():
    answer = (
        "The Snapshot → Managed Disk → Specialized VM approach is safer because it "
        "never modifies the original VM and does not run sysprep, so the user profile "
        "is preserved, unlike generalized capture which wiped it."
    )
    query = "Why is the Snapshot Managed Disk Specialized VM method considered safer than Generalized Capture?"
    decision = reflect(query, answer, [{"text": CLONING_CHUNK}], attempt=0, model_used=WEAK_MODEL)
    assert decision["reason"] != "possible_contradiction"


# Regression: "What happens if Central India and Availability Zone 1 don't match
# the original VM?" was correctly answered "...if the region or zone does not
# exactly match the original VM, the VM creation step will fail" - and flagged
# as a contradiction, because the answer negates "match" while the source says
# "Must Match Exactly". The question itself negates "match", so the answer is
# only restating the condition it was asked about.

MATCH_CHUNK = (
    "Region and Zone Must Match Exactly. The managed disk must be created in "
    "Central India, Zone 1, exactly matching Win11-Recovered. If the region or zone "
    "is different, the VM creation in Step 1.3 will fail."
)


def test_answer_echoing_a_negation_from_the_question_is_not_a_contradiction():
    query = "What happens if Central India and Availability Zone 1 don't match the original VM?"
    answer = (
        "If the region or the availability zone does not exactly match the original "
        "VM, the VM creation step will fail."
    )
    assert _has_contradiction(query, answer, MATCH_CHUNK.lower()) is False


def test_same_answer_to_a_positive_question_is_still_judged_on_its_merits():
    # Asked positively, an answer that denies the match against a source that
    # asserts it IS worth flagging - the echo exemption must not swallow this.
    query = "Does the disk region match the original VM?"
    answer = "The disk region does not match the original VM."
    assert _has_contradiction(query, answer, MATCH_CHUNK.lower()) is True


# Regressions found by the answer-quality eval (eval/run_answer_eval.py):
# keywords were matched as raw substrings with exact spelling.

RUNBOOK_CHUNK = (
    "vm cloning runbook - build server clones. method: snapshot -> managed disk -> "
    "specialized vm (safe, no sysprep). the previous approach, generalized capture, "
    "ran sysprep on the source vm."
)


def test_keyword_inside_a_longer_word_is_not_a_source_assertion():
    # "run" (from the question) matched inside "runbook", so the source looked
    # like it asserted "run" and a correct "does not run Sysprep" was flagged.
    answer = "No. The Specialized VM method does not run Sysprep."
    assert _has_contradiction("Does the Specialized VM method run Sysprep?", answer, RUNBOOK_CHUNK) is False


def test_inflected_form_in_the_source_counts_as_the_same_word():
    # Answer: "not deallocated"; source: "does not deallocate" - the source
    # negates the same word, so the answer agrees with it.
    chunk = ("a vm that is stopped (deallocated) in the portal does not incur compute charges. "
             "shutting down from inside the operating system does not deallocate the vm.")
    answer = "If you only shut it down from inside the OS, it is not deallocated and charges continue."
    query = "Does a VM that is stopped and deallocated still incur compute charges?"
    assert _has_contradiction(query, answer, chunk) is False


def test_whole_word_matching_still_catches_a_real_contradiction():
    chunk = "the backup job runs every night at 2am."
    answer = "The backup job does not run at night."
    assert _has_contradiction("when does the backup job run", answer, chunk) is True
