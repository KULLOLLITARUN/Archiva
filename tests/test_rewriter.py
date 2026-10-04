"""
Tests for chatbot/rewriter.py — follow-up detection and the anchor used to
resolve a vague follow-up's reference.

Regression coverage for two real bugs found live:
  1. "what main highlight in this" (a genuine new question about the
     previous answer) was rewritten using ONLY the previous question as
     the anchor, which dominated retrieval and generation and caused the
     LLM to just re-answer the previous question instead of the new one -
     the model never saw what "this" referred to.
  2. is_followup()'s old "< 6 words -> followup" fallback caught short but
     complete, unrelated new questions ("what need to be in resume", "give
     summary") and anchored them to whatever the previous turn happened to
     be about, dragging retrieval off-topic and producing false "not found"
     answers - see test_short_self_contained_new_topic_query_is_not_a_followup.
"""

from chatbot.memory import ConversationMemory
from chatbot.rewriter import is_followup, rewrite


# ── is_followup (pure, no LLM) ────────────────────────────────────────────────

def test_query_with_reference_word_is_a_followup():
    assert is_followup("what main highlight in this") is True  # contains "this"


def test_query_with_followup_signal_word_is_a_followup():
    assert is_followup("tell me more about the pricing details please") is True


def test_long_self_contained_query_is_not_a_followup():
    query = "What is the total revenue reported for the first quarter of 2026?"
    assert is_followup(query) is False


def test_short_self_contained_new_topic_query_is_not_a_followup():
    # Regression: is_followup() used to fall back to "< 6 words -> followup",
    # which caught short but complete, unrelated new questions and dragged
    # them into anchoring against whatever the previous turn happened to be
    # about. Found live: "what need to be in resume" (5 words, no reference
    # word) got anchored to an unrelated prior Q&A about omitting age/DOB,
    # and retrieval got dragged off-topic -> a false "not found" answer for
    # a question the document actually did answer.
    assert is_followup("what need to be in resume") is False


def test_bare_short_command_without_signal_word_is_not_a_followup():
    # "give summary" (2 words) has no reference/signal word either - same
    # bug, compounded: it anchored to the PREVIOUS turn's failed "not found"
    # answer, cascading the failure forward.
    assert is_followup("give summary") is False


# ── rewrite ────────────────────────────────────────────────────────────────────

def _memory_with_last_turn(query: str, answer: str) -> ConversationMemory:
    memory = ConversationMemory.__new__(ConversationMemory)  # skip disk load
    memory.sessions = {}
    memory.sessions["s1"] = [
        type("Entry", (), {"query": query, "answer": answer})()
    ]
    return memory


def test_rewrite_leaves_non_followup_queries_unchanged():
    memory = _memory_with_last_turn("prior question", "prior answer")
    query = "What is the total revenue reported for the first quarter of 2026?"
    assert rewrite(query, memory, "s1") == query


def test_rewrite_is_a_noop_with_no_prior_turn():
    memory = ConversationMemory.__new__(ConversationMemory)
    memory.sessions = {}
    query = "what about this"
    assert rewrite(query, memory, "s1") == query


def test_rewrite_anchors_to_both_prior_question_and_prior_answer():
    memory = _memory_with_last_turn(
        "What information should be included in the education section for freshers?",
        "For freshers, the education section should include: degree, institution, "
        "graduation date, relevant projects, and GPA.",
    )
    result = rewrite("what main highlight in this", memory, "s1")

    assert "what main highlight in this" in result
    assert "What information should be included in the education section for freshers?" in result
    assert "degree, institution, graduation date" in result


def test_rewrite_truncates_a_long_prior_answer():
    long_answer = "x" * 1000
    memory = _memory_with_last_turn("prior question", long_answer)
    result = rewrite("what about it", memory, "s1")
    # The full 1000-char answer must not appear verbatim - only a preview.
    assert long_answer not in result
    assert "x" * 300 in result


def test_rewrite_does_not_anchor_a_short_new_topic_query():
    # Regression for the live "what need to be in resume" bug: an unrelated
    # prior turn about DOB/age omission must not get dragged into a short
    # but self-contained new question just because it's short.
    memory = _memory_with_last_turn(
        "should age and date of birth be on a resume",
        "No, age and date of birth should be omitted from a resume.",
    )
    query = "what need to be in resume"
    assert rewrite(query, memory, "s1") == query


def test_rewrite_does_not_anchor_to_a_prior_refusal():
    # Regression: once one turn fails ("not found"), a later short command
    # like "give summary" must not anchor to that failure and cascade it.
    memory = _memory_with_last_turn("what need to be in resume", "Not found in the document.")
    query = "give summary"
    assert rewrite(query, memory, "s1") == query


# ── Whole-word signals and openers ─────────────────────────────────────────────

def test_signal_inside_another_word_is_not_a_followup():
    # "it" used to match inside "with", "edit", "limit", "item", "wait"...
    for query in ("Which clauses deal with termination?", "How do I edit the config file?",
                  "What is the spending limit per item?", "What is the audit schedule?"):
        assert is_followup(query) is False, query


def test_signal_as_a_whole_word_is_still_a_followup():
    assert is_followup("What does it cost?") is True
    assert is_followup("Explain that in more detail") is True


def test_query_opening_with_and_is_a_followup():
    # Found live: "And what is the IFSC code?" was not anchored to the
    # previous answer about the invoice's bank, so retrieval returned a
    # 856-passage book instead of the invoice and the question was refused.
    assert is_followup("And what is the IFSC code?") is True
    assert is_followup("Also, who signed it off?") is True
    assert is_followup("but what about the second one") is True


def test_and_in_the_middle_of_a_query_is_not_an_opener():
    assert is_followup("What are the payment terms and the delivery dates?") is False


def test_followup_is_anchored_to_the_previous_turn():
    memory = _memory_with_last_turn("Which bank and UPI ID are on the invoice?", "ICICI and ifox@icici.")
    anchored = rewrite("And what is the IFSC code?", memory, "s1")
    assert "Which bank and UPI ID are on the invoice?" in anchored
    assert "ICICI" in anchored
