"""Tests for chatbot/intent.py — deterministic intent detection (no LLM)."""

from chatbot.intent import detect_intent


def test_meta_intent_for_document_inventory_questions():
    assert detect_intent("how many documents are loaded") == "meta"
    assert detect_intent("what files do we have") == "meta"


def test_compare_intent():
    assert detect_intent("compare the two contracts") == "compare"
    assert detect_intent("what's the difference between plan A and plan B") == "compare"


def test_summarize_intent():
    assert detect_intent("give me a summary of this document") == "summarize"


def test_explain_intent():
    assert detect_intent("explain the termination clause") == "explain"


def test_default_qa_intent():
    assert detect_intent("what is the notice period") == "qa"


def test_meta_takes_priority_over_other_keywords():
    # contains both a meta phrase and "compare" — meta is checked first
    assert detect_intent("how many docs compare pricing") == "meta"
