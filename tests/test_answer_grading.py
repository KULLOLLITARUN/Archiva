"""Tests for eval/answer_grading.py - the rules that decide whether an eval
answer passes. A grading bug would make the whole eval lie, so the rules
are pinned here."""

import json
from pathlib import Path

from eval.answer_grading import grade, is_not_found, is_unavailable, normalize, validate_case

CASES_FILE = Path(__file__).resolve().parent.parent / "eval" / "answer_cases.json"


def case(**overrides):
    base = {"id": "c", "category": "single", "question": "q?", "expect": "answer",
            "must_include": [["disk-build-1"]]}
    base.update(overrides)
    return base


# ── normalize ───────────────────────────────────────────────────────────────────

def test_normalize_handles_markdown_and_lookalike_characters():
    assert normalize("**Disk‑Build‑1** in  West Europe") == "disk-build-1 in west europe"


# ── answer cases ────────────────────────────────────────────────────────────────

def test_answer_with_every_required_fact_passes():
    ok, reasons = grade(case(), "The VM uses **Disk‑Build‑1**.")
    assert ok and reasons == []


def test_missing_fact_fails_with_a_reason():
    ok, reasons = grade(case(must_include=[["disk-build-1"], ["west europe"]]), "It uses Disk-Build-1.")
    assert not ok and any("west europe" in r for r in reasons)


def test_any_alternative_in_a_group_is_enough():
    c = case(must_include=[["6 weeks", "six weeks"]])
    assert grade(c, "Six weeks of paid leave.")[0]


def test_regex_alternatives():
    c = case(must_include=[["re:zone\\s*2"]])
    assert grade(c, "Availability Zone 2.")[0]
    assert not grade(c, "Availability Zone 12.")[0]    # a real regex, not a substring


def test_not_found_reply_fails_an_answer_case():
    ok, reasons = grade(case(), "Not found in the document.")
    assert not ok and any("not found" in r for r in reasons)


def test_one_chained_step_saying_not_found_fails_an_answer_case():
    answer = "**Step one?**\nDisk-Build-1.\n\n**Step two?**\nNot found in the document."
    assert not grade(case(), answer)[0]


def test_forbidden_text_fails():
    ok, reasons = grade(case(must_not_include=["build-clone-7"]), "Disk-Build-1 for Build-Clone-7.")
    assert not ok and any("forbidden" in r for r in reasons)


def test_contradiction_flag_fails_unless_allowed():
    ok, reasons = grade(case(), "Disk-Build-1.", reflection_reason="possible_contradiction_max_attempts_reached")
    assert not ok and any("contradiction" in r for r in reasons)
    assert grade(case(allow_contradiction_flag=True), "Disk-Build-1.",
                 reflection_reason="possible_contradiction")[0]


# ── not_found cases ─────────────────────────────────────────────────────────────

def test_not_found_case_passes_on_not_found():
    assert grade(case(expect="not_found", must_include=[]), "Not found in the document.")[0]


def test_not_found_case_fails_on_an_invented_answer():
    ok, reasons = grade(case(expect="not_found", must_include=[]), "The CEO is Jane Doe.")
    assert not ok and any("hallucination" in r for r in reasons)


# ── partial (chained with a missing prerequisite) ───────────────────────────────

PARTIAL_OK = ("**Which disk creates Build-Clone-7?** Not found in the document.\n\n"
              "**What zone must the answer to step 1 be in?** Could not answer this step because it "
              "depends on step 1, which was not found in the documents.")


def test_partial_case_passes_when_step_missing_and_dependent_skipped():
    assert grade(case(expect="partial", must_include=[]), PARTIAL_OK)[0]


def test_partial_case_fails_if_the_dependent_step_was_guessed():
    guessed = "**Which disk?** Not found in the document.\n\n**What zone?** Zone 2."
    ok, reasons = grade(case(expect="partial", must_include=[]), guessed)
    assert not ok and any("skipped" in r for r in reasons)


# ── provider errors ─────────────────────────────────────────────────────────────

def test_provider_unavailable_is_recognised():
    assert is_unavailable("Service temporarily unavailable. Please try again.")
    assert not is_unavailable("Disk-Build-1.")
    assert is_not_found("**Not found in the document.**")


# ── the real case file ──────────────────────────────────────────────────────────

def test_every_case_in_the_case_file_is_valid_and_ids_are_unique():
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    problems = {c.get("id"): validate_case(c) for c in cases if validate_case(c)}
    assert problems == {}
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids))


def test_case_file_covers_every_failure_category():
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    categories = {c["category"] for c in cases}
    assert {"single", "chained", "chained-missing", "unanswerable", "negation", "numeric"} <= categories


def test_validate_case_rejects_an_answer_case_that_can_never_fail():
    assert validate_case(case(must_include=[]))
    assert validate_case(case(expect="maybe"))
