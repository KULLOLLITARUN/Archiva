"""
eval/answer_grading.py - Deterministic grading for the answer-quality eval.

Pure functions, no LLM and no store, so the grading rules themselves are unit
tested (tests/test_answer_grading.py) and a case can't pass or fail because
of a grader bug.

A case looks like:
    {
      "id": "chain-disk-region",
      "category": "chained",
      "question": "...",
      "expect": "answer" | "not_found" | "partial",
      "must_include": [["disk-build-1"], ["west europe"], ["zone 2", "re:zone\\s*2"]],
      "must_not_include": ["build-clone-7"],
      "allow_contradiction_flag": false
    }

* must_include is a list of GROUPS. Every group must be satisfied, and a group
  is satisfied if ANY of its alternatives appears. An alternative prefixed
  "re:" is a regular expression; anything else is a plain substring.
* Matching is done on normalised text: lower case, markdown emphasis removed,
  typographic hyphens/spaces/quotes turned into ASCII, whitespace collapsed -
  so "**Disk‑Build‑1**" matches "disk-build-1".
* expect:
    answer    - must NOT be a not-found reply
    not_found - must be a not-found reply (and must_include is not checked)
    partial   - a chained question where a prerequisite is missing: some step
                says not-found AND a later step is skipped with a reason
* A reply flagged "possible_contradiction" fails unless the case sets
  allow_contradiction_flag - every false contradiction flag we have seen was
  on a correct answer, so it is treated as a defect.
"""

import re
from typing import Dict, List, Tuple

_NOT_FOUND = "not found in the document"
_SKIPPED = "could not answer this step"
_UNAVAILABLE = "service temporarily unavailable"

_DASHES = re.compile("[‐‑‒–—―−]")
_SPACES = re.compile("[      　]")
_EMPHASIS = re.compile(r"[*_`]+")

VALID_EXPECT = ("answer", "not_found", "partial")


def normalize(text: str) -> str:
    text = _DASHES.sub("-", text or "")
    text = _SPACES.sub(" ", text)
    text = (text.replace("‘", "'").replace("’", "'")
                .replace("“", '"').replace("”", '"'))
    text = _EMPHASIS.sub("", text)
    return " ".join(text.lower().split())


def _matches(alternative: str, text: str) -> bool:
    if alternative.startswith("re:"):
        return re.search(alternative[3:], text) is not None
    return normalize(alternative) in text


def is_not_found(answer: str) -> bool:
    return _NOT_FOUND in normalize(answer)


def is_unavailable(answer: str) -> bool:
    """The LLM provider failed (rate limit / outage) - an infrastructure error, not a wrong answer."""
    return _UNAVAILABLE in normalize(answer)


def validate_case(case: Dict) -> List[str]:
    """Return a list of problems with a case definition (empty if valid)."""
    problems = []
    for key in ("id", "category", "question", "expect"):
        if not case.get(key):
            problems.append(f"missing '{key}'")
    if case.get("expect") not in VALID_EXPECT:
        problems.append(f"expect must be one of {VALID_EXPECT}")
    for group in case.get("must_include", []):
        if not isinstance(group, list) or not group:
            problems.append("must_include must be a list of non-empty lists")
            break
    if case.get("expect") == "answer" and not case.get("must_include"):
        problems.append("an 'answer' case needs must_include, or it can never fail")
    return problems


def grade(case: Dict, answer: str, reflection_reason: str = "") -> Tuple[bool, List[str]]:
    """
    Grade one answer. Returns (passed, reasons) where reasons explains every
    failed check (empty when passed).
    """
    text = normalize(answer)
    expect = case.get("expect", "answer")

    reasons = _check_expectation(expect, text)
    if expect != "not_found":
        reasons += [f"missing any of {group}" for group in case.get("must_include", [])
                    if not any(_matches(alt, text) for alt in group)]
    reasons += [f"contains forbidden {bad!r}" for bad in case.get("must_not_include", [])
                if _matches(bad, text)]
    if "possible_contradiction" in (reflection_reason or "") and not case.get("allow_contradiction_flag"):
        reasons.append(f"flagged as a contradiction ({reflection_reason})")

    return (not reasons), reasons


def _check_expectation(expect: str, text: str) -> List[str]:
    """Is the reply the right KIND of reply (an answer / not found / partial)?"""
    not_found = _NOT_FOUND in text
    if expect == "not_found":
        return [] if not_found else ["expected a not-found reply, got an answer (possible hallucination)"]
    if expect == "partial":
        reasons = [] if not_found else ["expected the missing step to say not found"]
        if _SKIPPED not in text:
            reasons.append("expected the dependent step to be skipped with a reason")
        return reasons
    # For a chained question this also catches ONE step saying not-found.
    return ["answered 'not found' although the documents contain the answer"] if not_found else []
