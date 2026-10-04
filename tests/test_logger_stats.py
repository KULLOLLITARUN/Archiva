"""Tests for monitor/logger.py's in-memory reflection counters (shown on
Stats & maintenance). They count the loop's FINAL reflection_reason, so
each case below uses a reason agents/loop.py actually returns."""

import copy

import pytest

from monitor import logger


@pytest.fixture(autouse=True)
def _fresh_stats(monkeypatch):
    # The counters are module-global; give each test its own copy.
    monkeypatch.setattr(logger, "_stats", copy.deepcopy(logger._stats))
    ref = logger._stats["reflection_stats"]
    for key in ("total_reflected", "retry_search", "retry_model", "refused", "best_effort"):
        ref[key] = 0


def _log(**fields):
    logger._update_reflection_stats({"attempts": 1, "healing_action": "NONE", **fields})
    return logger._stats["reflection_stats"]


@pytest.mark.parametrize("reason", [
    "explicit_not_found", "no_chunks_retrieved", "answer_too_short_max_attempts",
    "ungrounded_numbers_strong_model_failed", "no_results_after_retry",
])
def test_every_refusal_reason_counts_as_refused(reason):
    assert _log(reflection_reason=reason)["refused"] == 1


def test_best_effort_after_max_attempts_is_not_a_refusal():
    ref = _log(reflection_reason="low_overlap_max_attempts_reached", attempts=3, healing_action="INCREASE_TOP_K")
    assert (ref["best_effort"], ref["refused"]) == (1, 0)


def test_retries_are_attributed_by_the_last_healing_action():
    _log(reflection_reason="passed_all_checks", attempts=2, healing_action="REWRITE_QUERY")
    ref = _log(reflection_reason="passed_all_checks", attempts=2, healing_action="STRICT_PROMPT")
    assert (ref["retry_search"], ref["retry_model"]) == (1, 1)


def test_a_first_attempt_pass_counts_as_no_retry():
    ref = _log(reflection_reason="passed_all_checks")
    assert (ref["retry_search"], ref["retry_model"], ref["refused"], ref["best_effort"]) == (0, 0, 0, 0)


def test_get_stats_exposes_best_effort():
    _log(reflection_reason="possible_contradiction_max_attempts_reached", attempts=3)
    assert logger.get_stats()["reflection_stats"]["best_effort"] == 1
