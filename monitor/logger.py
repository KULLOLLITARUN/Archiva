"""
monitor/logger.py — Pipeline logger with in-memory stats counters.

Upgrade (Part 6 — observability):
  - log_pipeline() now handles: retrieval_latency_ms, reranker_scores,
    failure_type, tokens_used fields in log_data.
  - _update_healing_stats() tracks healing action distribution.
  - log_feedback() integration: after each non-"accept" decision, writes to
    the JSONL feedback store for offline adaptive tuning.
  - All new fields are optional with safe defaults for backward compat.

Fix #17: Added in-memory counters (total_queries, blocked_queries, …).

SonarQube notes:
  - All stat mutations go through _update_stats() (narrow write surface).
  - No global mutable dict/list exported directly — callers use get_stats().
"""

import asyncio
import json
import os
import random
from collections import deque
from datetime import datetime, timezone
from typing import Dict, Optional


# ── In-memory stats counters ──────────────────────────────────────────────────

_stats: Dict = {
    "total_queries":     0,
    "blocked_queries":   0,
    "flagged_responses": 0,
    "model_usage":       {"fast": 0, "strong": 0, "none": 0},
    "intent_counts":     {"qa": 0, "explain": 0, "summarize": 0, "compare": 0},
    # Reflection breakdown
    "reflection_stats": {
        "total_reflected":  0,
        "retry_search":     0,
        "retry_model":      0,
        "refused":          0,
        "best_effort":      0,
        "accepted_attempt": {"1": 0, "2": 0, "3": 0},
        "avg_confidence":   0.0,
        "_confidence_sum":  0.0,
        "_confidence_n":    0,
    },
    # Failure type distribution (Part 6)
    "failure_types": {
        "RETRIEVAL_FAILURE":    0,
        "INSUFFICIENT_CONTEXT": 0,
        "HALLUCINATION":        0,
        "OUTDATED_DATA":        0,
        "FORMAT_ERROR":         0,
        "UNKNOWN":              0,
        "NONE":                 0,
    },
    # Healing action distribution (Part 6)
    "healing_actions": {
        "REWRITE_QUERY":   0,
        "INCREASE_TOP_K":  0,
        "STRICT_PROMPT":   0,
        "REINGEST":        0,
        "NONE":            0,
    },
    # Retrieval performance (Part 6)
    "avg_retrieval_latency_ms": 0.0,
    "total_tokens_used":        0,
}

_latency_window: deque          = deque(maxlen=100)
_retrieval_latency_window: deque = deque(maxlen=100)


# ── Public read ────────────────────────────────────────────────────────────────

def get_stats() -> dict:
    """Return a snapshot of all in-memory counters (for GET /stats)."""
    avg_lat = (
        round(sum(_latency_window) / len(_latency_window), 1)
        if _latency_window else 0.0
    )
    avg_ret = (
        round(sum(_retrieval_latency_window) / len(_retrieval_latency_window), 1)
        if _retrieval_latency_window else 0.0
    )

    ref = _stats["reflection_stats"]
    clean_reflection = {
        "total_reflected":  ref["total_reflected"],
        "retry_search":     ref["retry_search"],
        "retry_model":      ref["retry_model"],
        "refused":          ref["refused"],
        "best_effort":      ref["best_effort"],
        "accepted_attempt": dict(ref["accepted_attempt"]),
        "avg_confidence":   round(ref["avg_confidence"], 4),
    }

    return {
        **_stats,
        "reflection_stats":          clean_reflection,
        "avg_latency_ms":            avg_lat,
        "avg_retrieval_latency_ms":  avg_ret,
        "latency_samples":           len(_latency_window),
    }


# ── Private stat update helpers ───────────────────────────────────────────────

def _update_core_counters(log_data: dict) -> None:
    _stats["total_queries"] += 1
    if log_data.get("safety_decision") in ("regex", "qwen_blocked"):
        _stats["blocked_queries"] += 1
    if log_data.get("flagged"):
        _stats["flagged_responses"] += 1


def _update_model_usage(log_data: dict) -> None:
    from config import GROQ_FAST, GROQ_STRONG
    model = log_data.get("model_used", "none")
    if model == GROQ_FAST:
        _stats["model_usage"]["fast"] += 1
    elif model == GROQ_STRONG:
        _stats["model_usage"]["strong"] += 1
    else:
        _stats["model_usage"]["none"] += 1


def _update_intent_counts(log_data: dict) -> None:
    intent = log_data.get("intent", "qa")
    if intent in _stats["intent_counts"]:
        _stats["intent_counts"][intent] += 1


def _update_latency(log_data: dict) -> None:
    latency = log_data.get("latency_ms")
    if isinstance(latency, (int, float)):
        _latency_window.append(latency)

    ret_latency = log_data.get("retrieval_latency_ms")
    if isinstance(ret_latency, (int, float)):
        _retrieval_latency_window.append(ret_latency)

    tokens = log_data.get("tokens_used")
    if isinstance(tokens, (int, float)):
        _stats["total_tokens_used"] += int(tokens)


# The loop's final reasons for answering "Not found in the document."
_REFUSED_REASONS = frozenset({
    "explicit_not_found", "no_chunks_retrieved", "answer_too_short_max_attempts",
    "ungrounded_numbers_strong_model_failed", "no_results_after_retry",
})


def _update_reflection_stats(log_data: dict) -> None:
    ref     = _stats["reflection_stats"]
    reason  = log_data.get("reflection_reason", "") or ""
    attempts= log_data.get("attempts", 1) or 1

    if log_data.get("reflected"):
        ref["total_reflected"] += 1

    # reflection_reason is the loop's FINAL outcome (agents/loop.py), never an
    # intermediate one like "low_overlap" - matching those left these buckets
    # near zero. Outcome: refused (the answer became "Not found"), or the best
    # of several attempts returned without passing every check.
    if reason in _REFUSED_REASONS:
        ref["refused"] += 1
    elif reason.endswith("_max_attempts_reached"):
        ref["best_effort"] += 1

    # Which kind of retry got there: healing_action is the LAST healing step.
    action = log_data.get("healing_action") or "NONE"
    if attempts > 1 and action in ("REWRITE_QUERY", "INCREASE_TOP_K"):
        ref["retry_search"] += 1
    elif attempts > 1 and action == "STRICT_PROMPT":
        ref["retry_model"] += 1

    attempt_key = str(min(attempts, 3))
    if attempt_key in ref["accepted_attempt"]:
        ref["accepted_attempt"][attempt_key] += 1

    confidence = log_data.get("confidence")
    if isinstance(confidence, (int, float)):
        ref["_confidence_sum"] += float(confidence)
        ref["_confidence_n"]   += 1
        ref["avg_confidence"]   = ref["_confidence_sum"] / ref["_confidence_n"]


def _update_failure_types(log_data: dict) -> None:
    """Track structured failure type distribution (Part 6)."""
    ft = log_data.get("failure_type", "NONE") or "NONE"
    bucket = ft if ft in _stats["failure_types"] else "UNKNOWN"
    _stats["failure_types"][bucket] += 1


def _update_healing_stats(log_data: dict) -> None:
    """Track which healing actions were taken (Part 6)."""
    action = log_data.get("healing_action", "NONE") or "NONE"
    bucket = action if action in _stats["healing_actions"] else "NONE"
    _stats["healing_actions"][bucket] += 1


def _update_stats(log_data: dict) -> None:
    """Update all counters synchronously from log_data."""
    _update_core_counters(log_data)
    _update_model_usage(log_data)
    _update_intent_counts(log_data)
    _update_latency(log_data)
    _update_reflection_stats(log_data)
    _update_failure_types(log_data)
    _update_healing_stats(log_data)


# ── Feedback integration ──────────────────────────────────────────────────────

def _maybe_log_feedback(log_data: dict) -> None:
    """Write a feedback record if this request had a healing cycle."""
    failure_type = log_data.get("failure_type")
    if not failure_type or failure_type in ("NONE", "UNKNOWN"):
        return

    try:
        from monitor.feedback import log_feedback
        log_feedback(
            query        = log_data.get("query", ""),
            failure_type = failure_type,
            fix          = log_data.get("healing_action"),
            success      = not log_data.get("reflected", False)
                           or log_data.get("confidence", 0.0) >= 0.4,
            confidence   = log_data.get("confidence", 0.0),
        )
    except Exception as exc:
        print(f"  [WARN]  [logger] Feedback write failed: {exc}")


# ── Async logger ───────────────────────────────────────────────────────────────

async def log_pipeline(log_data: dict) -> None:
    """
    Async non-blocking pipeline logger.

    Stats counters updated on every request (100% rate).
    Feedback store written when failure_type is set.
    File logging uses 10% sampling to avoid disk pressure.
    """
    _update_stats(log_data)
    _maybe_log_feedback(log_data)

    # File logging: 10% sampling
    if random.random() >= 0.1:
        return

    os.makedirs("logs", exist_ok=True)
    log_data["timestamp"] = datetime.now(timezone.utc).isoformat()

    try:
        with open("logs/pipeline.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_data) + "\n")
    except Exception as e:
        print(f"  [WARN]  Logger failed: {e}")
