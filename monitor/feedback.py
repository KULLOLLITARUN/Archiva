"""
monitor/feedback.py — Feedback store for adaptive tuning.

Part 6 (Feedback Store):
Appends structured feedback records to a JSONL file.  Records are used offline
to analyse failure patterns and tune retrieval / prompting strategies.

Schema per record:
    {
        "query":        str,
        "failure_type": str,   # RETRIEVAL_FAILURE | HALLUCINATION | …
        "fix":          str,   # REWRITE_QUERY | STRICT_PROMPT | …
        "success":      bool,  # did the fix result in a valid answer?
        "ts":           str,   # ISO-8601 UTC timestamp
        "confidence":   float
    }
"""

import json
import os
from datetime import datetime, timezone
from typing import Optional


def _feedback_path() -> str:
    from config import FEEDBACK_LOG_PATH
    return FEEDBACK_LOG_PATH


def log_feedback(
    query: str,
    failure_type: Optional[str],
    fix: Optional[str],
    success: bool,
    confidence: float = 0.0,
) -> None:
    """
    Append a feedback record to the JSONL feedback store.

    Args:
        query:        The original user query.
        failure_type: Structured failure type (from reflect()) or None.
        fix:          The healing action applied (from healer.py) or None.
        success:      True if the healed attempt produced a valid answer.
        confidence:   Final confidence score (0.0–1.0).
    """
    path = _feedback_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    record = {
        "query":        query,
        "failure_type": failure_type or "NONE",
        "fix":          fix or "NONE",
        "success":      success,
        "confidence":   round(confidence, 4),
        "ts":           datetime.now(timezone.utc).isoformat(),
    }

    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:
        print(f"  [WARN]  [feedback] Write failed: {exc}")
