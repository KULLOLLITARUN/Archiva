"""
agents/validator.py — Post-response hallucination detector.

Fix #14: Raised overlap threshold from 3 to 8 meaningful words.
         Three-word overlap was too weak — hallucinated answers could
         easily share 3 common words with source material.

Fix #14: Added ungrounded_numbers check: if the answer contains
         numbers or dates not present in any chunk text, flag it.
         Making up specific figures is the most common hallucination pattern.
"""

import re
from typing import Dict, List

from agents.reflection import strip_structural_numbers
from config import STOPWORDS

# Matches standalone numbers, percentages, dates, and version numbers
# e.g. "42", "3.14", "99%", "2024-04-10", "v1.2.3", "10ms"
_NUMBER_RE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|\d+(?:\.\d+)?(?:%|ms|s|gb|mb|kb)?)\b",
    re.IGNORECASE,
)


def _extract_numbers(text: str) -> set:
    """Return the set of all number-like tokens found in *text*."""
    return set(_NUMBER_RE.findall(text.lower()))


def validate(answer: str, chunks: List[dict]) -> Dict:
    """
    Post-response validator — NO LLM.
    Uses stopword-filtered word overlap to detect hallucination risk.

    Returns:
        {"valid": True, "flagged": bool, "reason": str}
    """
    # Check 1: explicit not-found response is always valid
    if "not found" in answer.lower():
        return {"valid": True, "flagged": False, "reason": "not_found_response"}

    all_chunk_text = " ".join(c["text"] for c in chunks)

    chunk_words = {
        w.lower() for w in all_chunk_text.split()
        if w.lower() not in STOPWORDS
    }
    answer_words = {
        w.lower() for w in answer.split()
        if w.lower() not in STOPWORDS
    }

    overlap = len(answer_words & chunk_words)
    flagged = False
    reason  = "ok"

    # Check 2: answer too long
    if len(answer.split()) > 500:
        flagged = True
        reason  = "answer_too_long"

    # Check 3: insufficient overlap with source material
    # Fix #14: raised threshold from 3 to 8 meaningful words
    if overlap < 8:
        flagged = True
        reason  = reason + "|low_overlap" if reason != "ok" else "low_overlap"

    # Check 4 (Fix #14): ungrounded numbers/dates
    # If the answer contains any numeric value not present in any chunk,
    # it is likely hallucinated — models commonly fabricate specific figures.
    # List ordinals and step/section labels aren't facts (see reflection.py).
    answer_numbers = _extract_numbers(strip_structural_numbers(answer))
    chunk_numbers  = _extract_numbers(all_chunk_text)
    ungrounded     = answer_numbers - chunk_numbers
    if ungrounded:
        flagged = True
        reason  = reason + "|ungrounded_numbers" if reason != "ok" else "ungrounded_numbers"

    return {"valid": True, "flagged": flagged, "reason": reason}
