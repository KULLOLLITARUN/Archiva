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

from agents.reflection import _filter_stopwords, strip_structural_numbers

# Matches standalone numbers, percentages, dates, and version numbers
# e.g. "42", "3.14", "99%", "2024-04-10", "v1.2.3", "10ms"
_NUMBER_RE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|\d+(?:\.\d+)?(?:%|ms|s|gb|mb|kb)?)\b",
    re.IGNORECASE,
)


# Inline citations, "[Source: file.pdf, page 1]" (or 【...】), name the file
# and page an answer came from. They're provenance, not claims: a file
# name like "Invoice_page-0001.pdf" was being read as the ungrounded
# number "0001", flagging correct answers. Same pattern as main.py's.
_CITATION_RE = re.compile(r"[\[【]\s*Source:[^\]】]*[\]】]", re.IGNORECASE)

# Meaningful words an answer must share with its sources. Short answers
# can't reach 8 distinct words however well grounded they are ("The bank
# is ICICI and the UPI ID is ifox@icici."), so they need half of theirs.
_MIN_OVERLAP_WORDS = 8


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

    # Check against what the model was shown: the loop prompts with each
    # chunk's parent section (_context_text) when it has one, so a figure
    # copied correctly from that wider section ("3-8 minutes") isn't in the
    # child chunk's own text and was flagged as made up.
    all_chunk_text = " ".join(c.get("_context_text") or c.get("text", "") for c in chunks)
    claims = _CITATION_RE.sub(" ", answer)

    # Letters-only words, as reflection counts them: splitting on whitespace
    # kept markdown and punctuation attached ("**ICICI**", "bank,"), so
    # words the source plainly contains didn't match.
    chunk_words  = _filter_stopwords(all_chunk_text)
    answer_words = _filter_stopwords(claims)

    overlap = len(answer_words & chunk_words)
    min_overlap = min(_MIN_OVERLAP_WORDS, max(1, len(answer_words) // 2))
    flagged = False
    reason  = "ok"

    # Check 2: answer too long
    if len(claims.split()) > 500:
        flagged = True
        reason  = "answer_too_long"

    # Check 3: insufficient overlap with source material
    # Fix #14: raised threshold from 3 to 8 meaningful words
    if overlap < min_overlap:
        flagged = True
        reason  = reason + "|low_overlap" if reason != "ok" else "low_overlap"

    # Check 4 (Fix #14): ungrounded numbers/dates
    # If the answer contains any numeric value not present in any chunk,
    # it is likely hallucinated — models commonly fabricate specific figures.
    # List ordinals and step/section labels aren't facts (see reflection.py).
    answer_numbers = _extract_numbers(strip_structural_numbers(claims))
    chunk_numbers  = _extract_numbers(all_chunk_text)
    ungrounded     = answer_numbers - chunk_numbers
    if ungrounded:
        flagged = True
        reason  = reason + "|ungrounded_numbers" if reason != "ok" else "ungrounded_numbers"

    return {"valid": True, "flagged": flagged, "reason": reason}
