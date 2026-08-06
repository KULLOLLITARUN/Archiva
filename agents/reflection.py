"""
agents/reflection.py — Pure-Python answer quality evaluator.

Upgrade (Part 1.2 / structured failure classification):
  reflect() now returns two additional keys:
    "valid"        : bool  — True when decision == "accept"
    "failure_type" : str   — structured enum used by root_cause.py

Failure type mapping:
  RETRIEVAL_FAILURE      ← no results, answer_too_short, low_overlap (mild)
  INSUFFICIENT_CONTEXT   ← low_overlap on retry (need more chunks)
  HALLUCINATION          ← ungrounded_numbers, possible_contradiction
  FORMAT_ERROR           ← answer_too_long
  UNKNOWN                ← anything else

Original design principles retained:
  - Single Responsibility: one function per concern, clear helper split.
  - No magic numbers: all thresholds from config or named constants.
  - Zero LLM calls: deterministic and fast.
  - Full type annotations: no implicit Any.
"""

import re
from typing import Dict, List, Set

from config import GROQ_STRONG, MIN_OVERLAP_RATIO, STOPWORDS

# ── Named constants ────────────────────────────────────────────────────────────

_MIN_ANSWER_WORDS:    int   = 8
_MAX_ANSWER_WORDS:    int   = 500
_CONF_SCALE:          float = 3.0
_CONF_RETRY_PENALTY:  float = 0.2
_CONF_MIN:            float = 0.1

_NEGATION_PHRASES: tuple = (
    "there is no",
    "does not",
    "never",
    "not mentioned",
)

_NUMBER_PATTERN: re.Pattern = re.compile(r"\b\d[\d,\.%/-]*\b")

# ── Failure type constants ────────────────────────────────────────────────────

_FT_RETRIEVAL     = "RETRIEVAL_FAILURE"
_FT_INSUFFICIENT  = "INSUFFICIENT_CONTEXT"
_FT_HALLUCINATION = "HALLUCINATION"
_FT_OUTDATED      = "OUTDATED_DATA"
_FT_FORMAT        = "FORMAT_ERROR"
_FT_UNKNOWN       = "UNKNOWN"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _filter_stopwords(text: str) -> Set[str]:
    """Return lowercase non-stopword tokens from *text*."""
    return {
        word
        for word in re.findall(r"[a-z]+", text.lower())
        if word not in STOPWORDS
    }


def _all_chunk_text(chunks: List[dict]) -> str:
    return " ".join(c.get("text", "") for c in chunks).lower()


def _compute_overlap(answer: str, chunks: List[dict]) -> float:
    answer_words = _filter_stopwords(answer)
    chunk_words  = _filter_stopwords(_all_chunk_text(chunks))
    shared = len(answer_words & chunk_words)
    return shared / max(len(answer_words), 1)


def _extract_numbers(text: str) -> List[str]:
    return _NUMBER_PATTERN.findall(text)


def _numbers_are_grounded(answer: str, chunk_text: str) -> bool:
    numbers = _extract_numbers(answer)
    for number in numbers:
        if number not in chunk_text:
            return False
    return True


def _has_contradiction(query: str, answer: str, chunk_text: str) -> bool:
    answer_lower = answer.lower()
    has_negation = any(phrase in answer_lower for phrase in _NEGATION_PHRASES)
    if not has_negation:
        return False
    query_keywords = _filter_stopwords(query)
    if not query_keywords:
        return False
    return any(kw in chunk_text for kw in query_keywords)


def _build_confidence(overlap_ratio: float, attempt: int) -> float:
    base    = min(1.0, overlap_ratio * _CONF_SCALE)
    penalty = _CONF_RETRY_PENALTY * attempt
    return max(_CONF_MIN, base - penalty)


def _map_failure_type(reason: str, attempt: int) -> str:
    """
    Map a reflection reason string to a structured failure type enum value.
    Used by root_cause.py to select the appropriate healing action.
    """
    if reason in ("no_results_found", "explicit_not_found",
                  "no_chunks_retrieved", "answer_too_short",
                  "answer_too_short_max_attempts"):
        return _FT_RETRIEVAL

    if reason in ("low_overlap",):
        # First encounter → retrieval failure; subsequent → not enough context
        return _FT_RETRIEVAL if attempt == 0 else _FT_INSUFFICIENT

    if reason in ("low_overlap_retry_model",):
        return _FT_INSUFFICIENT

    if reason in ("ungrounded_numbers", "ungrounded_numbers_strong_model_failed",
                  "possible_contradiction"):
        return _FT_HALLUCINATION

    if reason in ("answer_too_long",):
        return _FT_FORMAT

    return _FT_UNKNOWN


# ── Public API ────────────────────────────────────────────────────────────────

def reflect(
    query: str,
    answer: str,
    chunks: List[dict],
    attempt: int,
    model_used: str,
) -> Dict:
    """
    Evaluate answer quality against retrieved chunks.
    Pure Python — no LLM calls. Fast and deterministic.

    Returns:
        {
            "decision":     "accept" | "retry_search" | "retry_model" | "refuse",
            "reason":       str,
            "confidence":   float,
            "valid":        bool,        # NEW — True iff decision == "accept"
            "failure_type": str,         # NEW — structured failure classification
        }
    """
    answer_stripped = answer.strip()

    # ── Check 1: Explicit not-found marker ───────────────────────────────────
    if "not found in the document" in answer_stripped.lower():
        return _decision("refuse", "explicit_not_found", 0.0, attempt)

    # ── Check 2: Answer too short ─────────────────────────────────────────────
    word_count = len(answer_stripped.split())
    if word_count < _MIN_ANSWER_WORDS:
        if attempt < 2:
            return _decision("retry_search", "answer_too_short", 0.0, attempt)
        return _decision("refuse", "answer_too_short_max_attempts", 0.0, attempt)

    # ── Check 3: No chunks retrieved ──────────────────────────────────────────
    if not chunks:
        return _decision("refuse", "no_chunks_retrieved", 0.0, attempt)

    # ── Pre-compute overlap ────────────────────────────────────────────────────
    overlap_ratio = _compute_overlap(answer_stripped, chunks)
    chunk_text    = _all_chunk_text(chunks)

    # ── Check 4: Low word overlap ──────────────────────────────────────────────
    if overlap_ratio < MIN_OVERLAP_RATIO:
        if attempt < 2:
            return _decision("retry_search", "low_overlap", 0.0, attempt)
        return _decision("retry_model", "low_overlap_retry_model", 0.0, attempt)

    # ── Check 5: Hallucinated numbers ─────────────────────────────────────────
    if not _numbers_are_grounded(answer_stripped, chunk_text):
        if model_used != GROQ_STRONG:
            return _decision("retry_model", "ungrounded_numbers", 0.0, attempt)
        return _decision("refuse", "ungrounded_numbers_strong_model_failed", 0.0, attempt)

    # ── Check 6: Answer too long ──────────────────────────────────────────────
    if word_count > _MAX_ANSWER_WORDS and model_used != GROQ_STRONG:
        return _decision("retry_model", "answer_too_long", 0.0, attempt)

    # ── Check 7: Simple contradiction detection ────────────────────────────────
    if _has_contradiction(query, answer_stripped, chunk_text):
        return _decision("retry_model", "possible_contradiction", 0.0, attempt)

    # ── Check 8: All checks passed ────────────────────────────────────────────
    confidence = _build_confidence(overlap_ratio, attempt)
    return _decision("accept", "passed_all_checks", confidence, attempt)


def should_force_strong_model(decision: dict, current_model: str) -> bool:
    """Return True if the reflection result demands GROQ_STRONG on next attempt."""
    return (
        decision["decision"] == "retry_model"
        and current_model != GROQ_STRONG
    )


# ── Private factory ────────────────────────────────────────────────────────────

def _decision(decision: str, reason: str, confidence: float, attempt: int) -> Dict:
    """Construct and return a standardised reflection result dict."""
    return {
        "decision":     decision,
        "reason":       reason,
        "confidence":   confidence,
        "valid":        decision == "accept",          # NEW
        "failure_type": _map_failure_type(reason, attempt),  # NEW
    }
