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

import math
import re
from typing import Dict, List, Optional, Set

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
_FT_OUTDATED      = "OUTDATED_DATA"  # kept for the root_cause.py mapping table;
                                      # _map_failure_type() below never returns
                                      # it — staleness isn't derivable from a
                                      # query/answer/chunks alone the way the
                                      # other failure types are, so REINGEST is
                                      # operator-triggered only, not reflected
                                      # into automatically. See main.py's
                                      # GET /admin/reingestion-queue docstring.
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


def _normalize_number_text(text: str) -> str:
    """
    Collapse formatting differences that don't change a number's meaning:
    thousands-separator commas ("1,000" -> "1000") and whitespace between a
    digit and a trailing "%" ("50 %" -> "50%"). Applied to both the answer
    and the chunk text before comparison so cosmetic reformatting by the
    LLM isn't mistaken for a hallucinated number.
    """
    text = re.sub(r"(\d)\s+%", r"\1%", text)
    return text.replace(",", "")


_MAX_SUM_VERIFY_POOL: int = 12   # cap subset-sum search regardless of chunk size


def _plain_numeric_value(number_str: str) -> Optional[float]:
    """
    Parse *number_str* as a plain numeric value for arithmetic
    verification, or None if it's not a clean number — a date
    ("2024-01-01"), a fraction ("3/4"), etc. aren't meaningful as table
    values to sum, and treating them as such would risk false-clearing a
    genuinely fabricated number that happens to coincide with some sum.
    """
    s = number_str.rstrip("%")
    if not re.fullmatch(r"\d+(\.\d+)?", s):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _achievable_sums(values: List[float], max_pool: int = _MAX_SUM_VERIFY_POOL) -> Set[float]:
    """
    Every sum achievable by adding together any subset (2+ terms — a
    single grounded value is already covered by the direct verbatim
    check in _numbers_are_grounded) of *values*. Capped to the first
    `max_pool` values so this stays bounded regardless of how many
    numbers a chunk contains (2^12 = 4096 combinations, trivial).
    """
    pool = values[:max_pool]
    reachable: Set[float] = {0.0}
    for v in pool:
        reachable |= {s + v for s in reachable}
    return reachable


def _isclose_to_any(value: float, candidates: Set[float]) -> bool:
    return any(math.isclose(value, c, rel_tol=1e-9, abs_tol=1e-6) for c in candidates)


def _numbers_are_grounded(answer: str, chunk_text: str) -> bool:
    """
    Check that every number in *answer* is grounded: either it appears
    verbatim in *chunk_text*, or it's exactly explainable as the sum of a
    small set of numbers that DO appear in *chunk_text* — e.g. an LLM
    correctly summing a table column ("total costs" = 80000 + 85000 +
    90000 + 95000 = 350000) shouldn't be flagged as a hallucination just
    because "350000" itself isn't written anywhere in the source.

    This is a narrow, bounded exception, not a general "trust computed
    numbers" rule: a number that doesn't match any subset sum of the
    grounded values is still rejected, same as before.
    """
    normalized_answer = _normalize_number_text(answer)
    normalized_chunk  = _normalize_number_text(chunk_text)

    ungrounded = [n for n in _extract_numbers(normalized_answer) if n not in normalized_chunk]
    if not ungrounded:
        return True

    grounded_values = [
        v for v in (
            _plain_numeric_value(n) for n in _extract_numbers(normalized_chunk)
        ) if v is not None
    ]
    if not grounded_values:
        return False

    achievable = _achievable_sums(grounded_values)
    for number_str in ungrounded:
        value = _plain_numeric_value(number_str)
        if value is None or not _isclose_to_any(value, achievable):
            return False
    return True


_NEGATION_WINDOW: int = 50   # chars — how close a negation phrase must be to
                              # a keyword to count as actually negating it


def _has_contradiction(query: str, answer: str, chunk_text: str) -> bool:
    """
    Return True only when the answer appears to contradict a positive
    assertion in the source chunks.

    Requires, symmetrically on BOTH sides:
      - The answer negates a query keyword NEAR that keyword — not just
        "the answer contains a negation phrase somewhere". An honest,
        correctly-hedged answer like "X mentions Y but does not define Z"
        contains "does not" without contradicting anything about Y; the
        old version treated any negation phrase anywhere in the whole
        answer as evidence, so a query keyword like "engineer" that's
        simply the document's topic (and therefore appears "positively"
        dozens of times in the source) would false-positive against an
        unrelated "does not" three sentences away.
      - The chunk text positively asserts that SAME keyword outside of
        any negation context (i.e. the source says it exists/happened,
        the answer says it doesn't).
    """
    answer_lower = answer.lower()
    chunk_lower  = chunk_text.lower()

    query_keywords = _filter_stopwords(query)
    if not query_keywords:
        return False

    for kw in query_keywords:
        # ── Answer side: is THIS keyword actually negated nearby, not just
        #    present somewhere alongside an unrelated negation phrase? ──
        kw_idx = answer_lower.find(kw)
        keyword_is_negated_in_answer = False
        while kw_idx != -1:
            window_start = max(0, kw_idx - _NEGATION_WINDOW)
            window = answer_lower[window_start : kw_idx + len(kw) + _NEGATION_WINDOW]
            if any(phrase in window for phrase in _NEGATION_PHRASES):
                keyword_is_negated_in_answer = True
                break
            kw_idx = answer_lower.find(kw, kw_idx + 1)

        if not keyword_is_negated_in_answer:
            continue

        # ── Chunk side: does the source positively assert this same keyword? ──
        kw_idx = chunk_lower.find(kw)
        while kw_idx != -1:
            window_start = max(0, kw_idx - _NEGATION_WINDOW)
            window = chunk_lower[window_start : kw_idx + len(kw) + _NEGATION_WINDOW]
            if not any(phrase in window for phrase in _NEGATION_PHRASES):
                # Chunk positively asserts this keyword → real contradiction
                return True
            kw_idx = chunk_lower.find(kw, kw_idx + 1)

    return False




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
