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

# Words that negate whatever they govern, matched on word boundaries (so "no"
# never fires inside "know"/"note", nor "not" inside "another"). Contractions
# ("doesn't", "isn't", ...) are covered by the n't alternative.
_NEGATION_WORD_RE: re.Pattern = re.compile(
    r"\b(?:no|not|never|none|neither|nor|without|cannot|\w+n't)\b"
)

_NUMBER_PATTERN: re.Pattern = re.compile(r"\b\d[\d,\.%/-]*\b")

# Numbers that label a document's structure rather than state a fact. The
# grounding checks require every number in an answer to appear verbatim in the
# retrieved text, and these failed it whenever the model numbered its own list
# ("5. **Phase 4**") or named a step the retrieved chunks spell differently
# ("Step 1.2" vs "STEP 1.2 — ..."), so a correct, cited how-to answer was
# replaced with "Not found in the document". They are removed from the answer
# before the check; every other number (amounts, dates, ports, versions) must
# still be in the source.
#   "Step 1.2", "Phase 4", "Section 3.1.2"
_LABEL_WORD_NUMBER_RE: re.Pattern = re.compile(
    r"\b(?:step|phase|section|part|stage|chapter|clause|article|appendix|item|task)s?"
    r"\s*#?\s*\d+(?:\.\d+)*",
    re.IGNORECASE,
)
#   "5. Do this" / "- 2) Do that" at the start of a line
_LIST_ORDINAL_RE: re.Pattern = re.compile(
    r"^\s*(?:[-*•]\s+)?\d{1,2}[.)](?=\s)", re.MULTILINE,
)
#   "1.1 Create a snapshot" at the start of a line, a table cell, or after <br>,
#   unless a unit follows ("1.5 GB" is a fact, not a label).
_OUTLINE_LABEL_RE: re.Pattern = re.compile(
    r"(?:^|\||<br\s*/?>)\s*(?:[-*•]\s+)?(\d{1,2}(?:\.\d{1,2})+)"
    r"(?=\s+(?!(?:[kmgt]i?b|bytes?|ms|s|sec|seconds?|min|minutes?|h|hrs?|hours?|days?"
    r"|weeks?|months?|years?|x|percent|ghz|mhz|cores?|vcpus?|users?)\b)[A-Za-z])",
    re.IGNORECASE | re.MULTILINE,
)

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


def strip_structural_numbers(text: str) -> str:
    """*text* with list ordinals and step/section labels blanked out (see above)."""
    text = _LABEL_WORD_NUMBER_RE.sub(" ", text)
    text = _LIST_ORDINAL_RE.sub(" ", text)
    return _OUTLINE_LABEL_RE.sub(lambda m: m.group(0).replace(m.group(1), " "), text)


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
    normalized_answer = _normalize_number_text(strip_structural_numbers(answer))
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


# How far BEFORE a keyword a negation word may sit and still be said to
# negate it: at most this many words between the two, all in the same clause.
# "There is no warranty coverage" -> negates "warranty"; "no coverage for
# accidental damage" -> negates "accidental". But "it never modifies the
# original VM" does NOT negate "vm" (3 words in between): that negation
# belongs to "modifies", and the sentence still asserts the VM exists.
_NEGATION_REACH_WORDS: int = 2
_NEGATION_LOOKBACK_CHARS: int = 60   # cap on text scanned before a keyword

# Words/punctuation that end a clause. A negation never reaches across these,
# so "...specialized VM -> approach is safer because it never modifies..."
# can't smuggle "never" back onto a keyword in an earlier clause.
_CLAUSE_BREAK_RE: re.Pattern = re.compile(
    r"[.;:!?\n()\[\]\u2192\u2014\u2013]|,|\b(?:because|but|so|which|while|whereas|however|although)\b"
)

_TRAILING_NEGATION_RE: re.Pattern = re.compile(r"\bnot (?:mentioned|stated|specified|provided|covered)\b")
_TRAILING_REACH_WORDS: int = 3      # "X is not mentioned": negation FOLLOWS the keyword


def _is_negated_at(text: str, idx: int, end: int) -> bool:
    """
    True if the keyword occurrence text[idx:end] is negated BY a negation
    word that governs it: one from _NEGATION_WORD_RE within
    _NEGATION_REACH_WORDS words before it in the same clause, or
    "not mentioned" (and similar) within a few words after it in the same clause.
    """
    before = text[max(0, idx - _NEGATION_LOOKBACK_CHARS):idx]
    breaks = list(_CLAUSE_BREAK_RE.finditer(before))
    if breaks:
        before = before[breaks[-1].end():]

    negations = list(_NEGATION_WORD_RE.finditer(before))
    if negations:
        gap_words = before[negations[-1].end():].split()
        if len(gap_words) <= _NEGATION_REACH_WORDS:
            return True

    after = text[end: end + _NEGATION_LOOKBACK_CHARS]
    clause_end = _CLAUSE_BREAK_RE.search(after)
    if clause_end:
        after = after[:clause_end.start()]
    leading = " ".join(after.split()[: _TRAILING_REACH_WORDS + 2])
    return _TRAILING_NEGATION_RE.search(leading) is not None


_STEM_SUFFIXES = ("ing", "ed", "es", "s", "e")


def _stem(word: str) -> str:
    """Strip one common English ending: deallocated/deallocate -> deallocat."""
    for suffix in _STEM_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _occurrences(text: str, kw: str):
    """
    (start, end) of every whole-word occurrence of *kw* or a simple inflection
    of it. Whole-word: the query word "run" must not match inside "runbook"
    (seen live - the source title "VM Cloning Runbook" counted as the source
    asserting "run", and a correct "does not run Sysprep" was flagged).
    Inflections: "deallocated" in the answer and "does not deallocate" in the
    source are the same word; exact matching missed that the source negates
    it and flagged a correct "it is not deallocated".
    """
    pattern = re.compile(r"\b" + re.escape(_stem(kw)) + r"(?:e|es|ed|ing|s|d)?\b")
    for match in pattern.finditer(text):
        yield match.start(), match.end()


def _has_contradiction(query: str, answer: str, chunk_text: str) -> bool:
    """
    Return True only when the answer appears to contradict the source about
    a query keyword.

    For a keyword to count, ALL of these must hold:
      - The answer negates it: a negation phrase GOVERNS the keyword (same
        clause, a few words before it - see _is_negated_at), not merely
        sits somewhere near it. An honest answer like "...the specialized VM
        approach is safer because it never modifies the original" contains
        "never" right next to "specialized" without denying anything about
        it; proximity alone used to flag that as a contradiction.
      - The question itself doesn't negate it. "What happens if the zones
        don't match?" answered with "...if the zones do not match, creation
        fails" merely repeats the question's own condition.
      - The source asserts the same keyword and NEVER negates it anywhere.
        If the source itself uses the keyword in a negated sense somewhere
        ("safe, no sysprep" / "does not run sysprep"), an answer negating it
        is faithful to the source, not contradicting it - and documents that
        contrast an old and a new approach do this constantly.

    Still a cheap heuristic, not entailment: it can miss subtle
    contradictions. It is tuned to avoid flagging correct answers, because a
    false flag triggers a retry and a "possible contradiction / 0%
    confident" badge on a right answer, which is worse than a miss here.
    """
    answer_lower = answer.lower()
    chunk_lower  = chunk_text.lower()
    query_lower  = query.lower()

    for kw in _filter_stopwords(query):
        if not any(_is_negated_at(answer_lower, s, e) for s, e in _occurrences(answer_lower, kw)):
            continue
        if any(_is_negated_at(query_lower, s, e) for s, e in _occurrences(query_lower, kw)):
            continue   # the QUESTION negates it ("...if they don't match"); echoing that is not a contradiction

        source_positions = list(_occurrences(chunk_lower, kw))
        if not source_positions:
            continue
        if any(_is_negated_at(chunk_lower, s, e) for s, e in source_positions):
            continue   # the source negates it somewhere too: consistent, not contradictory
        return True

    return False




def _adds_grounded_fact(query: str, answer: str, chunks: List[dict]) -> bool:
    """True if *answer* has a content word that the chunks contain but the query doesn't."""
    new_words = _filter_stopwords(answer) - _filter_stopwords(query)
    return bool(new_words & _filter_stopwords(_all_chunk_text(chunks)))


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
    # A terse answer is fine when it carries a fact from the documents
    # ("Daniel Okafor is the account manager." is 6 words). Only short
    # answers that add nothing beyond the question ("Yes.", an echo of the
    # question) are retried.
    word_count = len(answer_stripped.split())
    if word_count < _MIN_ANSWER_WORDS and not _adds_grounded_fact(query, answer_stripped, chunks):
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
