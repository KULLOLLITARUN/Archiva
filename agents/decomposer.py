"""
agents/decomposer.py — Multi-hop query decomposition.

Detects when a user's question actually contains multiple DISTINCT,
independently-answerable sub-questions (as opposed to a single complex
question that merely touches multiple topics, e.g. the existing "compare"
intent) and splits it so each part gets its own retrieve->generate->reflect
pass through agents/loop.py, then the answers are merged.

Cost control: should_decompose() is a free, deterministic pre-filter run on
every query. The LLM call in decompose_query() only fires when that
pre-filter says a query is plausibly multi-part — most queries never reach
it, keeping this close to free for the common case.

Sequential/dependent sub-questions: decompose_query() is told to produce
self-contained sub-questions, but a small fast model asked to "preserve
original wording" will often leave a pronoun in a later part ("...and how
fast must THEY respond to alerts") that plainly refers back to an earlier
sub-question's answer ("who is on the response team"). agents/loop.py runs
sub-questions in order (not in parallel) and anchor_to_prior_answer() below
resolves that reference before the dependent sub-question runs — same
anchor-don't-rewrite pattern chatbot/rewriter.py already uses for
conversational follow-ups, applied here within one decomposed query's
sub-question chain instead of across chat turns.
"""

import json
import re
from typing import List

from config import GROQ_FAST
from llm.groq_manager import groq_manager

# ── Configuration ─────────────────────────────────────────────────────────────

_MIN_WORDS_TO_CONSIDER = 10
_CONNECTORS = (" and ", " also ", " as well as ", ";")

_SYSTEM_PROMPT = (
    "You are a query analysis agent for a document Q&A system. Decide whether "
    "the user's question contains multiple DISTINCT, independently-answerable "
    "sub-questions, as opposed to a single question that merely mentions "
    "several things.\n\n"
    "Examples:\n"
    '- "What is the vacation policy and how fast must oncall respond to alerts?" '
    "-> TWO unrelated sub-questions.\n"
    '- "Compare the vacation policy across regions" -> ONE question (a single '
    "comparison, not two separate asks).\n\n"
    "Rules:\n"
    "- If it is a single question, return a JSON array with exactly ONE item: "
    "the original question, unchanged.\n"
    "- If it has 2-4 distinct sub-questions, return each as its own "
    "self-contained question, preserving the original wording where possible.\n"
    "- Never invent a sub-question that isn't implied by the original text.\n"
    "- Return ONLY a JSON array of strings. No markdown, no explanation."
)

_MAX_TOKENS = 200
_TEMPERATURE = 0.0
_MAX_SUBQUESTIONS = 4


# ── Cheap pre-filter (no LLM call) ─────────────────────────────────────────────

def should_decompose(query: str) -> bool:
    """
    Deterministic pre-filter: True means "plausibly multi-part, worth
    checking with the LLM" — NOT "definitely decompose". Keeps the LLM call
    in decompose_query() off the hot path for ordinary short/simple queries.
    """
    q = query.strip()
    if len(q.split()) < _MIN_WORDS_TO_CONSIDER:
        return False
    if q.count("?") >= 2:
        return True
    lowered = q.lower()
    return any(conn in lowered for conn in _CONNECTORS)


# ── LLM-assisted split ─────────────────────────────────────────────────────────

def decompose_query(query: str) -> List[str]:
    """
    Ask the LLM whether *query* splits into distinct sub-questions.

    Returns a list of 1+ questions. A single-item list (usually [query]
    itself) means "don't decompose" — callers should treat that as a no-op.
    Falls back to [query] on any error, same failure-mode convention as
    agents/query_rewriter.py.
    """
    try:
        _, client = groq_manager.get_client()
        response = client.chat.completions.create(
            model=GROQ_FAST,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=_TEMPERATURE,
            max_tokens=_MAX_TOKENS,
        )
        raw = (response.choices[0].message.content or "").strip()
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return [query]

        parsed = json.loads(match.group())
        sub_queries = [str(q).strip() for q in parsed if isinstance(q, str) and q.strip()]
        if not sub_queries:
            return [query]
        return sub_queries[:_MAX_SUBQUESTIONS]

    except Exception as exc:
        print(f"  [WARN]  [decomposer] Decomposition failed: {exc} — treating as a single question")
        return [query]


# ── Sequential reference resolution ─────────────────────────────────────────────

# Deliberately narrow — only pronouns that are rarely anything OTHER than a
# back-reference in a short question. "that"/"this"/"these" are excluded on
# purpose: they're common as relative-clause/demonstrative words ("the team
# THAT manages the roadmap") and would false-trigger constantly if included.
_REFERENCE_WORDS = ("it", "its", "they", "their", "them")
_REFERENCE_RE = re.compile(r"\b(" + "|".join(_REFERENCE_WORDS) + r")\b", re.IGNORECASE)


def anchor_to_prior_answer(sub_query: str, prior_answer: str) -> str:
    """
    If sub_query contains a pronoun that plausibly points at the previous
    sub-question's answer ("what is THEIR vacation policy"), anchor it to
    that answer so both retrieval and generation see the resolved
    reference. A no-op when there's nothing to resolve — most decomposed
    sub-question pairs are genuinely independent, and this must not fire
    on every short sub-question the way chatbot.rewriter.is_followup()'s
    "<6 words" fallback would (many self-contained sub-questions ARE
    short — "What is the vacation policy?" is 5 words).
    """
    if not prior_answer or not _REFERENCE_RE.search(sub_query):
        return sub_query
    return f"{sub_query} (referring to: {prior_answer})"
