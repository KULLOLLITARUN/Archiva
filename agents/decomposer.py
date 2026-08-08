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
