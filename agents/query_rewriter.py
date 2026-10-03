"""
agents/query_rewriter.py — LLM-assisted BM25 query rewriter.

Called ONLY when reflection decides retry_search.
Uses GROQ_FAST with low token budget so the overhead is minimal.

SonarQube notes:
  - Single responsibility: this module does one thing — rewrite a query.
  - Explicit return type: str (never None).
  - Narrow except clause: catches the base Exception but logs it.
  - No bare 'print' in prod — uses a structured log prefix for easy grep.
"""

from config import GROQ_FAST
from llm.groq_manager import groq_manager, light_completion_params

# ── Configuration ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a search query optimizer for a BM25 document search system. "
    "Given a question that returned poor results, produce a better "
    "keyword-focused search query. Return ONLY the query, no explanation, "
    "no quotes, max 15 words."
)

_MAX_TOKENS: int = 60            # small budget — we only need a short query
_TEMPERATURE: float = 0.3        # slightly creative but still focused
_FAILED_ANSWER_PREVIEW: int = 200  # chars of failed_answer to send in prompt
_QUERY_MAX_CHARS: int = 200      # truncate the returned query to this length


def rewrite_for_retry(
    original_query: str,
    failed_answer: str,
    reflection_reason: str,
    attempt: int,
) -> str:
    """
    Use Groq (GROQ_FAST, low token budget) to produce a better BM25 search
    query when the previous attempt did not find good results.

    Args:
        original_query:    The user's original question.
        failed_answer:     The answer text that reflection rejected.
        reflection_reason: The reason string from reflect().
        attempt:           Current attempt number (0-based from the loop).

    Returns:
        A new query string.  Falls back to *original_query* on any error.
    """
    user_message = (
        f"Original question: {original_query}\n"
        f"Poor answer received: {failed_answer[:_FAILED_ANSWER_PREVIEW]}\n"
        f"Failure reason: {reflection_reason}\n"
        f"Attempt number: {attempt}\n"
        "Write a better BM25 search query using different keywords:"
    )

    try:
        current_key, client = groq_manager.get_client()
        response = client.chat.completions.create(
            model=GROQ_FAST,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            temperature=_TEMPERATURE,
            **light_completion_params(GROQ_FAST, _MAX_TOKENS),
        )
        raw_query: str = response.choices[0].message.content or ""
        new_query = raw_query.strip()[:_QUERY_MAX_CHARS]

        if not new_query:
            # Empty response — fall back to original
            return original_query

        # Structured log: easy to grep in production logs
        print(f"  🔄 [query_rewriter] Query rewritten (attempt {attempt}): {new_query}")
        return new_query

    except Exception as exc:  # groq SDK errors, network errors, etc.
        # Never crash the loop — return the original so retrieval can continue
        print(f"  ⚠️  [query_rewriter] Rewrite failed (attempt {attempt}): {exc}")
        return original_query
