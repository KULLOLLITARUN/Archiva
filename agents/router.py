from config import GROQ_FAST, GROQ_STRONG, COMPLEX_KEYWORDS, ROUTER_LONG_QUERY


def route(query: str, top_score: float) -> str:
    """
    Deterministic router — NO LLM, NO Qwen. (RULE 21)
    Returns the model ID to use for this request.

    Fix #1: Changed threshold from 1.0 to 3.0.
    BM25 scores are unbounded (not capped at 1.0), so the old threshold
    of < 1.0 routed almost every query to the expensive GROQ_STRONG model,
    making GROQ_FAST practically unreachable.
    New rule: score < 3.0 → weak match → use strong model.
              score >= 3.0 → good keyword overlap → fast model is sufficient.
    """
    q = query.lower()

    if any(kw in q for kw in COMPLEX_KEYWORDS):
        return GROQ_STRONG

    if len(query.split()) > ROUTER_LONG_QUERY:
        return GROQ_STRONG

    # Fix #1: BM25 threshold raised from 1.0 to 3.0 so GROQ_FAST is reachable
    if top_score < 3.0:
        return GROQ_STRONG

    return GROQ_FAST
