def detect_intent(query: str) -> str:
    """
    Deterministic intent detector (no LLM).
    Returns one of: "compare", "summarize", "explain", "qa".
    """
    q = query.lower()

    compare_kw   = {"compare", "difference", "versus", "vs", "between", "contrast", "similarities", "differences"}
    summarize_kw = {"summarize", "summary", "overview", "brief", "key points"}
    explain_kw   = {"explain", "what does", "meaning", "clarify", "simple", "how does"}

    if any(kw in q for kw in compare_kw):
        return "compare"
    if any(kw in q for kw in summarize_kw):
        return "summarize"
    if any(kw in q for kw in explain_kw):
        return "explain"
    return "qa"
