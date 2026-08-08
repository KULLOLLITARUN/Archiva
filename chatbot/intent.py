def detect_intent(query: str) -> str:
    """
    Deterministic intent detector (no LLM).
    Returns one of: "meta", "compare", "summarize", "explain", "qa".
    """
    q = query.lower()

    # "do we have" alone is too broad — tighten to document-inventory forms
    meta_kw      = {"how many doc", "how many file", "how many docx", "how many pdf",
                    "list file", "list doc", "what doc", "what file", "which file",
                    "loaded doc", "uploaded doc",
                    "do we have any doc", "do we have any file",
                    "how many loaded", "show me the doc", "show me the file"}
    compare_kw   = {"compare", "difference", "versus", "vs", "between", "contrast", "similarities", "differences"}
    summarize_kw = {"summarize", "summary", "overview", "brief", "key points"}
    explain_kw   = {"explain", "what does", "meaning", "clarify", "simple", "how does"}

    if any(kw in q for kw in meta_kw):
        return "meta"
    if any(kw in q for kw in compare_kw):
        return "compare"
    if any(kw in q for kw in summarize_kw):
        return "summarize"
    if any(kw in q for kw in explain_kw):
        return "explain"
    return "qa"
