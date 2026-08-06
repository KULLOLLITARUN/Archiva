"""
chatbot/normalizer.py — Query normalizer.

Fix #19: The old pattern [^a-zA-Z0-9\\s?.,'!] stripped all non-ASCII
         characters and all technical punctuation. This broke:
         - File paths  (/ \\ _ -)
         - Error codes ([ ] { })
         - Regex/shell patterns (^ ~ | = + < >)
         - Non-English characters (accented letters, CJK, etc.)

New pattern uses \\w (which matches Unicode word chars including
non-ASCII letters and digits) and explicitly whitelists common
technical punctuation that should survive normalization.
"""

import re


def normalize(query: str) -> str:
    """
    Strip, collapse whitespace, remove disallowed characters,
    and truncate to 500 chars.

    Fix #19: Updated character whitelist to allow:
     - Unicode letters/digits  (\\w covers a-z A-Z 0-9 _ and non-ASCII)
     - Common technical chars  / \\\\ - : @ # % ( ) [ ] { } = + < > | ^ ~
     - Standard punctuation    ? . , ' !
    """
    query = query.strip()
    query = re.sub(r"\s+", " ", query)
    # Fix #19: broadened whitelist — keep Unicode word chars + technical symbols
    query = re.sub(r"[^\w\s?.,'!\-_/:@#%()\[\]{}=+<>|^~\\]", "", query)
    return query[:500]
