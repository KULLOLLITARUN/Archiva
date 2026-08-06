"""
ingestion/tokenizer.py — Single source-of-truth tokenizer for BM25.

Fix #4: Previously _tokenize() was duplicated in both embedder.py and
search.py. Any drift between the two silently degraded search quality
because the index and query tokens wouldn't match.

Import _tokenize from here in both embedder.py and search.py.
"""

import re
from typing import List


def _tokenize(text: str) -> List[str]:
    """
    Lowercase and split on whitespace/punctuation boundaries.
    Keeps alphanumeric tokens only — consistent with BM25 best practice.
    """
    return re.findall(r"[a-z0-9]+", text.lower())
