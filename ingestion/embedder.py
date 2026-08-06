"""
ingestion/embedder.py — BM25 index builder.

Replaces the old vector-embedding layer with a pure keyword index.
No API keys, no model downloads, no network calls.

Public API (unchanged signatures for compatibility):
    build_bm25_index(chunks)  -> BM25Okapi
    EmbeddingError            -> kept as alias so import lines in main.py don't break
    EmbeddingQuotaError       -> same

Fix #4: _tokenize() now imported from ingestion.tokenizer (single source of truth).
"""

from typing import List

from rank_bm25 import BM25Okapi

# Fix #4: import shared tokenizer instead of defining a duplicate here
from ingestion.tokenizer import _tokenize  # noqa: F401 — re-exported for any legacy imports


# ── Compatibility shims (main.py catches these) ──────────────────────────────

class EmbeddingError(RuntimeError):
    """Kept for interface compatibility."""


class EmbeddingQuotaError(EmbeddingError):
    """Kept for interface compatibility."""


# ── Index builder ─────────────────────────────────────────────────────────────

def build_bm25_index(chunks: List[dict]) -> BM25Okapi:
    """
    Build a BM25 index over the provided chunk list.

    Args:
        chunks: list of chunk dicts, each containing a ``text`` key.

    Returns:
        A fitted BM25Okapi instance.
    """
    corpus = [_tokenize(chunk["text"]) for chunk in chunks]
    return BM25Okapi(corpus)
