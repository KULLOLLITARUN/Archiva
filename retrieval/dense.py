"""
retrieval/dense.py — Sentence-transformer dense retrieval with disk cache.

Part 2.1 / Part 3.3: Provides dense vector search as the second leg of the
hybrid BM25 + dense retrieval pipeline.

Design:
  - SentenceTransformerEmbedder: lazy-loads the model on first use so startup
    is not blocked.  Thread-safe (lock around model load).
  - Embeddings are cached to disk via diskcache; cache key = SHA-256(text).
  - precompute_embeddings(store) eagerly embeds every chunk in the store and
    stores the embedding under chunk["_embedding"] so query-time is fast.
  - dense_search(query, store, top_k) uses cosine similarity.

Fallback: if sentence-transformers / numpy are unavailable (pkg not installed),
  dense retrieval returns [] silently so BM25-only mode still works.
"""

import threading
from typing import List, Optional

# ── Graceful import (dense retrieval is optional) ────────────────────────────

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    _DENSE_AVAILABLE = True
except ImportError:
    _DENSE_AVAILABLE = False
    np = None               # type: ignore[assignment]

try:
    import diskcache
    _CACHE_AVAILABLE = True
except ImportError:
    _CACHE_AVAILABLE = False
    diskcache = None        # type: ignore[assignment]

from config import DENSE_MODEL, EMBEDDING_CACHE_DIR


# ── Embedder singleton ────────────────────────────────────────────────────────

class SentenceTransformerEmbedder:
    """
    Lazy-loading sentence-transformer embedder with disk-cache for embeddings.

    Thread-safe: the model is loaded once behind a lock on first call.
    """

    def __init__(self, model_name: str = DENSE_MODEL, cache_dir: str = EMBEDDING_CACHE_DIR) -> None:
        self._model_name = model_name
        self._model: Optional[object] = None
        self._lock = threading.Lock()
        self._cache = None
        self._cache_dir = cache_dir

    def _load(self) -> None:
        """Load the model and cache; idempotent after first call."""
        if self._model is not None:
            return
        if not _DENSE_AVAILABLE:
            return
        with self._lock:
            if self._model is not None:
                return  # double-checked locking
            print(f"  [DL]  [dense] Loading sentence-transformer: {self._model_name}")
            self._model = SentenceTransformer(self._model_name)
            if _CACHE_AVAILABLE:
                self._cache = diskcache.Cache(self._cache_dir)
            print(f"  [OK]  [dense] Model ready.")

    def embed(self, text: str) -> Optional["np.ndarray"]:
        """
        Return a 1-D numpy embedding for *text*, using the disk cache if available.
        Returns None if dense retrieval is not available.
        """
        if not _DENSE_AVAILABLE:
            return None

        self._load()
        if self._model is None:
            return None

        # Cache lookup
        if self._cache is not None:
            cached = self._cache.get(text)
            if cached is not None:
                return np.array(cached)

        vec = self._model.encode(text, normalize_embeddings=True)  # type: ignore[union-attr]

        # Store in cache
        if self._cache is not None:
            self._cache.set(text, vec.tolist())

        return vec

    def embed_batch(self, texts: List[str]) -> Optional["np.ndarray"]:
        """
        Batch-encode a list of texts, using cache where possible.
        Returns an (N, D) array or None if dense is unavailable.
        """
        if not _DENSE_AVAILABLE or not texts:
            return None

        self._load()
        if self._model is None:
            return None

        results = []
        uncached_idxs = []
        uncached_texts = []

        for i, text in enumerate(texts):
            if self._cache is not None:
                cached = self._cache.get(text)
                if cached is not None:
                    results.append((i, np.array(cached)))
                    continue
            uncached_idxs.append(i)
            uncached_texts.append(text)

        if uncached_texts:
            vecs = self._model.encode(uncached_texts, normalize_embeddings=True)  # type: ignore[union-attr]
            for local_i, (idx, text) in enumerate(zip(uncached_idxs, uncached_texts)):
                vec = vecs[local_i]
                if self._cache is not None:
                    self._cache.set(text, vec.tolist())
                results.append((idx, vec))

        if not results:
            return None

        # Reconstruct in original order
        results.sort(key=lambda t: t[0])
        return np.stack([v for _, v in results])


# ── Module-level singleton ────────────────────────────────────────────────────

embedder = SentenceTransformerEmbedder()


# ── Precompute (store build time) ─────────────────────────────────────────────

def precompute_embeddings(chunks: List[dict]) -> None:
    """
    Embed all chunks and store the vector under chunk["_embedding"].

    Called after add_file() so that query-time dense search only needs to
    embed the query (not the chunks). Safe to call multiple times — already-
    embedded chunks are skipped.
    """
    if not _DENSE_AVAILABLE:
        return

    to_embed = [c for c in chunks if "_embedding" not in c]
    if not to_embed:
        return

    texts = [c["text"] for c in to_embed]
    vecs  = embedder.embed_batch(texts)

    if vecs is None:
        return

    for chunk, vec in zip(to_embed, vecs):
        chunk["_embedding"] = vec


# ── Dense search ──────────────────────────────────────────────────────────────

def dense_search(
    query: str,
    store,          # MultiDocStore — typed loosely to avoid circular import
    top_k: int = 20,
    file_ids: Optional[List[str]] = None,
) -> List[dict]:
    """
    Return top_k chunks ranked by cosine similarity to the query embedding.

    Returns [] if dense retrieval is unavailable (graceful fallback).
    Cosine similarity works because both query and chunk embeddings are
    L2-normalised (normalize_embeddings=True in encode()).
    """
    if not _DENSE_AVAILABLE:
        return []

    query_vec = embedder.embed(query)
    if query_vec is None:
        return []

    pool = store.get_all_chunks()

    if file_ids:
        pool = [c for c in pool if c["metadata"]["file_id"] in file_ids]

    # Only chunks that have been pre-embedded
    embedded = [(c, c["_embedding"]) for c in pool if "_embedding" in c]
    if not embedded:
        # Fall back: embed on the fly (happens only before precompute runs)
        print("  [WARN]  [dense] No pre-computed embeddings — computing on the fly.")
        texts = [c["text"] for c in pool]
        vecs  = embedder.embed_batch(texts)
        if vecs is None:
            return []
        embedded = list(zip(pool, vecs))

    if not embedded:
        return []

    chunks_list, vecs_list = zip(*embedded)
    matrix = np.stack(vecs_list)           # (N, D)
    scores = matrix @ query_vec            # cosine sim (both normalised)

    scored = [
        {**chunk, "score": float(score)}
        for chunk, score in zip(chunks_list, scores)
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]
