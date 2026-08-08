"""
agents/context_optimizer.py — MMR deduplication, chunk compression, citations.

Part 5: Reduces hallucination by ensuring the context fed to the LLM is:
  1. Non-redundant (MMR removes similar chunks).
  2. Concise (chunks > MAX_CHUNK_TOKENS words are hard-truncated).
  3. Properly cited (each chunk gets an inline [Source: …] header).

Public API:
    optimize_context(chunks, query, top_k) → List[dict]
"""

import re
from typing import List, Optional

from config import DOCUMENT_INJECTION_PATTERNS, FINAL_K, MMR_LAMBDA, MAX_CHUNK_TOKENS

# ── Graceful numpy import ──────────────────────────────────────────────────────

try:
    import numpy as np
    _NP_AVAILABLE = True
except ImportError:
    _NP_AVAILABLE = False
    np = None  # type: ignore[assignment]


# ── Document-content injection screening ─────────────────────────────────────

def _contains_injection_pattern(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in DOCUMENT_INJECTION_PATTERNS)


def screen_injected_chunks(chunks: List[dict]) -> List[dict]:
    """
    Drop chunks whose source text matches a known prompt-injection pattern
    before they reach the LLM context.

    agents/safety.py screens the user's *query* for these phrases, but a
    retrieved chunk's text comes from an uploaded document — untrusted
    content that flows straight into the system prompt via `Context: ...`
    with no screening otherwise. An uploaded file containing "ignore
    previous instructions..." would reach the model unfiltered without
    this check.
    """
    safe = []
    for chunk in chunks:
        if _contains_injection_pattern(chunk.get("text", "")):
            meta = chunk.get("metadata", {})
            print(
                f"  [WARN]  [context_optimizer] Dropped chunk with "
                f"injection-like content: {meta.get('filename', '?')} "
                f"p{meta.get('page', '?')}"
            )
            continue
        safe.append(chunk)
    return safe


# ── MMR (Maximal Marginal Relevance) ─────────────────────────────────────────

def apply_mmr(
    chunks: List[dict],
    query: str,
    top_k: int = FINAL_K,
    lambda_: float = MMR_LAMBDA,
) -> List[dict]:
    """
    Select *top_k* chunks using Maximal Marginal Relevance.

    MMR balances relevance to *query* against redundancy among already-selected
    chunks.  lambda_=1.0 → pure relevance; lambda_=0.0 → pure diversity.

    Requires precomputed "_embedding" on each chunk (set by dense.py).
    Falls back to score-order if embeddings are unavailable.
    """
    if not _NP_AVAILABLE or not chunks:
        return chunks[:top_k]

    # Check that at least some chunks have embeddings
    embedded = [c for c in chunks if "_embedding" in c]
    if not embedded:
        return chunks[:top_k]   # no embeddings → fallback

    # We only run MMR on chunks that have embeddings; others are appended after.
    non_embedded = [c for c in chunks if "_embedding" not in c]

    from retrieval.dense import embedder
    query_vec = embedder.embed(query)
    if query_vec is None:
        return chunks[:top_k]

    # Relevance scores (cosine sim to query) — already stored in "score" or
    # recomputed from embedding to be accurate post-reranking.
    try:
        vecs = np.stack([c["_embedding"] for c in embedded if c.get("_embedding") is not None])
        if len(vecs) == 0:
            return chunks[:top_k]
        relevance = (vecs @ query_vec)                          # (N,)

        selected_idxs: List[int] = []
        remaining     = list(range(len(embedded)))

        for _ in range(min(top_k, len(embedded))):
            if not remaining:
                break

            if not selected_idxs:
                # First pick: highest relevance
                best = max(remaining, key=lambda i: relevance[i])
            else:
                selected_vecs = np.stack([vecs[i] for i in selected_idxs])  # (S, D)
                # Max similarity to any already-selected chunk
                sim_to_selected = (vecs[remaining] @ selected_vecs.T).max(axis=1)  # (R,)
                rel_remaining   = relevance[remaining]
                mmr_scores      = lambda_ * rel_remaining - (1 - lambda_) * sim_to_selected
                local_best      = int(np.argmax(mmr_scores))
                best            = remaining[local_best]

            selected_idxs.append(best)
            remaining.remove(best)

        result = [embedded[i] for i in selected_idxs]
        budget = top_k - len(result)
        result.extend(non_embedded[:budget])
        return result
    except Exception as exc:
        print(f"  [WARN]  [context_optimizer] MMR calculation failed: {exc} — fallback to top_k chunks")
        return chunks[:top_k]

    return result


# ── Chunk compression ─────────────────────────────────────────────────────────

def compress_chunk(chunk: dict, max_tokens: int = MAX_CHUNK_TOKENS) -> dict:
    """
    Hard-truncate the chunk text to *max_tokens* words.
    Returns a shallow copy with the text field trimmed (never modifies in-place).
    """
    words = chunk["text"].split()
    if len(words) <= max_tokens:
        return chunk
    trimmed = " ".join(words[:max_tokens]) + " […]"
    return {**chunk, "text": trimmed}


# ── Citation injection ────────────────────────────────────────────────────────

def attach_citations(chunks: List[dict]) -> List[dict]:
    """
    Prepend a citation header to each chunk's text so the LLM can trace back
    every claim to a source without needing to hallucinate a citation.

    Format: [Source: filename | Page N | Section X]
    """
    cited = []
    for chunk in chunks:
        meta  = chunk.get("metadata", {})
        fname = meta.get("filename", "unknown")
        page  = meta.get("page", "?")
        sec   = meta.get("section", "")
        header = f"[Source: {fname} | Page {page}" + (f" | {sec}" if sec else "") + "]"
        cited.append({**chunk, "text": f"{header}\n{chunk['text']}"})
    return cited


# ── Public entry point ────────────────────────────────────────────────────────

def optimize_context(
    chunks: List[dict],
    query: str,
    top_k: int = FINAL_K,
) -> List[dict]:
    """
    Full context optimization pipeline:
      1. Screen out chunks containing prompt-injection-like content.
      2. MMR deduplication (remove semantically redundant chunks).
      3. Compress any chunk exceeding MAX_CHUNK_TOKENS words.
      4. Attach inline citation headers.

    Args:
        chunks: Reranked chunks (from cross-encoder or score-sort).
        query:  Current search query (used for MMR relevance scoring).
        top_k:  Maximum number of chunks to keep.

    Returns:
        Optimized, cited, compressed chunk list ready for context window.
    """
    # Step 1 — Screen document-content injection attempts
    screened = screen_injected_chunks(chunks)

    # Step 2 — MMR
    diverse = apply_mmr(screened, query, top_k=top_k)

    # Step 3 — Compress long chunks
    compressed = [compress_chunk(c) for c in diverse]

    # Step 4 — Attach citations
    cited = attach_citations(compressed)

    return cited
