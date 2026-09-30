"""
retrieval/warmup.py — Load the embedding and reranker models at startup.

Both models load lazily on first use, and the first inference pays a further
one-time cost, so the first question after every restart used to take about
8 seconds longer than every later one (measured: first rerank 8.1s, steady
0.6s). Warming them on a background thread at startup moves that cost out of
the first user's wait. The server still accepts requests immediately; a
question that arrives mid-warm-up simply waits on the same lock the lazy
loader already uses, so nothing is loaded twice.
"""

import time


def warm_up_models() -> float:
    """
    Run one tiny embedding and one tiny rerank so both models are loaded and
    primed. Never raises: a failure here only means the first real query pays
    the cost, exactly as before. Returns the seconds spent.
    """
    started = time.perf_counter()

    try:
        from retrieval.dense import embedder
        embedder.embed("warm up")
    except Exception as exc:
        print(f"  [WARN]  [warmup] Embedding model warm-up failed: {exc}")

    try:
        from retrieval.reranker import rerank
        rerank(
            [{"chunk_id": "warmup", "text": "warm up text", "metadata": {}}],
            query="warm up",
            final_k=1,
        )
    except Exception as exc:
        print(f"  [WARN]  [warmup] Reranker warm-up failed: {exc}")

    elapsed = time.perf_counter() - started
    print(f"[WARM] Models warmed in {elapsed:.1f}s")
    return elapsed
