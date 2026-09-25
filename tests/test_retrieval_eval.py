"""
Fast regression smoke test built on eval/golden_queries.json.

This mirrors both halves of eval/run_eval.py so a retrieval regression
fails CI immediately, not just on a manual run:
  - BM25-only: no model download, always runs.
  - Hybrid (BM25 + dense + RRF): this does need the sentence-transformer
    model, but that's not new cost — retrieval/store.py's add_file()
    (used by build_store() below, and by test_reingest.py /
    test_tabular_retrieval.py elsewhere in this suite) already triggers a
    real embedding precompute regardless of whether this test exists.
    Skips gracefully (not fails) if dense retrieval genuinely can't run
    in a given environment (e.g. sentence-transformers not installed),
    same fail-open convention as retrieval/dense.py itself.

Run `python eval/run_eval.py` for the full human-readable report.
"""

import json
import sys
from pathlib import Path

import pytest

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
sys.path.insert(0, str(EVAL_DIR.parent))

from eval.run_eval import GOLDEN_PATH, MIN_HIT_RATE, build_store, evaluate
from retrieval.dense import _DENSE_AVAILABLE


def _load_golden() -> list:
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_bm25_hit_rate_meets_minimum_threshold():
    golden = _load_golden()
    store = build_store()
    result = evaluate(store, golden, use_dense=False)

    failures = [row for row in result["rows"] if not row["hit"]]
    assert result["hit_rate"] >= MIN_HIT_RATE, (
        f"BM25 hit-rate {result['hit_rate']:.0%} below {MIN_HIT_RATE:.0%} threshold. "
        f"Failing queries: {[row['query'] for row in failures]}"
    )


@pytest.mark.skipif(not _DENSE_AVAILABLE, reason="sentence-transformers not installed")
def test_hybrid_hit_rate_meets_minimum_threshold():
    golden = _load_golden()
    store = build_store()
    store.wait_for_embeddings()
    result = evaluate(store, golden, use_dense=True)

    failures = [row for row in result["rows"] if not row["hit"]]
    assert result["hit_rate"] >= MIN_HIT_RATE, (
        f"Hybrid hit-rate {result['hit_rate']:.0%} below {MIN_HIT_RATE:.0%} threshold. "
        f"Failing queries: {[row['query'] for row in failures]}"
    )
