"""
Fast regression smoke test built on eval/golden_queries.json.

This is the BM25-only half of eval/run_eval.py, kept in the regular pytest
suite (no model download, no network) so a retrieval regression fails CI
immediately. Run `python eval/run_eval.py` for the full BM25 + hybrid/dense
report when you want the complete picture.
"""

import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
sys.path.insert(0, str(EVAL_DIR.parent))

from eval.run_eval import GOLDEN_PATH, MIN_HIT_RATE, build_store, evaluate


def test_bm25_hit_rate_meets_minimum_threshold():
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        golden = json.load(f)

    store = build_store()
    result = evaluate(store, golden, use_dense=False)

    failures = [row for row in result["rows"] if not row["hit"]]
    assert result["hit_rate"] >= MIN_HIT_RATE, (
        f"BM25 hit-rate {result['hit_rate']:.0%} below {MIN_HIT_RATE:.0%} threshold. "
        f"Failing queries: {[row['query'] for row in failures]}"
    )
