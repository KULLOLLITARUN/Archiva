"""
eval/run_eval.py - Offline retrieval-quality regression check.

Builds an in-memory store from eval/fixtures/, runs each query in
golden_queries.json through both BM25-only and the full hybrid (BM25 +
dense + RRF) pipeline, and reports hit@k (did the expected source file show
up in the top-k results) plus a keyword-grounding spot-check.

This is a retrieval-quality smoke test, not an end-to-end answer-quality
eval - it doesn't call the LLM, so it's fast and needs no API key. Run it
after touching retrieval/search.py, retrieval/dense.py, retrieval/reranker.py,
ingestion/chunker.py, or any config threshold that affects retrieval, to
catch a regression before it reaches a real query.

Usage:
    python eval/run_eval.py            # runs BM25 + hybrid, prints report
    python eval/run_eval.py --bm25-only  # skip dense (no model download)

Exit code is non-zero if hit-rate for either mode drops below MIN_HIT_RATE,
so this can be wired into CI.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.chunker import chunk_document
from ingestion.parser import compute_hash, parse_file
from retrieval.search import search as bm25_search
from retrieval.store import MultiDocStore

FIXTURES_DIR = Path(__file__).parent / "fixtures"
GOLDEN_PATH = Path(__file__).parent / "golden_queries.json"
TOP_K = 5
MIN_HIT_RATE = 0.8  # fail the run if fewer than 80% of golden queries hit


def build_store() -> MultiDocStore:
    store = MultiDocStore()
    for path in sorted(FIXTURES_DIR.glob("*.txt")):
        content = path.read_bytes()
        pages = parse_file(content, path.name)
        file_id = path.stem
        chunks = chunk_document(pages=pages, file_id=file_id, filename=path.name, file_type="txt")
        store.add_file(
            file_id=file_id, filename=path.name,
            content_hash=compute_hash(content), chunks=chunks, file_type="txt",
        )
    return store


def evaluate(store: MultiDocStore, golden: list, use_dense: bool) -> dict:
    hits = 0
    rows = []

    for case in golden:
        query = case["query"]
        expected_file = case["expected_filename"]
        expected_keywords = [k.lower() for k in case["expected_keywords"]]

        if use_dense:
            from retrieval.search import hybrid_retrieve
            results = hybrid_retrieve(query, store, top_k=TOP_K)
        else:
            results = bm25_search(query, store, top_k=TOP_K)

        retrieved_files = [r["metadata"]["filename"] for r in results]
        file_hit = expected_file in retrieved_files

        top_text = " ".join(r["text"].lower() for r in results)
        keyword_hit = any(kw in top_text for kw in expected_keywords)

        hit = file_hit and keyword_hit
        hits += int(hit)
        rows.append({
            "query": query, "expected": expected_file,
            "retrieved": retrieved_files, "hit": hit,
        })

    hit_rate = hits / len(golden) if golden else 0.0
    return {"hit_rate": hit_rate, "hits": hits, "total": len(golden), "rows": rows}


def print_report(label: str, result: dict) -> None:
    print(f"\n-- {label} --")
    for row in result["rows"]:
        mark = "PASS" if row["hit"] else "FAIL"
        print(f"  [{mark}] {row['query']!r}")
        print(f"         expected={row['expected']!r} retrieved={row['retrieved']}")
    print(f"  hit_rate = {result['hit_rate']:.0%} ({result['hits']}/{result['total']})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bm25-only", action="store_true",
                         help="Skip the hybrid/dense pass (no model download).")
    args = parser.parse_args()

    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        golden = json.load(f)

    store = build_store()
    print(f"Built eval store: {store.total_chunks()} chunks across {len(store.files)} fixture file(s).")

    bm25_result = evaluate(store, golden, use_dense=False)
    print_report("BM25-only", bm25_result)

    ok = bm25_result["hit_rate"] >= MIN_HIT_RATE

    if not args.bm25_only:
        store.wait_for_embeddings()
        hybrid_result = evaluate(store, golden, use_dense=True)
        print_report("Hybrid (BM25 + dense + RRF)", hybrid_result)
        ok = ok and hybrid_result["hit_rate"] >= MIN_HIT_RATE

    print(f"\n{'PASS' if ok else 'FAIL'} - minimum hit-rate threshold: {MIN_HIT_RATE:.0%}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
