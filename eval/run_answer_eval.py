"""
eval/run_answer_eval.py - Offline end-to-end answer-quality regression check.

run_eval.py only checks retrieval (did the right chunk get found); it never
calls the LLM. This script closes that gap: it runs each golden query through
the REAL pipeline (chatbot.intent -> agents.safety -> agents.loop, the same
path main.py's /chat endpoint uses) against the eval fixtures, and checks
that the generated answer actually contains the expected fact - not just
that the right chunk was retrieved.

This is a judgment call, not a full LLM-as-judge eval: it's a keyword
presence check on the final answer text, same spirit as run_eval.py's
retrieval keyword-grounding check, just one hop further down the pipeline.
It will not catch subtly wrong phrasing or a correct-sounding hallucination
that happens to include the expected keyword - it catches the case that
matters most in practice, an answer that drops or gets the queried fact
wrong.

Needs a real GROQ_API_KEY (unlike run_eval.py) and makes real LLM calls, so
it is NOT wired into the default CI test job - run it manually, or add a
dedicated CI job with a GROQ_API_KEY secret if you want it gated on every PR.

Usage:
    python eval/run_answer_eval.py

Exit code is non-zero if hit-rate drops below MIN_HIT_RATE.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

from chatbot.intent import detect_intent
from agents.safety import safety_check
from agents.loop import run_reflection_loop
from eval.run_eval import build_store, FIXTURES_DIR, GOLDEN_PATH  # noqa: F401 (FIXTURES_DIR re-export)

MIN_HIT_RATE = 0.8  # fail the run if fewer than 80% of golden queries hit


def evaluate(store) -> dict:
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        golden = json.load(f)

    hits = 0
    rows = []

    for case in golden:
        query = case["query"]
        expected_answer_keywords = [k.lower() for k in case.get("expected_answer_keywords", [])]

        intent = detect_intent(query)
        safety = safety_check(query)
        if not safety["safe"]:
            rows.append({"query": query, "answer": "(blocked by safety check)", "hit": False})
            continue

        result = run_reflection_loop(query, store, intent)
        answer = result.get("answer", "")

        hit = any(kw in answer.lower() for kw in expected_answer_keywords)
        hits += int(hit)
        rows.append({
            "query": query,
            "answer": answer,
            "flagged_reason": result.get("reflection_reason", "not_reflected"),
            "hit": hit,
        })

    hit_rate = hits / len(golden) if golden else 0.0
    return {"hit_rate": hit_rate, "hits": hits, "total": len(golden), "rows": rows}


def print_report(result: dict) -> None:
    print("\n-- Answer-quality (end-to-end, real LLM calls) --")
    for row in result["rows"]:
        mark = "PASS" if row["hit"] else "FAIL"
        print(f"  [{mark}] {row['query']!r}")
        print(f"         answer={row['answer'][:200]!r}")
    print(f"  hit_rate = {result['hit_rate']:.0%} ({result['hits']}/{result['total']})")


def main() -> int:
    store = build_store()
    print(f"Built eval store: {store.total_chunks()} chunks across {len(store.files)} fixture file(s).")
    store.wait_for_embeddings()

    result = evaluate(store)
    print_report(result)

    ok = result["hit_rate"] >= MIN_HIT_RATE
    print(f"\n{'PASS' if ok else 'FAIL'} - minimum hit-rate threshold: {MIN_HIT_RATE:.0%}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
