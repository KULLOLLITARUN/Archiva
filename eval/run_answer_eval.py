"""
eval/run_answer_eval.py - End-to-end answer-quality eval (real LLM calls).

run_eval.py only checks retrieval. This runs each case in
eval/answer_cases.json through the REAL pipeline (intent -> safety -> the
reflection loop, the same path /chat uses) and grades the final answer with
eval/answer_grading.py: required facts present, "not found" when the
documents can't answer, a chained question with a missing prerequisite
skipping its dependent step, no false "possible contradiction" flag.

Categories (the failure patterns found in real use):
  single          one fact
  chained         a later step needs an earlier step's answer
  chained-missing the first step can't be answered; the rest must be skipped
  unanswerable    must say "Not found" instead of inventing something
  negation        answers full of "not"/"never" that must not be flagged
  numeric         arithmetic over values in the text (sums, differences)

Per case it records pass/fail with reasons, latency, attempts, and the REAL
token usage reported by the API for every LLM call the case made (answer,
planner, rewriter, judge, safety). Results go to eval/results/ and can be
compared against a saved baseline, so a change is judged by numbers.

Needs a Groq key and makes real calls, so it is not part of the default test
run. Groq's free tier allows about 8000 tokens a minute; a chained case can
use most of that, so the runner paces itself and, if the provider still
returns "Service temporarily unavailable", waits and retries that case
instead of scoring it as a wrong answer.

Usage:
    python eval/run_answer_eval.py                      # all cases, fixture docs
    python eval/run_answer_eval.py --only chained       # one category
    python eval/run_answer_eval.py --case chain-vendor-cost
    python eval/run_answer_eval.py --save-baseline      # record this run as the baseline
    python eval/run_answer_eval.py --store db --cases eval/answer_cases.local.json
        # your own questions against the documents loaded in Postgres (read-only)

Exit code is non-zero if the pass rate is below --min-pass, or if any case
could not be run because the provider kept failing.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.answer_grading import grade, is_unavailable, validate_case  # noqa: E402

EVAL_DIR = Path(__file__).parent
DEFAULT_CASES = EVAL_DIR / "answer_cases.json"
DEFAULT_BASELINE = EVAL_DIR / "answer_baseline.json"
RESULTS_DIR = EVAL_DIR / "results"

MIN_PASS_RATE = 0.8
UNAVAILABLE_RETRY_WAIT_S = 65     # a full rate-limit window
UNAVAILABLE_RETRIES = 2


# ── Token accounting ────────────────────────────────────────────────────────────

class TokenMeter:
    """Wraps the Groq SDK so every chat completion's real usage is counted."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.prompt = self.completion = self.calls = 0

    def install(self):
        from groq.resources.chat.completions import Completions
        original = Completions.create
        meter = self

        def counting_create(self_, *args, **kwargs):
            response = original(self_, *args, **kwargs)
            usage = getattr(response, "usage", None)
            if usage is not None:
                meter.prompt += getattr(usage, "prompt_tokens", 0) or 0
                meter.completion += getattr(usage, "completion_tokens", 0) or 0
            meter.calls += 1
            return response

        Completions.create = counting_create


# ── Running ─────────────────────────────────────────────────────────────────────

def load_cases(path: Path, only, case_id):
    with open(path, "r", encoding="utf-8") as f:
        cases = json.load(f)
    problems = {c.get("id", f"#{i}"): validate_case(c) for i, c in enumerate(cases)}
    problems = {k: v for k, v in problems.items() if v}
    if problems:
        raise SystemExit(f"Invalid cases in {path}: {problems}")
    if case_id:
        cases = [c for c in cases if c["id"] == case_id]
    if only:
        cases = [c for c in cases if c["category"] in only]
    return cases


def build_store(kind: str):
    if kind == "db":
        from db.store_sync import load_store_from_postgres   # read-only SELECTs
        return load_store_from_postgres()
    from eval.run_eval import build_store as build_fixture_store
    store = build_fixture_store()
    store.wait_for_embeddings()
    return store


def run_case(case, store, meter):
    from agents.loop import run_reflection_loop
    from agents.safety import safety_check
    from cache.semantic_cache import semantic_cache
    from chatbot.intent import detect_intent

    question = case["question"]
    for attempt in range(UNAVAILABLE_RETRIES + 1):
        semantic_cache.clear()          # every case must hit the real pipeline
        meter.reset()
        started = time.perf_counter()

        if not safety_check(question)["safe"]:
            result = {"answer": "(blocked by the safety check)", "reflection_reason": "blocked"}
        else:
            result = run_reflection_loop(question, store, detect_intent(question))
        latency = time.perf_counter() - started
        answer = result.get("answer", "")

        if not is_unavailable(answer):
            break
        if attempt < UNAVAILABLE_RETRIES:
            print(f"      provider unavailable (rate limit?) - waiting {UNAVAILABLE_RETRY_WAIT_S}s and retrying")
            time.sleep(UNAVAILABLE_RETRY_WAIT_S)

    record = {
        "id": case["id"], "category": case["category"], "question": question,
        "answer": answer, "reflection_reason": result.get("reflection_reason", ""),
        "attempts": result.get("attempts"), "latency_s": round(latency, 2),
        "prompt_tokens": meter.prompt, "completion_tokens": meter.completion, "llm_calls": meter.calls,
    }
    if is_unavailable(answer):
        record.update(status="error", passed=False, reasons=["provider unavailable after retries"])
    else:
        passed, reasons = grade(case, answer, record["reflection_reason"])
        record.update(status="pass" if passed else "fail", passed=passed, reasons=reasons)
    return record


# ── Reporting ───────────────────────────────────────────────────────────────────

def summarize(records):
    graded = [r for r in records if r["status"] != "error"]
    by_cat = defaultdict(lambda: {"pass": 0, "total": 0})
    for r in graded:
        by_cat[r["category"]]["total"] += 1
        by_cat[r["category"]]["pass"] += int(r["passed"])
    total_tokens = sum(r["prompt_tokens"] + r["completion_tokens"] for r in records)
    return {
        "cases": len(records),
        "graded": len(graded),
        "passed": sum(r["passed"] for r in graded),
        "errors": len(records) - len(graded),
        "pass_rate": round(sum(r["passed"] for r in graded) / len(graded), 3) if graded else 0.0,
        "contradiction_flags": sum("possible_contradiction" in (r["reflection_reason"] or "") for r in records),
        "avg_latency_s": round(sum(r["latency_s"] for r in graded) / len(graded), 2) if graded else 0.0,
        "total_tokens": total_tokens,
        "avg_tokens_per_case": round(total_tokens / len(records)) if records else 0,
        "max_tokens_case": max(records, key=lambda r: r["prompt_tokens"] + r["completion_tokens"])["id"] if records else None,
        "by_category": dict(sorted(by_cat.items())),
    }


def print_report(records, summary):
    print("\n== Cases ==")
    for r in records:
        mark = {"pass": "PASS", "fail": "FAIL", "error": "ERR "}[r["status"]]
        tokens = r["prompt_tokens"] + r["completion_tokens"]
        print(f"  [{mark}] {r['id']:34s} {r['latency_s']:5.1f}s {tokens:6d} tok  attempts={r['attempts']}")
        for reason in r["reasons"]:
            print(f"         - {reason}")
        if r["status"] != "pass":
            print(f"         answer: {r['answer'][:220]!r}")

    print("\n== By category ==")
    for cat, v in summary["by_category"].items():
        print(f"  {cat:16s} {v['pass']}/{v['total']}")

    print("\n== Totals ==")
    print(f"  pass rate          {summary['pass_rate']:.0%} ({summary['passed']}/{summary['graded']} graded)")
    print(f"  errors (provider)  {summary['errors']}")
    print(f"  false contradiction flags  {summary['contradiction_flags']}")
    print(f"  avg latency        {summary['avg_latency_s']}s")
    print(f"  tokens             {summary['total_tokens']} total, {summary['avg_tokens_per_case']} avg/case "
          f"(heaviest: {summary['max_tokens_case']})")


def compare_to_baseline(records, summary, baseline_path: Path):
    if not baseline_path.exists():
        return
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    before = {r["id"]: r for r in baseline["records"]}
    now = {r["id"]: r for r in records}
    regressions = [i for i in now if i in before and before[i]["passed"] and now[i]["status"] == "fail"]
    fixed = [i for i in now if i in before and not before[i]["passed"] and now[i]["passed"]]
    b = baseline["summary"]
    print(f"\n== Versus baseline ({baseline.get('created', '?')}) ==")
    print(f"  pass rate   {b['pass_rate']:.0%} -> {summary['pass_rate']:.0%}")
    print(f"  avg tokens  {b['avg_tokens_per_case']} -> {summary['avg_tokens_per_case']}")
    print(f"  avg latency {b['avg_latency_s']}s -> {summary['avg_latency_s']}s")
    print(f"  newly failing: {regressions or 'none'}")
    print(f"  newly passing: {fixed or 'none'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end answer-quality eval.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--store", choices=("fixtures", "db"), default="fixtures",
                        help="fixtures: eval/fixtures/*.txt. db: documents loaded in Postgres (read-only).")
    parser.add_argument("--only", action="append", help="Run only this category (repeatable).")
    parser.add_argument("--case", help="Run a single case by id.")
    parser.add_argument("--pace", type=float, default=4.0,
                        help="Seconds to wait between cases, to stay under the provider's rate limit.")
    parser.add_argument("--min-pass", type=float, default=MIN_PASS_RATE)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--save-baseline", action="store_true", help="Save this run as the baseline.")
    args = parser.parse_args()

    cases = load_cases(args.cases, args.only, args.case)
    if not cases:
        print("No cases selected.")
        return 1

    meter = TokenMeter()
    meter.install()
    store = build_store(args.store)
    print(f"Store: {store.total_chunks()} chunks across {len(store.files)} file(s) [{args.store}] | {len(cases)} case(s)")

    records = []
    for i, case in enumerate(cases, start=1):
        print(f"  ({i}/{len(cases)}) {case['id']}", flush=True)
        records.append(run_case(case, store, meter))
        if i < len(cases):
            time.sleep(args.pace)

    summary = summarize(records)
    print_report(records, summary)
    compare_to_baseline(records, summary, args.baseline)

    payload = {"created": datetime.now().isoformat(timespec="seconds"),
               "cases_file": str(args.cases), "store": args.store,
               "summary": summary, "records": records}
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"answer_eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults written to {out}")
    if args.save_baseline:
        args.baseline.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Baseline saved to {args.baseline}")

    ok = summary["pass_rate"] >= args.min_pass and summary["errors"] == 0
    print(f"\n{'PASS' if ok else 'FAIL'} - minimum pass rate {args.min_pass:.0%}, no provider errors")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
