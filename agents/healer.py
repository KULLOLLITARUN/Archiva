"""
agents/healer.py — Applies healing actions to AgentState based on root cause.

Part 1.3: Mutates state in-place so the next loop iteration picks up the fix.

Actions:
  REWRITE_QUERY   — rewrites state.rewritten_query via the LLM query rewriter
  INCREASE_TOP_K  — raises state.top_k (capped at 20) to fetch more context
  STRICT_PROMPT   — switches state.prompt_mode to "strict" (hallucinaton guard)
  REINGEST        — appends the original query to the reingestion queue; no
                    immediate fix (data is stale), but logs it for operators
  NONE            — no-op
"""

import json
import os
from datetime import datetime, timezone

from agents.state import AgentState
from agents.query_rewriter import rewrite_for_retry

# ── Constants ─────────────────────────────────────────────────────────────────

_TOP_K_STEP: int = 5    # how much to increase top_k each INCREASE_TOP_K action
_TOP_K_MAX:  int = 20   # never retrieve more than this many chunks

# Lazily imported from config to avoid circular at module load
def _reingestion_queue_path() -> str:
    from config import REINGESTION_QUEUE_PATH
    return REINGESTION_QUEUE_PATH


# ── Healing actions ────────────────────────────────────────────────────────────

def _heal_rewrite_query(state: AgentState) -> None:
    """Ask the LLM to rephrase the query to improve BM25/dense retrieval."""
    new_query = rewrite_for_retry(
        original_query    = state.original_query,
        failed_answer     = state.answer or "",
        reflection_reason = state.failure_reason or "low_quality",
        attempt           = state.attempt,
    )
    state.rewritten_query = new_query
    state.search_queries.append(new_query)
    print(f"  💊  [healer] REWRITE_QUERY → {new_query!r}")


def _heal_increase_top_k(state: AgentState) -> None:
    """Fetch more candidate chunks on the next retrieval pass."""
    old = state.top_k
    state.top_k = min(state.top_k + _TOP_K_STEP, _TOP_K_MAX)
    print(f"  💊  [healer] INCREASE_TOP_K {old} → {state.top_k}")


def _heal_strict_prompt(state: AgentState) -> None:
    """Switch to strict prompt mode to reduce hallucination or verbosity."""
    state.prompt_mode = "strict"
    print("  💊  [healer] STRICT_PROMPT enabled")


def _heal_reingest(state: AgentState) -> None:
    """
    Queue the original query for operator-driven reingestion review.
    Does NOT immediately fix: it logs the signal for future action.
    """
    queue_path = _reingestion_queue_path()
    os.makedirs(os.path.dirname(queue_path) or ".", exist_ok=True)

    record = {
        "query":   state.original_query,
        "reason":  state.failure_reason or "OUTDATED_DATA",
        "queued_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        with open(queue_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(f"  💊  [healer] REINGEST queued: {state.original_query!r}")
    except Exception as exc:
        print(f"  [WARN]  [healer] REINGEST queue write failed: {exc}")


# ── Dispatch table ─────────────────────────────────────────────────────────────

_HEALERS = {
    "REWRITE_QUERY":   _heal_rewrite_query,
    "INCREASE_TOP_K":  _heal_increase_top_k,
    "STRICT_PROMPT":   _heal_strict_prompt,
    "REINGEST":        _heal_reingest,
}


# ── Public entry point ────────────────────────────────────────────────────────

def apply_healing(state: AgentState, action: str) -> None:
    """
    Apply the healing action returned by analyze_failure() to state in-place.

    Args:
        state:  The AgentState to mutate.
        action: One of REWRITE_QUERY | INCREASE_TOP_K | STRICT_PROMPT |
                REINGEST | NONE.
    """
    healer_fn = _HEALERS.get(action)
    if healer_fn is None:
        print(f"  💊  [healer] action={action!r} — no-op")
        return
    healer_fn(state)
