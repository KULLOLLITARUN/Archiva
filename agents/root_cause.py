"""
agents/root_cause.py — Maps failure types to healing actions.

Part 1.3: Given an AgentState with a failure_type set by reflection,
returns the string action token that healer.py will execute.

Mapping:
  RETRIEVAL_FAILURE     → REWRITE_QUERY   (try different keywords)
  INSUFFICIENT_CONTEXT  → INCREASE_TOP_K  (retrieve more chunks)
  HALLUCINATION         → STRICT_PROMPT   (constrain the model harder)
  OUTDATED_DATA         → REINGEST        (queue re-indexing)
  FORMAT_ERROR          → STRICT_PROMPT   (model was verbose; constrain it)
  UNKNOWN               → NONE            (no healing possible)
"""

from agents.state import AgentState

# ── Action token constants ────────────────────────────────────────────────────

ACTION_REWRITE_QUERY  = "REWRITE_QUERY"
ACTION_INCREASE_TOP_K = "INCREASE_TOP_K"
ACTION_STRICT_PROMPT  = "STRICT_PROMPT"
ACTION_REINGEST       = "REINGEST"
ACTION_NONE           = "NONE"

_FAILURE_TO_ACTION: dict = {
    "RETRIEVAL_FAILURE":    ACTION_REWRITE_QUERY,
    "INSUFFICIENT_CONTEXT": ACTION_INCREASE_TOP_K,
    "HALLUCINATION":        ACTION_STRICT_PROMPT,
    "OUTDATED_DATA":        ACTION_REINGEST,
    "FORMAT_ERROR":         ACTION_STRICT_PROMPT,
}


def analyze_failure(state: AgentState) -> str:
    """
    Inspect state.failure_type and return the recommended healing action string.

    Returns one of:
        REWRITE_QUERY | INCREASE_TOP_K | STRICT_PROMPT | REINGEST | NONE
    """
    if not state.failure_type:
        return ACTION_NONE

    action = _FAILURE_TO_ACTION.get(state.failure_type, ACTION_NONE)
    print(
        f"  [TRACE]  [root_cause] failure_type={state.failure_type!r} "
        f"-> action={action!r} (attempt {state.attempt})"
    )
    return action
