from typing import Optional

from chatbot.memory import ConversationMemory
from config import FOLLOWUP_SIGNALS

# Keeps the rewritten query bounded — a previous "summarize"/"explain"
# answer can run long, and only a preview is needed to resolve the
# follow-up's reference, not the full text.
_PREV_ANSWER_PREVIEW = 300


def is_followup(query: str) -> bool:
    """
    Deterministic follow-up detector (no LLM).
    True if any FOLLOWUP_SIGNAL is in the query OR word count < 6.
    """
    q = query.lower()
    if any(signal in q for signal in FOLLOWUP_SIGNALS):
        return True
    if len(query.split()) < 6:
        return True
    return False


def rewrite(query: str, memory: ConversationMemory, session_id: str) -> str:
    """
    If the query is a follow-up, anchor it to BOTH the previous question
    and the previous ANSWER — not just the previous question.

    Anchoring to the question alone (the original behaviour) is the wrong
    anchor for a follow-up that's actually asking something NEW about the
    prior answer's content — e.g. "what's the main highlight in this?".
    The appended question dominates the rewritten query (it's usually
    longer and more specific than a short follow-up), so retrieval fetches
    the same chunks as before and the LLM just re-answers the previous
    question instead of the new one: the model has no way to know what
    "this" refers to without seeing what it actually said. Same principle
    as agents/decomposer.py's anchor_to_prior_answer() for multi-hop
    sub-questions, applied here across chat turns instead of within one
    decomposed query.
    """
    if is_followup(query):
        last = memory.get_last(session_id)
        if last:
            prev_answer = last.answer[:_PREV_ANSWER_PREVIEW]
            return (
                f'{query} (context: previous question was "{last.query}" '
                f'- your previous answer was "{prev_answer}")'
            )
    return query
