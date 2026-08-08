from chatbot.memory import ConversationMemory
from config import FOLLOWUP_SIGNALS

# Keeps the rewritten query bounded — a previous "summarize"/"explain"
# answer can run long, and only a preview is needed to resolve the
# follow-up's reference, not the full text.
_PREV_ANSWER_PREVIEW = 300


def is_followup(query: str) -> bool:
    """
    Deterministic follow-up detector (no LLM).
    True if any FOLLOWUP_SIGNAL is in the query.

    Deliberately NOT based on word count. A blanket "< 6 words → treat as
    a follow-up" fallback used to be here, but short queries are routinely
    self-contained new questions on their own topic - "what need to be in
    resume" (5 words) and "give summary" (2 words) are complete, unambiguous
    requests, not references to whatever the previous answer happened to be
    about. Anchoring them to unrelated prior context drags retrieval toward
    the wrong chunks and can turn a perfectly answerable question into a
    false "not found in the document" (found live: a DOB/age-omission
    answer got dragged into the anchor for "what need to be in resume",
    and that failure then cascaded into "give summary" too, since each
    follow-up anchors to the previous turn's answer - including a bad one).
    Same principle agents/decomposer.py's anchor_to_prior_answer() already
    follows for sub-questions: only anchor on an actual reference signal,
    never on length alone.
    """
    q = query.lower()
    return any(signal in q for signal in FOLLOWUP_SIGNALS)


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
