from typing import Optional

from chatbot.memory import ConversationMemory
from config import FOLLOWUP_SIGNALS


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
    If the query is a follow-up and there is prior context, append
    the last query as a reference anchor.
    """
    if is_followup(query):
        last = memory.get_last(session_id)
        if last:
            return f"{query} (referring to: {last.query})"
    return query
