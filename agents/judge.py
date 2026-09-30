"""
agents/judge.py — Lightweight LLM faithfulness judge.

Called as an optional second-pass check in loop.py when the heuristic
reflection passes but confidence is still below MEDIUM_CONFIDENCE_THRESHOLD.

Design principles:
  - Uses GROQ_FAST (small model, cheap) with a strict yes/no prompt.
  - Max 20 output tokens — we only need "YES" or "NO".
  - Returns a simple dict: {"faithful": bool, "reason": str}.
  - Never raises — any failure returns {"faithful": True, ...} so the loop
    proceeds rather than blocking on a judge error.
  - Zero side effects: pure function on (query, answer, chunks).
"""

from typing import Dict, List

from config import GROQ_FAST
from llm.groq_manager import groq_manager, light_completion_params

# ── Configuration ─────────────────────────────────────────────────────────────

_MAX_TOKENS: int     = 20
_TEMPERATURE: float  = 0.0       # fully deterministic
_CONTEXT_CHARS: int  = 800       # max chars of chunk text to include in prompt
_ANSWER_CHARS: int   = 400       # max chars of answer to include

_SYSTEM_PROMPT = (
    "You are a strict factual grounding checker. "
    "Given a CONTEXT extracted from source documents and an ANSWER, "
    "determine if every claim in the ANSWER is directly supported by the CONTEXT.\n"
    "Reply with exactly one word: YES if the answer is fully supported, "
    "NO if any claim lacks grounding.\n"
    "Do not explain. Do not add punctuation. Just YES or NO."
)


# ── Public API ────────────────────────────────────────────────────────────────

def judge_faithfulness(
    query: str,
    answer: str,
    chunks: List[dict],
) -> Dict:
    """
    Ask the LLM to verify that every claim in *answer* is grounded in *chunks*.

    Args:
        query:   The user query (for context).
        answer:  The generated answer to check.
        chunks:  Retrieved/reranked source chunks.

    Returns:
        {
            "faithful": bool,   # True = grounded, False = possible hallucination
            "reason":   str,    # "judge_pass" | "judge_fail" | "judge_error"
        }
    """
    context_text = " ".join(
        c.get("text", "") for c in chunks
    )[:_CONTEXT_CHARS]

    user_message = (
        f"QUESTION: {query}\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        f"ANSWER:\n{answer[:_ANSWER_CHARS]}\n\n"
        "Is the ANSWER fully supported by the CONTEXT? Reply YES or NO:"
    )

    try:
        _, client = groq_manager.get_client()
        response = client.chat.completions.create(
            model=GROQ_FAST,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            temperature=_TEMPERATURE,
            **light_completion_params(GROQ_FAST, _MAX_TOKENS),
        )
        raw: str = (response.choices[0].message.content or "").strip().upper()
        if not raw:
            # No verdict at all (e.g. the model ran out of tokens). That is an
            # error, NOT a "NO": reading "" as unfaithful would condemn every
            # answer the judge looked at.
            print("  [WARN]  [judge] empty verdict -- assuming faithful")
            return {"faithful": True, "reason": "judge_error"}
        faithful = raw.startswith("YES")
        reason   = "judge_pass" if faithful else "judge_fail"
        print(f"  [INFO]  [judge] faithfulness={faithful!r} (raw={raw!r})")
        return {"faithful": faithful, "reason": reason}

    except Exception as exc:
        # Never block the loop on a judge error
        print(f"  [WARN]  [judge] LLM judge failed ({exc}) -- assuming faithful")
        return {"faithful": True, "reason": "judge_error"}
