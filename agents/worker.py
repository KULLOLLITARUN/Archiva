"""
agents/worker.py — Groq LLM call with multi-key round-robin and streaming.

Upgrade (Part 7):
  - Replaced module-level singleton client with groq_manager.get_client().
  - On RateLimitError / APIStatusError: marks the failing key and retries
    with the next healthy key (up to len(GROQ_API_KEYS) additional attempts).
  - Exponential backoff still applies between retries.
  - Streaming (call_groq_stream) also uses key rotation.

Upgrade (Part 5 / STRICT_PROMPT mode):
  - build_prompt() accepts an optional prompt_mode="strict" argument.
  - Strict mode adds extra grounding constraints to reduce hallucination.

Original fixes retained:
  Fix #2:  3 retries with exponential back-off.
  Fix #8:  call_groq_stream() for SSE streaming.
"""

import time
from typing import Generator, Optional

from groq import Groq
from groq import RateLimitError, APIStatusError

from config import MAX_CONTEXT_TOKENS
from llm.groq_manager import groq_manager

# ── Retry configuration ────────────────────────────────────────────────────────

_MAX_RETRIES    = 3
_BACKOFF_BASE_S = 1  # seconds; doubles each attempt (1 → 2 → 4)


# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_prompt(
    query: str,
    context: str,
    intent: str,
    prompt_mode: str = "normal",
    files_summary: str = "",
) -> str:
    """
    Construct the full system+user prompt string.

    Args:
        query:         The user query.
        context:       Source-labeled, trimmed context from retrieved chunks.
        intent:        Detected intent (qa / explain / summarize / compare / meta).
        prompt_mode:   "normal" (default) or "strict" (anti-hallucination mode,
                       triggered by HALLUCINATION failure type in healer).
        files_summary: Document inventory summary string from store.
    """
    full_context = f"{files_summary}\n\n{context}".strip() if files_summary else context

    if prompt_mode == "strict":
        base = (
            "You are Archiva, a strictly grounded AI document intelligence assistant.\n"
            "CRITICAL RULES — VIOLATION IS NOT ACCEPTABLE:\n"
            "- Answer EXCLUSIVELY from the context provided. Zero exceptions.\n"
            "- Focus strictly on answering the user's query. If the context contains chunks from unrelated documents/topics, IGNORE them completely.\n"
            "- If ANY detail is not explicitly stated in the context → say: "
            "Not found in the document.\n"
            "- DO NOT paraphrase, infer, extrapolate, or use general knowledge.\n"
            "- DO NOT invent numbers, dates, statistics, or names.\n"
            "- Every claim MUST have an inline citation: [Source: filename, page N]\n"
            "- If unsure about ANY word → omit it entirely.\n\n"
            f"Context:\n{full_context}"
        )
    else:
        base = (
            "You are Archiva, an AI document intelligence assistant.\n"
            "STRICT RULES:\n"
            "- Answer ONLY from the context below.\n"
            "- Focus strictly on answering the specific question asked. Do NOT summarize or include unrelated documents or topics found in the context unless explicitly asked to compare.\n"
            "- If not found → say exactly: Not found in the document.\n"
            "- Do NOT infer, guess, or use general knowledge.\n"
            "- Do NOT make up facts or page numbers.\n"
            "- Cite your source: [Source: filename, page N]\n\n"
            f"Context:\n{full_context}"
        )

    intent_additions = {
        "qa":        "Provide a precise, direct answer.",
        "explain":   "Explain in simple language anyone can understand.",
        "summarize": "Summarize key points using bullet points.",
        "compare": (
            "Compare across sources. Label each source clearly. "
            "Show differences and similarities."
        ),
        "meta": (
            "List and count all relevant files accurately based on the system document summary "
            "and provided context."
        ),
    }

    addition = intent_additions.get(intent, intent_additions["qa"])
    return f"{base}\n\n{addition}"


# ── Non-streaming LLM call ─────────────────────────────────────────────────────

def call_groq(model_id: str, prompt: str, query: str) -> str:
    """
    Single LLM call to Groq with multi-key rotation and exponential back-off.

    Attempt order:
      1. Try with the current healthy key.
      2. On RateLimitError / APIStatusError → mark key failed, try next key.
      3. On transient error → exponential back-off, same key.
      4. After _MAX_RETRIES persistent failures → return friendly error string.
    """
    last_exc: Optional[Exception] = None
    # We allow up to _MAX_RETRIES total attempts across any keys.
    for attempt in range(_MAX_RETRIES):
        current_key, client = groq_manager.get_client()
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user",   "content": query},
                ],
                temperature=0.1,
                max_tokens=1024,
            )
            return response.choices[0].message.content

        except RateLimitError as exc:
            # Genuine rate-limit — blacklist this key and rotate to the next one.
            groq_manager.mark_failed(current_key)
            last_exc = exc
            print(
                f"  [WARN]  [worker] Key …{current_key[-6:]} rate-limited "
                f"(attempt {attempt + 1}/{_MAX_RETRIES}): {exc}"
            )
            # No sleep — rotate to next key immediately.

        except APIStatusError as exc:
            # Check if this is a permanent model error (not a key/rate issue).
            # model_not_found / model_decommissioned → no point rotating keys or retrying.
            err_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            body = str(exc)
            is_model_error = (
                "model_not_found" in body
                or "model_decommissioned" in body
                or err_code == 404
            )
            if is_model_error:
                print(f"  [ERR]  [worker] Permanent model error — not retrying: {exc}")
                raise  # Re-raise so callers (e.g. suggestions fallback) can catch it.
            # Any other APIStatusError (server errors etc.) → treat like rate limit.
            groq_manager.mark_failed(current_key)
            last_exc = exc
            print(
                f"  [WARN]  [worker] Key …{current_key[-6:]} API error "
                f"(attempt {attempt + 1}/{_MAX_RETRIES}): {exc}"
            )

        except Exception as exc:
            last_exc = exc
            wait = _BACKOFF_BASE_S * (2 ** attempt)  # 1 s, 2 s, 4 s
            print(
                f"  [WARN]  [worker] Groq call failed (attempt {attempt + 1}/{_MAX_RETRIES}): "
                f"{exc} — retrying in {wait}s"
            )
            time.sleep(wait)

    print(f"  [ERR]  [worker] Groq permanently failed after {_MAX_RETRIES} attempts: {last_exc}")
    return "Service temporarily unavailable. Please try again."


# ── Streaming LLM call ─────────────────────────────────────────────────────────

def call_groq_stream(
    model_id: str,
    prompt: str,
    query: str,
) -> Generator[str, None, None]:
    """
    Streaming Groq call — yields token strings as they arrive.

    Uses key rotation: on failure selects the next healthy key.
    Unlike call_groq(), does NOT retry mid-stream (SSE headers already sent).
    """
    current_key, client = groq_manager.get_client()
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user",   "content": query},
            ],
            temperature=0.1,
            max_tokens=1024,
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    except (RateLimitError, APIStatusError) as exc:
        groq_manager.mark_failed(current_key)
        print(f"  [WARN]  [worker/stream] Key …{current_key[-6:]} rate-limited: {exc}")
        yield f"\n[Stream interrupted: rate limit — please retry]"

    except Exception as exc:
        print(f"  [ERR]  [worker/stream] Stream error: {exc}")
        yield f"\n[Stream error: {exc}]"
