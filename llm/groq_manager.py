"""
llm/groq_manager.py — Thread-safe Groq multi-key round-robin manager.

Part 7 of the production upgrade: distributes LLM load across multiple
Groq API keys, preventing single-key exhaustion and rate-limit failures.

Features:
  - Round-robin selection across all configured keys.
  - Per-key failure tracking: failed keys are blacklisted for BACKOFF_S seconds.
  - Automatic recovery: blacklisted keys re-enter rotation after the backoff.
  - Safe for multi-threaded use (threading.Lock around index mutation).
  - Returns a ready-to-use Groq client, not just a key string.
"""

import re
import threading
import time
from typing import List, Optional

from groq import Groq

from config import GROQ_API_KEYS, GROQ_REQUEST_TIMEOUT_S

# ── Constants ─────────────────────────────────────────────────────────────────

_BACKOFF_S: float = 60.0       # seconds to blacklist a failed key
_MIN_KEYS: int    = 1          # must have at least one key


# ── Small-output calls on reasoning models ───────────────────────────────────

# Reasoning models (e.g. openai/gpt-oss-*) "think" before answering, and that
# thinking is paid for out of max_tokens. The helpers that ask for a tiny reply
# (a YES/NO verdict, one search query, a short JSON list) used budgets of
# 20-300 tokens, which the thinking alone exhausts: the API then returns
# finish_reason="length" with EMPTY content. Callers fell back silently
# (no decomposition, no query rewrite) or, worse, read the empty string as a
# NO verdict. Low reasoning effort plus a budget with room for that thinking
# fixes it; the visible reply stays as short as the prompt demands.
_REASONING_MIN_TOKENS: int = 600


def light_completion_params(model: str, max_tokens: int) -> dict:
    """
    Request kwargs for a short-output chat completion on *model*.

    Usage: client.chat.completions.create(model=m, messages=..., **light_completion_params(m, 100))
    Non-reasoning models get their max_tokens back unchanged.
    """
    if "gpt-oss" in model.lower():
        return {
            "max_tokens": max(max_tokens, _REASONING_MIN_TOKENS),
            "extra_body": {"reasoning_effort": "low"},
        }
    return {"max_tokens": max_tokens}


# ── Rate-limit retry hint ─────────────────────────────────────────────────────

# A per-minute 429 from Groq says exactly when the window frees up:
#   "... on tokens per minute (TPM): Limit 8000, Used 7000, Requested 1500.
#    Please try again in 2.55s. ..."
# Daily limits use the same wording with larger units ("in 7m12.5s", "in 1h2m").
# The retry-after header carries the same wait rounded up to whole seconds, so
# the message is preferred and the header is only a fallback.
_RETRY_HINT_RE = re.compile(r"try again in\s+((?:\d+(?:\.\d+)?(?:ms|h|m|s))+)", re.IGNORECASE)
_RETRY_PART_RE = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_UNIT_SECONDS  = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}


def retry_after_seconds(exc: Exception) -> Optional[float]:
    """
    How long Groq asked us to wait before retrying, in seconds, or None if
    the error carries no hint (outage, malformed body, non-HTTP error).
    """
    match = _RETRY_HINT_RE.search(str(exc))
    if match:
        return sum(
            float(value) * _UNIT_SECONDS[unit]
            for value, unit in _RETRY_PART_RE.findall(match.group(1))
        )

    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        header = headers.get("retry-after")
        if header is not None:
            try:
                return max(0.0, float(header))
            except ValueError:
                pass  # HTTP-date form; Groq doesn't send it, not worth parsing
    return None


# ── Manager ───────────────────────────────────────────────────────────────────

class GroqKeyManager:
    """
    Thread-safe round-robin Groq API key manager.

    Usage:
        manager = GroqKeyManager(["key1", "key2", "key3"])
        client  = manager.get_client()
        # ... use client ...
        # On failure:
        manager.mark_failed("key1")   # blacklist for BACKOFF_S seconds
        manager.mark_failed("key1", backoff_s=3)   # short cooldown from a 429 hint
    """

    def __init__(self, keys: List[str], backoff_s: float = _BACKOFF_S) -> None:
        if not keys:
            raise ValueError("GroqKeyManager requires at least one API key.")
        self._keys: List[str]       = list(keys)
        self._backoff_s: float      = backoff_s
        self._index: int            = 0
        self._failures: dict        = {}  # key → monotonic time it becomes healthy again
        self._lock: threading.Lock  = threading.Lock()

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_key(self) -> str:
        """
        Return the next healthy API key using round-robin.
        If all keys are blacklisted, wait for the first one to recover
        and return it (rather than raising — never crash the request).
        """
        with self._lock:
            now = time.monotonic()
            # First pass: find the next healthy key
            for _ in range(len(self._keys)):
                key = self._keys[self._index]
                self._index = (self._index + 1) % len(self._keys)
                available_at = self._failures.get(key)
                if available_at is None or now >= available_at:
                    # Key is healthy (or recovered from backoff)
                    if key in self._failures:
                        del self._failures[key]
                        print(f"  [groq_manager] Key ...{key[-6:]} recovered from backoff.")
                    return key

            # All keys are blacklisted — return the one whose backoff expires
            # soonest, even if still within backoff.
            soonest_key = min(self._failures, key=lambda k: self._failures[k])
            print(
                f"  [groq_manager] All keys blacklisted -- "
                f"forcing key ...{soonest_key[-6:]} (may still be rate-limited)."
            )
            return soonest_key

    def mark_failed(self, key: str, backoff_s: Optional[float] = None) -> None:
        """
        Blacklist *key* for *backoff_s* seconds (default BACKOFF_S).

        A 429 with a short "try again in 2.55s" hint passes that wait, so the
        key returns to rotation when Groq says it will work again rather than
        sitting out the full default backoff.
        """
        duration = self._backoff_s if backoff_s is None else max(0.0, backoff_s)
        with self._lock:
            self._failures[key] = time.monotonic() + duration
            print(f"  [groq_manager] Key ...{key[-6:]} blacklisted for {duration:g}s.")

    def get_client(self) -> tuple:
        """
        Return ``(api_key, Groq_client)`` for the next healthy key.
        The caller should pass the key to ``mark_failed()`` on error.

        timeout=GROQ_REQUEST_TIMEOUT_S bounds a single HTTP call so a
        stalled connection can't hang indefinitely. max_retries=0 disables
        the SDK's own internal retry-on-error — callers (agents/worker.py's
        call_groq()) already implement retry + key rotation on top of this
        client, so leaving the SDK's default retries on top would silently
        multiply attempts without that logic knowing about it.
        """
        key = self.get_key()
        return key, Groq(api_key=key, timeout=GROQ_REQUEST_TIMEOUT_S, max_retries=0)

    def healthy_count(self) -> int:
        """Return the number of currently healthy (non-blacklisted) keys."""
        now = time.monotonic()
        with self._lock:
            return sum(
                1 for k in self._keys
                if k not in self._failures or now >= self._failures[k]
            )

    def key_count(self) -> int:
        """Total number of configured keys."""
        return len(self._keys)


# ── Module-level singleton ─────────────────────────────────────────────────────

try:
    groq_manager = GroqKeyManager(GROQ_API_KEYS)
except ValueError:
    # No GROQ API keys configured — create a dummy manager that will
    # raise a descriptive error at request time rather than crashing at import.
    import warnings
    warnings.warn(
        "[groq_manager] No GROQ_API_KEY(S) configured in .env. "
        "All LLM calls will fail until a key is provided.",
        stacklevel=1,
    )

    class _NoKeyManager:
        """Placeholder that raises a clear error when called."""
        def get_client(self):
            raise RuntimeError(
                "No Groq API key configured. Set GROQ_API_KEY in your .env file."
            )
        def mark_failed(self, key, backoff_s=None):
            pass
        def healthy_count(self):
            return 0
        def key_count(self):
            return 0
        def get_key(self):
            raise RuntimeError("No Groq API key configured.")

    groq_manager = _NoKeyManager()  # type: ignore[assignment]
