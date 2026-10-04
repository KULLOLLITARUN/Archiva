"""Tests for llm/groq_manager.py — the 429 retry-hint parser and the key
manager's per-key cooldown."""

import httpx
import pytest
from groq import RateLimitError

import llm.groq_manager as gm
from llm.groq_manager import GroqKeyManager, retry_after_seconds


def _rate_limit_error(message: str, headers: dict = None) -> RateLimitError:
    response = httpx.Response(
        status_code=429, headers=headers or {},
        request=httpx.Request("POST", "https://api.groq.com/x"),
    )
    return RateLimitError(message, response=response, body=None)


def _groq_429_message(wait: str) -> str:
    # Shape of a real Groq per-minute limit error, as str(exc) renders it.
    return (
        "Error code: 429 - {'error': {'message': 'Rate limit reached for model "
        "`llama-3.1-8b-instant` in organization `org_x` service tier `on_demand` "
        "on tokens per minute (TPM): Limit 8000, Used 7000, Requested 1500. "
        f"Please try again in {wait}. Visit https://console.groq.com/docs/rate-limits "
        "for more information.', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}"
    )


# ── retry_after_seconds ───────────────────────────────────────────────────────

@pytest.mark.parametrize("wait, expected", [
    ("2.55s",   2.55),
    ("520ms",   0.52),
    ("1m30.5s", 90.5),
    ("7m12s",   432.0),
    ("1h2m3s",  3723.0),
])
def test_retry_after_parses_groq_message_hint(wait, expected):
    assert retry_after_seconds(_rate_limit_error(_groq_429_message(wait))) == pytest.approx(expected)


def test_retry_after_prefers_message_over_rounded_header():
    exc = _rate_limit_error(_groq_429_message("2.55s"), headers={"retry-after": "3"})
    assert retry_after_seconds(exc) == pytest.approx(2.55)


def test_retry_after_falls_back_to_header():
    exc = _rate_limit_error("rate limited", headers={"retry-after": "4"})
    assert retry_after_seconds(exc) == 4.0


def test_retry_after_ignores_unparseable_header():
    exc = _rate_limit_error("rate limited", headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"})
    assert retry_after_seconds(exc) is None


def test_retry_after_none_without_hint():
    assert retry_after_seconds(_rate_limit_error("rate limited")) is None
    assert retry_after_seconds(RuntimeError("connection reset")) is None


# ── GroqKeyManager cooldowns ──────────────────────────────────────────────────

class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr(gm.time, "monotonic", c)
    return c


def test_default_mark_failed_blacklists_for_full_backoff(clock):
    manager = GroqKeyManager(["key-a", "key-b"], backoff_s=60)
    manager.mark_failed("key-a")
    clock.now += 59
    assert manager.healthy_count() == 1
    clock.now += 1
    assert manager.healthy_count() == 2


def test_short_cooldown_returns_key_after_hint(clock):
    manager = GroqKeyManager(["key-a"], backoff_s=60)
    manager.mark_failed("key-a", backoff_s=3)
    assert manager.healthy_count() == 0
    clock.now += 3
    assert manager.healthy_count() == 1
    assert manager.get_key() == "key-a"


def test_short_cooldown_key_skipped_while_another_key_is_healthy(clock):
    manager = GroqKeyManager(["key-a", "key-b"], backoff_s=60)
    manager.mark_failed("key-a", backoff_s=3)
    assert [manager.get_key() for _ in range(3)] == ["key-b", "key-b", "key-b"]


def test_all_blacklisted_forces_key_that_recovers_soonest(clock):
    manager = GroqKeyManager(["key-a", "key-b"], backoff_s=60)
    manager.mark_failed("key-a")                 # back at +60s
    manager.mark_failed("key-b", backoff_s=5)    # back at +5s
    assert manager.get_key() == "key-b"
