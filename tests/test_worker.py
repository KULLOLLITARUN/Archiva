"""Tests for agents/worker.py — prompt construction and the Groq call's
multi-key rotation / retry behavior."""

import httpx
import pytest
from groq import RateLimitError, APIStatusError

import agents.worker as worker
import llm.groq_manager as gm
from agents.worker import build_prompt, call_groq
from llm.groq_manager import GroqKeyManager


def _http_response(status_code: int) -> httpx.Response:
    return httpx.Response(status_code=status_code, request=httpx.Request("POST", "https://api.groq.com/x"))


def _rate_limit_error() -> RateLimitError:
    return RateLimitError("rate limited", response=_http_response(429), body=None)


def _model_not_found_error() -> APIStatusError:
    return APIStatusError(
        "model_not_found", response=_http_response(404),
        body={"error": {"code": "model_not_found"}},
    )


# ── build_prompt (pure) ─────────────────────────────────────────────────────────

def test_build_prompt_normal_mode_contains_context_and_query_rules():
    prompt = build_prompt("what is X", "Context text here", "qa")
    assert "Context text here" in prompt
    assert "Not found in the document" in prompt
    assert "Provide a precise, direct answer." in prompt


def test_build_prompt_strict_mode_has_stronger_grounding_language():
    prompt = build_prompt("what is X", "ctx", "qa", prompt_mode="strict")
    assert "VIOLATION IS NOT ACCEPTABLE" in prompt
    assert "DO NOT invent numbers" in prompt


def test_build_prompt_intent_specific_addition_per_intent():
    assert "bullet points" in build_prompt("q", "c", "summarize")
    assert "Compare across sources" in build_prompt("q", "c", "compare")
    assert "simple language" in build_prompt("q", "c", "explain")


def test_build_prompt_unknown_intent_falls_back_to_qa_addition():
    assert "Provide a precise, direct answer." in build_prompt("q", "c", "some_unknown_intent")


def test_build_prompt_includes_files_summary_when_provided():
    prompt = build_prompt("q", "ctx", "qa", files_summary="3 files loaded: a.pdf, b.pdf, c.pdf")
    assert "3 files loaded" in prompt
    assert prompt.index("3 files loaded") < prompt.index("ctx")


# ── call_groq key rotation / retry ────────────────────────────────────────────

class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, plan):
        # plan: list of either a string (success) or an Exception instance to raise
        self._plan = list(plan)

    def create(self, **kwargs):
        item = self._plan.pop(0)
        if isinstance(item, Exception):
            raise item
        return _FakeResponse(item)


class _FakeClient:
    def __init__(self, plan):
        self.chat = type("Chat", (), {})()
        self.chat.completions = _FakeCompletions(plan)


class _FakeManager:
    """Cycles through provided keys; records mark_failed() calls."""

    def __init__(self, keys, clients_by_key):
        self._keys = keys
        self._clients = clients_by_key
        self._i = 0
        self.failed = []

    def get_client(self):
        key = self._keys[self._i % len(self._keys)]
        self._i += 1
        return key, self._clients[key]

    def mark_failed(self, key, backoff_s=None):
        self.failed.append(key)


def test_call_groq_returns_content_on_success(monkeypatch):
    client = _FakeClient(plan=["the answer"])
    manager = _FakeManager(["key1"], {"key1": client})
    monkeypatch.setattr(worker, "groq_manager", manager)

    result = call_groq("model-x", "system prompt", "user query")
    assert result == "the answer"


def test_call_groq_rotates_key_on_rate_limit_and_succeeds(monkeypatch):
    exc = _rate_limit_error()
    client1 = _FakeClient(plan=[exc])
    client2 = _FakeClient(plan=["recovered answer"])
    manager = _FakeManager(["key1", "key2"], {"key1": client1, "key2": client2})
    monkeypatch.setattr(worker, "groq_manager", manager)

    result = call_groq("model-x", "system prompt", "user query")
    assert result == "recovered answer"
    assert manager.failed == ["key1"]


def test_call_groq_returns_friendly_error_after_max_retries(monkeypatch):
    exc = _rate_limit_error()
    client = _FakeClient(plan=[exc, exc, exc])
    manager = _FakeManager(["key1"], {"key1": client})
    monkeypatch.setattr(worker, "groq_manager", manager)

    result = call_groq("model-x", "system prompt", "user query")
    assert result == "Service temporarily unavailable. Please try again."
    assert manager.failed == ["key1", "key1", "key1"]


def test_call_groq_reraises_on_permanent_model_error(monkeypatch):
    exc = _model_not_found_error()
    client = _FakeClient(plan=[exc])
    manager = _FakeManager(["key1"], {"key1": client})
    monkeypatch.setattr(worker, "groq_manager", manager)

    try:
        call_groq("bad-model", "system prompt", "user query")
        assert False, "expected APIStatusError to propagate"
    except APIStatusError:
        pass
    # Permanent model errors should NOT blacklist the key — it's a model
    # problem, not a key/rate-limit problem.
    assert manager.failed == []


# ── call_groq honours Groq's "try again in Xs" hint ───────────────────────────

def _hinted_rate_limit_error(wait: str) -> RateLimitError:
    message = (
        "Error code: 429 - {'error': {'message': 'Rate limit reached for model `m` "
        "on tokens per minute (TPM): Limit 8000, Used 7000, Requested 1500. "
        "Please try again in " + wait + ". Visit https://console.groq.com/docs/rate-limits'}}"
    )
    return RateLimitError(message, response=_http_response(429), body=None)


class _RealManagerWithFakeClients(GroqKeyManager):
    """Real key selection and cooldown logic; only the HTTP client is faked."""

    def __init__(self, clients_by_key, backoff_s=60):
        super().__init__(list(clients_by_key), backoff_s=backoff_s)
        self._clients = clients_by_key
        self.keys_used = []

    def get_client(self):
        key = self.get_key()
        self.keys_used.append(key)
        return key, self._clients[key]


class _Clock:
    """Fake monotonic clock that the worker's time.sleep() advances instead of blocking."""

    def __init__(self, early_s=0.0):
        self.now = 1000.0
        self.sleeps = []
        self._early_s = early_s

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds - self._early_s


def _install(monkeypatch, manager, early_s=0.0):
    clock = _Clock(early_s)
    monkeypatch.setattr(gm.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(worker.time, "sleep", clock.sleep)
    monkeypatch.setattr(worker, "groq_manager", manager)
    return clock


def test_call_groq_waits_short_hint_and_retries_same_key(monkeypatch):
    client = _FakeClient(plan=[_hinted_rate_limit_error("2.55s"), "answer after wait"])
    manager = _RealManagerWithFakeClients({"only-key": client})
    clock = _install(monkeypatch, manager)

    assert call_groq("model-x", "system prompt", "user query") == "answer after wait"
    assert manager.keys_used == ["only-key", "only-key"]
    assert clock.sleeps == [pytest.approx(2.55 + worker._HINT_MARGIN_S)]
    # The key came back as soon as the wait was over, not after the 60s backoff.
    assert manager.healthy_count() == 1


def test_call_groq_short_hint_key_is_healthy_even_if_sleep_wakes_early(monkeypatch, capsys):
    # Windows time.sleep() can return a few ms before time.monotonic() has
    # moved on that far; the key must still be healthy on waking, not handed
    # back through the "all keys blacklisted -- forcing" fallback.
    client = _FakeClient(plan=[_hinted_rate_limit_error("6.57s"), "ok"])
    manager = _RealManagerWithFakeClients({"only-key": client})
    _install(monkeypatch, manager, early_s=0.005)

    assert call_groq("model-x", "system prompt", "user query") == "ok"
    out = capsys.readouterr().out
    assert "recovered from backoff" in out
    assert "forcing" not in out


def test_call_groq_handles_millisecond_hint(monkeypatch):
    client = _FakeClient(plan=[_hinted_rate_limit_error("520ms"), "ok"])
    manager = _RealManagerWithFakeClients({"only-key": client})
    clock = _install(monkeypatch, manager)

    assert call_groq("model-x", "system prompt", "user query") == "ok"
    assert clock.sleeps == [pytest.approx(0.52 + worker._HINT_MARGIN_S)]


def test_call_groq_rotates_without_waiting_when_another_key_is_healthy(monkeypatch):
    client_a = _FakeClient(plan=[_hinted_rate_limit_error("2.55s")])
    client_b = _FakeClient(plan=["answer from b"])
    manager = _RealManagerWithFakeClients({"key-a": client_a, "key-b": client_b})
    clock = _install(monkeypatch, manager)

    assert call_groq("model-x", "system prompt", "user query") == "answer from b"
    assert manager.keys_used == ["key-a", "key-b"]
    assert clock.sleeps == []


def test_call_groq_does_not_wait_out_long_hint(monkeypatch):
    # A daily-limit style hint is beyond the cap: keep the old behaviour of
    # blacklisting for the full backoff and giving up with the friendly
    # answer, with no long sleep.
    exc = _hinted_rate_limit_error("7m12s")
    client = _FakeClient(plan=[exc, exc, exc])
    manager = _RealManagerWithFakeClients({"only-key": client})
    clock = _install(monkeypatch, manager)

    assert call_groq("model-x", "system prompt", "user query") == worker.SERVICE_UNAVAILABLE_ANSWER
    assert clock.sleeps == []
    assert manager.healthy_count() == 0


def test_call_groq_does_not_sleep_after_final_attempt(monkeypatch):
    exc = _hinted_rate_limit_error("2s")
    client = _FakeClient(plan=[exc, exc, exc])
    manager = _RealManagerWithFakeClients({"only-key": client})
    clock = _install(monkeypatch, manager)

    assert call_groq("model-x", "system prompt", "user query") == worker.SERVICE_UNAVAILABLE_ANSWER
    # Waits between attempts 1→2 and 2→3 only; nothing left to retry after the third.
    assert len(clock.sleeps) == worker._MAX_RETRIES - 1
