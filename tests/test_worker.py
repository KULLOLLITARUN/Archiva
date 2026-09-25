"""Tests for agents/worker.py — prompt construction and the Groq call's
multi-key rotation / retry behavior."""

import httpx
from groq import RateLimitError, APIStatusError

import agents.worker as worker
from agents.worker import build_prompt, call_groq


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

    def mark_failed(self, key):
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
