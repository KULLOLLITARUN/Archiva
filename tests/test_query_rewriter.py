"""Tests for agents/query_rewriter.py — LLM-assisted BM25 retry rewriter.
Called only when reflection decides retry_search; must never crash the loop."""

import agents.query_rewriter as query_rewriter
from agents.query_rewriter import rewrite_for_retry


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
    def __init__(self, content=None, exc=None):
        self._content = content
        self._exc = exc

    def create(self, **kwargs):
        if self._exc:
            raise self._exc
        return _FakeResponse(self._content)


class _FakeChat:
    def __init__(self, content=None, exc=None):
        self.completions = _FakeCompletions(content, exc)


class _FakeClient:
    def __init__(self, content=None, exc=None):
        self.chat = _FakeChat(content, exc)


def _mock_llm(monkeypatch, content=None, exc=None):
    monkeypatch.setattr(
        query_rewriter.groq_manager, "get_client",
        lambda: ("fake-key", _FakeClient(content, exc)),
    )


def test_rewrite_for_retry_returns_llm_query(monkeypatch):
    _mock_llm(monkeypatch, content="  termination notice period keywords  ")
    result = rewrite_for_retry(
        original_query="how do I end my contract",
        failed_answer="Not found in the document.",
        reflection_reason="low_overlap",
        attempt=1,
    )
    assert result == "termination notice period keywords"


def test_rewrite_for_retry_truncates_to_max_chars(monkeypatch):
    _mock_llm(monkeypatch, content="x" * 500)
    result = rewrite_for_retry("q", "a", "low_overlap", 0)
    assert len(result) == 200


def test_rewrite_for_retry_falls_back_to_original_on_empty_response(monkeypatch):
    _mock_llm(monkeypatch, content="   ")
    result = rewrite_for_retry("original question", "a", "low_overlap", 0)
    assert result == "original question"


def test_rewrite_for_retry_falls_back_to_original_on_llm_error(monkeypatch):
    _mock_llm(monkeypatch, exc=RuntimeError("groq down"))
    result = rewrite_for_retry("original question", "a", "low_overlap", 0)
    assert result == "original question"
