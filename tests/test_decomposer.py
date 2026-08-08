"""Tests for agents/decomposer.py — the pre-filter and LLM-assisted split
behind multi-hop query decomposition."""

import json

import agents.decomposer as decomposer
from agents.decomposer import decompose_query, should_decompose


# ── should_decompose (pure, no LLM) ────────────────────────────────────────────

def test_short_query_is_never_decomposed():
    assert should_decompose("what is the vacation policy") is False


def test_long_query_with_double_question_mark_is_flagged():
    query = "What is the vacation policy for full time staff? How many sick days do we get?"
    assert should_decompose(query) is True


def test_long_query_with_connector_is_flagged():
    query = "What is the vacation policy for full time staff and how fast must oncall respond to alerts"
    assert should_decompose(query) is True


def test_long_single_topic_query_without_connector_is_not_flagged():
    query = "Explain in detail how the parental leave policy changed for primary caregivers this year"
    assert should_decompose(query) is False


# ── decompose_query (LLM mocked) ───────────────────────────────────────────────

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
    def __init__(self, content):
        self._content = content

    def create(self, **kwargs):
        return _FakeResponse(self._content)


class _FakeChat:
    def __init__(self, content):
        self.completions = _FakeCompletions(content)


class _FakeClient:
    def __init__(self, content):
        self.chat = _FakeChat(content)


def _mock_llm_response(monkeypatch, content: str):
    fake_client = _FakeClient(content)
    monkeypatch.setattr(decomposer.groq_manager, "get_client", lambda: ("fake-key", fake_client))


def test_decompose_query_splits_into_listed_subquestions(monkeypatch):
    payload = json.dumps([
        "What is the vacation policy?",
        "How fast must oncall respond to alerts?",
    ])
    _mock_llm_response(monkeypatch, payload)

    result = decompose_query("What is the vacation policy and how fast must oncall respond to alerts?")
    assert result == [
        "What is the vacation policy?",
        "How fast must oncall respond to alerts?",
    ]


def test_decompose_query_single_item_response_means_no_split(monkeypatch):
    _mock_llm_response(monkeypatch, json.dumps(["Compare the vacation policy across regions"]))

    result = decompose_query("Compare the vacation policy across regions")
    assert result == ["Compare the vacation policy across regions"]


def test_decompose_query_strips_markdown_fences(monkeypatch):
    payload = "```json\n" + json.dumps(["a?", "b?"]) + "\n```"
    _mock_llm_response(monkeypatch, payload)

    assert decompose_query("a and b") == ["a?", "b?"]


def test_decompose_query_caps_at_max_subquestions(monkeypatch):
    payload = json.dumps([f"q{i}?" for i in range(10)])
    _mock_llm_response(monkeypatch, payload)

    result = decompose_query("a lot of questions")
    assert len(result) == 4


def test_decompose_query_falls_back_to_original_on_unparseable_response(monkeypatch):
    _mock_llm_response(monkeypatch, "not json at all")
    assert decompose_query("original query") == ["original query"]


def test_decompose_query_falls_back_to_original_on_llm_error(monkeypatch):
    def _raise():
        raise RuntimeError("no keys configured")
    monkeypatch.setattr(decomposer.groq_manager, "get_client", _raise)

    assert decompose_query("original query") == ["original query"]
