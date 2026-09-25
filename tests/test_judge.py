"""Tests for agents/judge.py — optional LLM faithfulness judge.
Called only as a second-pass check on borderline-confidence answers;
must never raise, and must fail open (faithful=True) on any error."""

import agents.judge as judge
from agents.judge import judge_faithfulness


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
        judge.groq_manager, "get_client",
        lambda: ("fake-key", _FakeClient(content, exc)),
    )


def test_judge_faithfulness_true_on_yes(monkeypatch):
    _mock_llm(monkeypatch, content="YES")
    result = judge_faithfulness("q", "the notice period is 30 days", [{"text": "30 days notice"}])
    assert result == {"faithful": True, "reason": "judge_pass"}


def test_judge_faithfulness_false_on_no(monkeypatch):
    _mock_llm(monkeypatch, content="NO")
    result = judge_faithfulness("q", "some ungrounded claim", [{"text": "unrelated text"}])
    assert result == {"faithful": False, "reason": "judge_fail"}


def test_judge_faithfulness_tolerates_extra_whitespace_and_case(monkeypatch):
    _mock_llm(monkeypatch, content="  yes  ")
    result = judge_faithfulness("q", "a", [{"text": "b"}])
    assert result["faithful"] is True


def test_judge_faithfulness_fails_open_on_llm_error(monkeypatch):
    _mock_llm(monkeypatch, exc=RuntimeError("groq down"))
    result = judge_faithfulness("q", "a", [{"text": "b"}])
    assert result == {"faithful": True, "reason": "judge_error"}


def test_judge_faithfulness_handles_missing_chunk_text_key(monkeypatch):
    # chunks without a "text" key should not raise — .get("text", "") covers it
    _mock_llm(monkeypatch, content="YES")
    result = judge_faithfulness("q", "a", [{"metadata": {}}])
    assert result == {"faithful": True, "reason": "judge_pass"}
