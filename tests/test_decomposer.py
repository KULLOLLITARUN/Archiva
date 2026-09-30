"""Tests for agents/decomposer.py — the pre-filter and LLM-assisted split
behind multi-hop query decomposition."""

import json

import agents.decomposer as decomposer
from agents.decomposer import (
    anchor_to_prior_answer,
    decompose_query,
    dependencies_of,
    display_query,
    resolve_dependent_query,
    sanitize_plan,
    should_decompose,
)


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


# ── anchor_to_prior_answer (sequential/dependent decomposition) ────────────────

def test_anchor_appends_prior_answer_when_reference_word_present():
    result = anchor_to_prior_answer(
        "What is their vacation policy?",
        "The roadmap team is led by Priya Shah.",
    )
    assert result == (
        "What is their vacation policy? "
        "(referring to: The roadmap team is led by Priya Shah.)"
    )


def test_anchor_is_a_noop_without_a_reference_word():
    # No pronoun at all - genuinely independent sub-question, must not be
    # anchored just because a prior answer happens to exist.
    result = anchor_to_prior_answer("What is planned for Q3 2026?", "Some prior answer.")
    assert result == "What is planned for Q3 2026?"


def test_anchor_is_a_noop_on_the_first_subquestion_with_no_prior_answer():
    result = anchor_to_prior_answer("What is their vacation policy?", "")
    assert result == "What is their vacation policy?"


def test_anchor_does_not_false_trigger_on_this_or_that():
    # "that"/"this"/"these" are deliberately excluded - too common as
    # relative-clause/demonstrative words to be a reliable back-reference
    # signal ("the team THAT manages the roadmap").
    result = anchor_to_prior_answer(
        "What is the policy for the team that manages the roadmap?",
        "Some prior answer.",
    )
    assert result == "What is the policy for the team that manages the roadmap?"


def test_anchor_matches_whole_words_only():
    # "it" must not match inside "wait" / "quite" / etc.
    result = anchor_to_prior_answer("Please wait quietly at the site.", "Some prior answer.")
    assert result == "Please wait quietly at the site."


def test_anchor_catches_they_them_its_variants():
    prior = "Priya Shah leads the team."
    assert "(referring to:" in anchor_to_prior_answer("Where do they work?", prior)
    assert "(referring to:" in anchor_to_prior_answer("How can I contact them?", prior)
    assert "(referring to:" in anchor_to_prior_answer("What is its main goal?", prior)
    assert "(referring to:" in anchor_to_prior_answer("What is their budget?", prior)


# ── Dependent chains ({N} placeholders) ────────────────────────────────────────

def test_then_connector_flags_a_sequential_query():
    query = "Find the vendor for Project Atlas then tell me when that vendor contract renews"
    assert should_decompose(query) is True


def test_dependencies_of_extracts_step_numbers():
    assert dependencies_of("Is {2} before the audit, given {1}?") == {1, 2}
    assert dependencies_of("What is the vacation policy?") == set()


def test_sanitize_plan_keeps_valid_backward_references():
    plan = ["Which vendor supplies Atlas?", "Renewal date of {1}?", "Is {2} before the audit?"]
    assert sanitize_plan(plan) == plan


def test_sanitize_plan_neutralises_self_forward_and_out_of_range_references():
    plan = ["What is {1}?", "Compare with {3}", "Look at {9} and {0}"]
    assert sanitize_plan(plan) == [
        "What is the previous result?",
        "Compare with the previous result",
        "Look at the previous result and the previous result",
    ]


def test_decompose_query_sanitises_bad_placeholders(monkeypatch):
    _mock_llm_response(monkeypatch, json.dumps(["Who leads it? {2}", "Ok {1}"]))
    assert decompose_query("who leads it and then ok") == ["Who leads it? the previous result", "Ok {1}"]


def test_display_query_makes_placeholders_readable():
    assert display_query("Renewal date of {1}?") == "Renewal date of the answer to step 1?"


def test_resolve_returns_query_unchanged_without_placeholders_and_makes_no_llm_call(monkeypatch):
    monkeypatch.setattr(decomposer.groq_manager, "get_client",
                        lambda: (_ for _ in ()).throw(AssertionError("no LLM call expected")))
    assert resolve_dependent_query("What is the policy?", {1: "x"}, {1: "q"}) == "What is the policy?"


def test_resolve_uses_llm_rewrite(monkeypatch):
    _mock_llm_response(monkeypatch, "  What is the renewal date of Acme Corp?  ")
    result = resolve_dependent_query(
        "What is the renewal date of {1}?", {1: "Acme Corp supplies Atlas."}, {1: "Who supplies Atlas?"}
    )
    assert result == "What is the renewal date of Acme Corp?"


def test_resolve_falls_back_to_inlined_answer_on_llm_error(monkeypatch):
    def _raise():
        raise RuntimeError("no keys configured")
    monkeypatch.setattr(decomposer.groq_manager, "get_client", _raise)

    result = resolve_dependent_query(
        "What is the renewal date of {1}?", {1: "Acme Corp"}, {1: "Who supplies Atlas?"}
    )
    assert result == "What is the renewal date of (Acme Corp)?"


def test_resolve_falls_back_when_llm_leaves_a_placeholder(monkeypatch):
    _mock_llm_response(monkeypatch, "What is the renewal date of {1}?")
    result = resolve_dependent_query("When does {1} renew?", {1: "Acme Corp"}, {1: "Who?"})
    assert "{1}" not in result and "Acme Corp" in result
