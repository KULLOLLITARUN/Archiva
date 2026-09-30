"""
Regression tests for small-output LLM helpers running on a *reasoning* model
(openai/gpt-oss-*), found live: the decomposer, query rewriter and judge asked
for 20-300 tokens, the model's hidden thinking used them all, and the API
returned finish_reason="length" with EMPTY content. Callers then silently
skipped decomposition / rewriting, and the judge read "" as a NO verdict.

Also covers the typographic look-alikes such models emit (non-breaking
hyphens in identifiers) that made a rewritten search query miss the document.
"""

import agents.decomposer as decomposer
import agents.judge as judge
import agents.query_rewriter as query_rewriter
from agents.decomposer import _compact, normalize_llm_text, resolve_dependent_query
from llm.groq_manager import light_completion_params


# ── light_completion_params ─────────────────────────────────────────────────────

def test_reasoning_model_gets_low_effort_and_a_realistic_budget():
    params = light_completion_params("openai/gpt-oss-20b", 20)
    assert params["extra_body"] == {"reasoning_effort": "low"}
    assert params["max_tokens"] >= 600          # room for the thinking, not just the reply


def test_reasoning_model_keeps_a_larger_requested_budget():
    assert light_completion_params("openai/gpt-oss-120b", 2000)["max_tokens"] == 2000


def test_ordinary_model_is_left_untouched():
    assert light_completion_params("llama-3.1-8b-instant", 60) == {"max_tokens": 60}


# ── fake client capturing what each helper sends ───────────────────────────────

class _Msg:
    def __init__(self, content): self.content = content


class _Choice:
    def __init__(self, content): self.message = _Msg(content)


class _Resp:
    def __init__(self, content): self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, content, sink): self._content, self._sink = content, sink

    def create(self, **kwargs):
        self._sink.append(kwargs)
        return _Resp(self._content)


class _Client:
    def __init__(self, content, sink):
        self.chat = type("Chat", (), {"completions": _Completions(content, sink)})()


def _patch(monkeypatch, module, content, model="openai/gpt-oss-20b"):
    sent = []
    monkeypatch.setattr(module.groq_manager, "get_client", lambda: ("key", _Client(content, sent)))
    monkeypatch.setattr(module, "GROQ_FAST", model)
    return sent


def test_decomposer_requests_low_reasoning_effort(monkeypatch):
    sent = _patch(monkeypatch, decomposer, '["a?", "b?"]')
    decomposer.decompose_query("a and b, long enough to matter")
    assert sent[0]["extra_body"] == {"reasoning_effort": "low"} and sent[0]["max_tokens"] >= 600


def test_dependency_resolver_requests_low_reasoning_effort(monkeypatch):
    sent = _patch(monkeypatch, decomposer, "What is the renewal date of Acme?")
    resolve_dependent_query("Renewal date of {1}?", {1: "Acme"}, {1: "Who?"})
    assert sent[0]["extra_body"] == {"reasoning_effort": "low"} and sent[0]["max_tokens"] >= 600


def test_query_rewriter_requests_low_reasoning_effort(monkeypatch):
    sent = _patch(monkeypatch, query_rewriter, "renewal date vendor contract")
    query_rewriter.rewrite_for_retry("when does it renew", "Not found.", "low_overlap", attempt=1)
    assert sent[0]["extra_body"] == {"reasoning_effort": "low"} and sent[0]["max_tokens"] >= 600


def test_judge_requests_low_reasoning_effort(monkeypatch):
    sent = _patch(monkeypatch, judge, "YES")
    judge.judge_faithfulness("q", "an answer", [{"text": "context"}])
    assert sent[0]["extra_body"] == {"reasoning_effort": "low"} and sent[0]["max_tokens"] >= 600


# ── judge: an empty reply is an error, never a NO ───────────────────────────────

def test_judge_treats_an_empty_reply_as_error_not_unfaithful(monkeypatch):
    _patch(monkeypatch, judge, "")
    verdict = judge.judge_faithfulness("q", "a perfectly good answer", [{"text": "context"}])
    assert verdict == {"faithful": True, "reason": "judge_error"}


def test_judge_still_reports_a_real_no(monkeypatch):
    _patch(monkeypatch, judge, "NO")
    assert judge.judge_faithfulness("q", "a", [{"text": "c"}])["faithful"] is False


def test_judge_still_reports_a_real_yes(monkeypatch):
    _patch(monkeypatch, judge, "YES")
    assert judge.judge_faithfulness("q", "a", [{"text": "c"}])["faithful"] is True


# ── look-alike characters ───────────────────────────────────────────────────────

def test_non_breaking_hyphens_become_plain_hyphens():
    assert normalize_llm_text("Disk‑Clone‑1") == "Disk-Clone-1"
    assert normalize_llm_text("a–b—c−d") == "a-b-c-d"


def test_odd_spaces_and_curly_quotes_are_normalised():
    assert normalize_llm_text("Snapshot → Disk") == "Snapshot → Disk"
    assert normalize_llm_text("“quoted” and it’s") == '"quoted" and it\'s'


def test_compact_strips_citation_markers_so_filenames_do_not_pollute_the_query():
    answer = "Created from **Disk‑Clone‑1**. [Source: Azure_VM_Cloning_Plan.docx, page 1] 【Source: x.docx, page 2】"
    compact = _compact(answer)
    assert compact == "Created from **Disk-Clone-1**."
    assert "Source" not in compact and "docx" not in compact


def test_mechanical_fallback_uses_clean_answer_text(monkeypatch):
    def _raise():
        raise RuntimeError("no keys")
    monkeypatch.setattr(decomposer.groq_manager, "get_client", _raise)
    result = resolve_dependent_query(
        "What zone must {1} be in?",
        {1: "Disk‑Clone‑1 [Source: a.docx, page 1]"}, {1: "Which disk?"},
    )
    assert result == "What zone must (Disk-Clone-1) be in?"


def test_llm_rewrite_with_lookalike_hyphens_is_normalised(monkeypatch):
    _patch(monkeypatch, decomposer, "What region must Disk‑Clone‑1 be in?")
    result = resolve_dependent_query("What region must {1} be in?", {1: "Disk-Clone-1"}, {1: "Which disk?"})
    assert result == "What region must Disk-Clone-1 be in?"


def test_planner_subquestions_are_normalised(monkeypatch):
    _patch(monkeypatch, decomposer, '["Which disk‑clone‑1?", "Zone of {1}?"]')
    assert decomposer.decompose_query("long question about a disk and its zone here") == [
        "Which disk-clone-1?", "Zone of {1}?",
    ]


# ── the resolver sees the original question ─────────────────────────────────────

def test_resolver_passes_the_original_question_to_the_model(monkeypatch):
    sent = _patch(monkeypatch, decomposer, "What happens if the region of Disk-Clone-1 does not match?")
    resolve_dependent_query(
        "What happens if the region of {1} does not match?",
        {1: "Disk-Clone-1"}, {1: "Which disk?"},
        original_query="Which disk is used, and what happens if its region doesn't match?",
    )
    user_message = sent[0]["messages"][1]["content"]
    assert "Original full question: Which disk is used, and what happens if its region doesn't match?" in user_message


def test_resolver_works_without_an_original_question(monkeypatch):
    sent = _patch(monkeypatch, decomposer, "When does Acme renew?")
    resolve_dependent_query("When does {1} renew?", {1: "Acme"}, {1: "Who?"})
    assert "Original full question" not in sent[0]["messages"][1]["content"]


# ── step_heading ────────────────────────────────────────────────────────────────

def test_heading_is_the_clean_rewrite_when_the_model_resolved_the_step():
    from agents.decomposer import step_heading
    assert step_heading(
        "What region must {1} be in?", "What region must Disk-Clone-1 be in?", {1: "Disk-Clone-1"},
    ) == "What region must Disk-Clone-1 be in?"


def test_heading_uses_generic_wording_for_the_paste_in_fallback():
    from agents.decomposer import _mechanical_resolve, step_heading
    answers = {1: "Disk-Clone-1"}
    pasted = _mechanical_resolve("What region must {1} be in?", answers)
    assert step_heading("What region must {1} be in?", pasted, answers) == "What region must the answer to step 1 be in?"
