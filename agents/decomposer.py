"""
agents/decomposer.py — Multi-hop query decomposition.

Detects when a user's question actually contains multiple DISTINCT,
independently-answerable sub-questions (as opposed to a single complex
question that merely touches multiple topics, e.g. the existing "compare"
intent) and splits it so each part gets its own retrieve->generate->reflect
pass through agents/loop.py, then the answers are merged.

Cost control: should_decompose() is a free, deterministic pre-filter run on
every query. The LLM call in decompose_query() only fires when that
pre-filter says a query is plausibly multi-part — most queries never reach
it, keeping this close to free for the common case.

Sequential/dependent sub-questions: decompose_query() is told to produce
self-contained sub-questions, but a small fast model asked to "preserve
original wording" will often leave a pronoun in a later part ("...and how
fast must THEY respond to alerts") that plainly refers back to an earlier
sub-question's answer ("who is on the response team"). agents/loop.py runs
sub-questions in order (not in parallel) and anchor_to_prior_answer() below
resolves that reference before the dependent sub-question runs — same
anchor-don't-rewrite pattern chatbot/rewriter.py already uses for
conversational follow-ups, applied here within one decomposed query's
sub-question chain instead of across chat turns.

Dependent chains (multi-step reasoning): anchoring only handles one loose
pronoun. When a later step genuinely needs an earlier step's RESULT ("find
the vendor for Project Atlas, THEN what is THAT vendor's renewal date, THEN
is it before the audit"), decompose_query() marks the dependency with a
"{N}" placeholder (N = 1-based index of an earlier step). At run time
resolve_dependent_query() turns each such step into a standalone question
using the actual earlier answers, and agents/loop.py skips any step whose
prerequisite could not be answered rather than retrieving for a question
that still contains an unresolved reference.
"""

import json
import re
from typing import Dict, List, Optional, Set, Tuple

from config import GROQ_FAST
from llm.groq_manager import groq_manager, light_completion_params

# ── Configuration ─────────────────────────────────────────────────────────────

_MIN_WORDS_TO_CONSIDER = 10
_CONNECTORS = (" and ", " also ", " as well as ", ";", " then ")

_SYSTEM_PROMPT = (
    "You are a query analysis agent for a document Q&A system. Decide whether "
    "the user's question contains multiple DISTINCT, independently-answerable "
    "sub-questions, as opposed to a single question that merely mentions "
    "several things.\n\n"
    "Examples:\n"
    '- "What is the vacation policy and how fast must oncall respond to alerts?" '
    "-> TWO unrelated sub-questions.\n"
    '- "Compare the vacation policy across regions" -> ONE question (a single '
    "comparison, not two separate asks).\n\n"
    "Some questions are a DEPENDENT CHAIN: a later step needs the ANSWER to an "
    "earlier step before it can even be asked. Mark that dependency with a "
    "{N} placeholder, where N is the 1-based number of the earlier step.\n"
    '- "Which vendor supplies Project Atlas, what is that vendor\'s renewal '
    'date, and is it before the audit?" -> ["Which vendor supplies Project '
    'Atlas?", "What is the renewal date of {1}?", "Is {2} before the audit '
    'date?"]\n\n'
    "Rules:\n"
    "- If it is a single question, return a JSON array with exactly ONE item: "
    "the original question, unchanged.\n"
    "- If it has 2-4 distinct sub-questions, return each as its own "
    "self-contained question, preserving the original wording where possible.\n"
    "- Use {N} only for a genuine dependency on an EARLIER step's answer. "
    "Independent sub-questions must NOT contain placeholders.\n"
    "- A later step that points back at something an earlier step asks "
    "about (\"that disk\", \"the same vendor\", \"it\", \"those\", \"they\") IS "
    "dependent: replace the pointer with {N} so the step stands alone once "
    "{N} is filled in.\n"
    '- "Which disk creates VM-1, what zone must that disk be in, and what '
    'happens if it is wrong?" -> ["Which disk creates VM-1?", "What zone must '
    '{1} be in?", "What happens if the zone of {1} is wrong?"]\n'
    "- Keep every qualifier from the original question in the wording of the "
    "step it belongs to (region, zone, date, name). Never drop one.\n"
    "- Never invent a sub-question that isn't implied by the original text.\n"
    "- Return ONLY a JSON array of strings. No markdown, no explanation."
)

_MAX_TOKENS = 300
_TEMPERATURE = 0.0
_MAX_SUBQUESTIONS = 4


# ── Cheap pre-filter (no LLM call) ─────────────────────────────────────────────

def should_decompose(query: str) -> bool:
    """
    Deterministic pre-filter: True means "plausibly multi-part, worth
    checking with the LLM" — NOT "definitely decompose". Keeps the LLM call
    in decompose_query() off the hot path for ordinary short/simple queries.
    """
    q = query.strip()
    if len(q.split()) < _MIN_WORDS_TO_CONSIDER:
        return False
    if q.count("?") >= 2:
        return True
    lowered = q.lower()
    return any(conn in lowered for conn in _CONNECTORS)


# ── Dependency placeholders ({N} = "the answer to step N") ─────────────────────

_PLACEHOLDER_RE = re.compile(r"\{(\d+)\}")


def dependencies_of(sub_query: str) -> Set[int]:
    """1-based step numbers this sub-question's {N} placeholders point at."""
    return {int(n) for n in _PLACEHOLDER_RE.findall(sub_query)}


def sanitize_plan(sub_queries: List[str]) -> List[str]:
    """
    Neutralise placeholders the LLM got wrong: a step may only depend on an
    EARLIER step (1 <= N < its own position). A self/forward/out-of-range
    reference can never be resolved, so it is replaced with plain wording
    instead of being left to poison retrieval as a literal "{3}".
    """
    cleaned: List[str] = []
    for position, q in enumerate(sub_queries, start=1):
        cleaned.append(_PLACEHOLDER_RE.sub(
            lambda m: m.group(0) if 1 <= int(m.group(1)) < position else "the previous result",
            q,
        ))
    return cleaned


# ── Deterministic back-reference linking ──────────────────────────────────────
# The planner is told to write {N} for "that vendor"-style references, but it
# doesn't always (seen in the answer-quality eval: "When does that vendor's
# contract renew?" came back with no placeholder, ran on its own, and
# answered "Not found"). This pass links the references it left behind.

_DEMONSTRATIVE_RE = re.compile(
    r"\b(?:that|this|those|these|the same)\s+([a-z][a-z0-9-]*)('s)?", re.IGNORECASE
)
# "Is that before ...", "Was it approved ...": a bare pronoun as the subject
# of a yes/no step refers to the previous step's answer.
_SUBJECT_PRONOUN_RE = re.compile(
    r"^(\s*(?:is|was|are|were|does|did|will|can)\s+)(?:that|this|it)\b", re.IGNORECASE
)


def _noun_key(word: str) -> str:
    word = word.lower()
    for suffix in ("'s", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def promote_back_references(sub_queries: List[str]) -> List[str]:
    """
    Turn leftover back-references into {N} placeholders.

      * "that/this/those/these/the same <noun>" where <noun> appears in an
        earlier step's question -> {k} for the most recent such step
        ("When does that vendor's contract renew?" after "Which vendor
        supplies Atlas?" -> "When does {1}'s contract renew?").
      * A bare subject pronoun opening a yes/no step ("Is that before the
        audit?") -> the previous step.

    Deliberately conservative: a demonstrative whose noun appears in no
    earlier step ("the team that manages the roadmap") is left alone, and
    steps that already carry a placeholder are not touched.
    """
    promoted: List[str] = []
    for position, query in enumerate(sub_queries, start=1):
        if position == 1 or dependencies_of(query):
            promoted.append(query)
            continue

        earlier_nouns = [
            {_noun_key(w) for w in re.findall(r"[a-z][a-z0-9-]*", q.lower())}
            for q in sub_queries[: position - 1]
        ]

        def link(match):
            noun = _noun_key(match.group(1))
            for step in range(position - 1, 0, -1):
                if noun in earlier_nouns[step - 1]:
                    return "{%d}%s" % (step, match.group(2) or "")
            return match.group(0)

        rewritten = _DEMONSTRATIVE_RE.sub(link, query)
        if rewritten == query:
            rewritten = _SUBJECT_PRONOUN_RE.sub(lambda m: m.group(1) + "{%d}" % (position - 1), query, count=1)
        promoted.append(rewritten)
    return promoted


# ── LLM-assisted split ─────────────────────────────────────────────────────────

def decompose_query(query: str) -> List[str]:
    """
    Ask the LLM whether *query* splits into distinct sub-questions.

    Returns a list of 1+ questions. A single-item list (usually [query]
    itself) means "don't decompose" — callers should treat that as a no-op.
    Falls back to [query] on any error, same failure-mode convention as
    agents/query_rewriter.py.
    """
    try:
        _, client = groq_manager.get_client()
        response = client.chat.completions.create(
            model=GROQ_FAST,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=_TEMPERATURE,
            **light_completion_params(GROQ_FAST, _MAX_TOKENS),
        )
        raw = (response.choices[0].message.content or "").strip()
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return [query]

        parsed = json.loads(match.group())
        sub_queries = [normalize_llm_text(str(q)).strip() for q in parsed if isinstance(q, str) and q.strip()]
        if not sub_queries:
            return [query]
        return sanitize_plan(promote_back_references(sub_queries[:_MAX_SUBQUESTIONS]))

    except Exception as exc:
        print(f"  [WARN]  [decomposer] Decomposition failed: {exc} — treating as a single question")
        return [query]


# ── Sequential reference resolution ─────────────────────────────────────────────

# Deliberately narrow — only pronouns that are rarely anything OTHER than a
# back-reference in a short question. "that"/"this"/"these" are excluded on
# purpose: they're common as relative-clause/demonstrative words ("the team
# THAT manages the roadmap") and would false-trigger constantly if included.
_REFERENCE_WORDS = ("it", "its", "they", "their", "them")
_REFERENCE_RE = re.compile(r"\b(" + "|".join(_REFERENCE_WORDS) + r")\b", re.IGNORECASE)


def anchor_to_prior_answer(sub_query: str, prior_answer: str) -> str:
    """
    If sub_query contains a pronoun that plausibly points at the previous
    sub-question's answer ("what is THEIR vacation policy"), anchor it to
    that answer so both retrieval and generation see the resolved
    reference. A no-op when there's nothing to resolve — most decomposed
    sub-question pairs are genuinely independent, and this must not fire
    on every short sub-question the way chatbot.rewriter.is_followup()'s
    "<6 words" fallback would (many self-contained sub-questions ARE
    short — "What is the vacation policy?" is 5 words).
    """
    if not prior_answer or not _REFERENCE_RE.search(sub_query):
        return sub_query
    return f"{sub_query} (referring to: {prior_answer})"


# ── Dependent-step resolution ───────────────────────────────────────────────────

_MAX_ANSWER_CHARS = 400
_RESOLVE_MAX_TOKENS = 100

_RESOLVE_PROMPT = (
    "You rewrite one step of a multi-step question into a standalone "
    "question. The original full question, the earlier steps and their "
    "answers are given. Replace every {N} placeholder (and any pronoun that "
    "refers to an earlier answer) with the concrete entity or value from that "
    "step's answer, so the question can be understood with no other context.\n"
    "Rules:\n"
    "- {N} stands for WHAT step N's question asks for, not for other things "
    "its answer mentions. If Step 1 asks \"Which disk creates VM-A?\" and the "
    "answer is \"VM-A is created from Disk-7\", then {1} = Disk-7, NOT VM-A.\n"
    "- Keep every part of the step's meaning. Use the original full question "
    "to keep qualifiers the step depends on (region, zone, date, name); never "
    "drop or generalise them.\n"
    "- Copy names and identifiers EXACTLY as written in the answers, using "
    "plain ASCII hyphens.\n"
    "- Keep the question's intent and wording otherwise unchanged.\n"
    "- Use only facts stated in the given answers; never add new facts.\n"
    "- Return ONLY the rewritten question. No quotes, no explanation."
)


# LLMs (gpt-oss in particular) emit typographic look-alikes: non-breaking
# hyphens (U+2011) inside identifiers ("Disk\u2011Clone\u20111"), narrow
# no-break spaces (U+202F), curly quotes. The document has the plain ASCII
# characters, so a step rewritten with the look-alikes searches for a token
# that exists nowhere: keyword search misses, and the step wrongly answers
# "Not found". Everything an LLM writes that becomes part of a search query
# goes through this first.
_DASHES_RE = re.compile("[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]")
_ODD_SPACES_RE = re.compile("[\u00a0\u2007\u2009\u200a\u202f\u205f\u3000]")
_CITATION_RE = re.compile(r"[\[\u3010][^\]\u3011]*Source:[^\]\u3011]*[\]\u3011]")


def normalize_llm_text(text: str) -> str:
    """ASCII-fy hyphen/space/quote look-alikes so LLM output matches document text."""
    text = _DASHES_RE.sub("-", text or "")
    text = _ODD_SPACES_RE.sub(" ", text)
    return (text.replace("\u2018", "'").replace("\u2019", "'")
                .replace("\u201c", '"').replace("\u201d", '"'))


def _compact(answer: str) -> str:
    """Shorten an answer for use inside a search query: no citations, plain ASCII punctuation."""
    text = " ".join(normalize_llm_text(_CITATION_RE.sub(" ", answer or "")).split())
    return text if len(text) <= _MAX_ANSWER_CHARS else text[:_MAX_ANSWER_CHARS].rstrip() + "..."


def display_query(sub_query: str) -> str:
    """User-facing text for a step: {N} becomes "the answer to step N"."""
    return _PLACEHOLDER_RE.sub(lambda m: f"the answer to step {m.group(1)}", sub_query)


def _mechanical_resolve(sub_query: str, answers: Dict[int, str]) -> str:
    """No-LLM fallback: inline the (compacted) answers into the placeholders."""
    return _PLACEHOLDER_RE.sub(
        lambda m: f"({_compact(answers.get(int(m.group(1)), ''))})", sub_query
    )


def step_heading(sub_query: str, resolved: str, answers: Dict[int, str]) -> str:
    """
    Heading to show the user for a dependent step.

    When the model cleanly rewrote the step ("What region must Disk-Clone-1
    be in?") that standalone question is the natural heading. If resolution
    fell back to pasting the earlier answers into the placeholders, that text
    is clutter, so fall back to the generic "the answer to step N" wording.
    """
    if resolved and resolved != _mechanical_resolve(sub_query, answers):
        return resolved
    return display_query(sub_query)


def resolve_dependent_query(
    sub_query: str,
    answers: Dict[int, str],
    questions: Dict[int, str],
    original_query: Optional[str] = None,
) -> str:
    """
    Turn a step containing {N} placeholders into a standalone question using
    the real answers of the steps it depends on.

    One small fast-model call rewrites it cleanly ("What is the renewal date
    of Acme Corp?"). If that call fails or returns nothing usable, falls back
    to inlining the compacted answers — worse retrieval phrasing, but never a
    crash and never an unresolved "{1}" left in the query. A step with no
    placeholders is returned unchanged with no LLM call.
    """
    deps = sorted(d for d in dependencies_of(sub_query) if d in answers)
    if not deps:
        return sub_query

    context = "\n".join(
        f"Step {n}: {questions.get(n, '')}\nAnswer {n}: {_compact(answers[n])}" for n in deps
    )
    if original_query:
        context = f"Original full question: {original_query}\n\n{context}"
    try:
        _, client = groq_manager.get_client()
        response = client.chat.completions.create(
            model=GROQ_FAST,
            messages=[
                {"role": "system", "content": _RESOLVE_PROMPT},
                {"role": "user", "content": f"{context}\n\nStep to rewrite: {sub_query}"},
            ],
            temperature=_TEMPERATURE,
            **light_completion_params(GROQ_FAST, _RESOLVE_MAX_TOKENS),
        )
        rewritten = " ".join(normalize_llm_text(response.choices[0].message.content or "").split()).strip("\"' ")
        if rewritten and not _PLACEHOLDER_RE.search(rewritten):
            return rewritten
    except Exception as exc:
        print(f"  [WARN]  [decomposer] Dependency resolution failed: {exc} — inlining answers")
    return _mechanical_resolve(sub_query, answers)
