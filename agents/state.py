"""
agents/state.py — Shared mutable state object for the self-healing agent loop.

Part 1.1: Encapsulates all per-request state so healing actions (root_cause,
healer) can read and modify it without passing dozens of arguments.

The AgentState is created fresh for every query — never shared across requests.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AgentState:
    """
    Mutable state carried through the self-healing agent loop.

    Fields are grouped by concern:
      Query         — original + rewritten query strings
      Retrieval     — raw and reranked document lists, dynamic top_k
      Generation    — produced answer, prompt mode
      Diagnostics   — failure classification, confidence, attempt counter
    """

    # ── Query ──────────────────────────────────────────────────────────────────
    original_query:  str            = ""
    rewritten_query: Optional[str]  = None    # set by healer on REWRITE_QUERY

    # ── Retrieval ──────────────────────────────────────────────────────────────
    documents:          List[dict]  = field(default_factory=list)
    reranked_documents: List[dict]  = field(default_factory=list)

    # Dynamic top_k — healer increases this on INSUFFICIENT_CONTEXT
    top_k: int = 5

    # ── Generation ─────────────────────────────────────────────────────────────
    answer:      Optional[str] = None
    prompt_mode: str           = "normal"   # "normal" | "strict"

    # ── Diagnostics ────────────────────────────────────────────────────────────
    failure_type:   Optional[str] = None    # RETRIEVAL_FAILURE | INSUFFICIENT_CONTEXT | …
    failure_reason: Optional[str] = None    # human-readable from reflect()
    confidence:     float         = 1.0
    attempt:        int           = 0
    max_attempts:   int           = 3

    # ── Accumulator ────────────────────────────────────────────────────────────
    search_queries: List[str] = field(default_factory=list)  # all queries tried

    def active_query(self) -> str:
        """Return the rewritten query if available, else the original."""
        return self.rewritten_query if self.rewritten_query else self.original_query

    def reset_healing(self) -> None:
        """Clear transient failure fields between attempts."""
        self.failure_type   = None
        self.failure_reason = None
