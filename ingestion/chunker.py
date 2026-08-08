"""
ingestion/chunker.py — Two-level Parent-Child text chunker.

Upgrade (Production Part 1 — Parent-Child Chunking):
  - chunk_page() now produces a TWO-LEVEL hierarchy:
      Parent chunks  — large sections (≤ PARENT_CHUNK_SIZE tokens).
                       Each parent gets a unique parent_id.
      Child chunks   — small precision units (≤ CHILD_CHUNK_SIZE tokens)
                       sliced from each parent.  Each child carries:
                         metadata["parent_id"]   — links back to parent
                         metadata["parent_text"] — full parent text (for LLM)
  - chunk_document() returns child chunks only (unchanged interface).
  - Log-file chunking is NOT split into parent-child — log entries are
    already semantically self-contained; they are returned as flat chunks
    with parent_id == chunk_id (self-referential).

Backward compat:
  - chunk_document() signature, return type, and all metadata keys unchanged.
  - New metadata keys added: parent_id, parent_text.
  - All previously produced chunk_ids remain unique.
"""

import hashlib
import re
from datetime import datetime, timezone
from typing import Dict, List, Set

from config import (
    CHUNK_SIZE, CHUNK_OVERLAP, MAX_CHUNKS_PER_FILE,
    PARENT_CHUNK_SIZE, CHILD_CHUNK_SIZE,
)


# ── Token estimator ───────────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """Soft heuristic token estimator: characters / 4."""
    return len(text) // 4


# ── Content hash (Part 3.2) ───────────────────────────────────────────────────

def content_hash(text: str) -> str:
    """Return the SHA-256 hex digest of the chunk text (UTF-8 encoded)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── Log-file detection ────────────────────────────────────────────────────────

_TIMESTAMP_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2}"
    r"|\d{2}/\d{2}/\d{4}"
    r"|[A-Za-z]{3}\s+\d{1,2}\s+\d{2}:\d{2})",
    re.IGNORECASE,
)


def is_log_file(text: str, sample_lines: int = 50, threshold: float = 0.30) -> bool:
    lines = [l for l in text.splitlines() if l.strip()][:sample_lines]
    if not lines:
        return False
    matched = sum(1 for l in lines if _TIMESTAMP_RE.search(l[:40]))
    return matched / len(lines) >= threshold


# ── Log-aware chunking ────────────────────────────────────────────────────────

_STACK_INDENT_RE = re.compile(r"^\s+at\s+|^\s+File\s+\"|^\s+Traceback", re.IGNORECASE)
_ERROR_LINE_RE   = re.compile(r"\b(ERROR|EXCEPTION|CRITICAL|FATAL|WARN)\b", re.IGNORECASE)


def _split_log_entries(text: str) -> List[str]:
    raw_lines = text.splitlines()
    entries: List[List[str]] = []
    current: List[str] = []
    in_stack = False

    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            continue

        starts_new_entry = bool(_TIMESTAMP_RE.search(line[:40]))

        if starts_new_entry and current:
            if not in_stack:
                entries.append(current)
                current = []
            in_stack = False

        current.append(line)

        if _ERROR_LINE_RE.search(line):
            in_stack = True
        elif in_stack and not _STACK_INDENT_RE.match(line):
            in_stack = False

    if current:
        entries.append(current)

    return ["\n".join(e) for e in entries if e]


def chunk_log(
    text: str,
    page_num: int,
    file_id: str,
    filename: str,
    file_type: str,
    uploaded_at: str,
    user_id: str = "",
) -> List[Dict]:
    """Log-aware chunker; keeps stack traces whole within one chunk.
    Log entries are flat chunks — parent_id == chunk_id (self-referential).
    """
    entries = _split_log_entries(text)
    chunks: List[Dict] = []
    current_entries: List[str] = []
    current_tokens = 0
    chunk_index = 0

    def _make_log_chunk(entries_list: List[str], idx: int) -> Dict:
        body = "\n".join(entries_list)
        cid = f"{file_id}_p{page_num}_c{idx}"
        return {
            "chunk_id": cid,
            "text": body,
            "metadata": {
                "doc_id":        file_id,
                "file_id":       file_id,
                "filename":      filename,
                "file_type":     file_type,
                "page":          page_num,
                "section":       f"Page {page_num}",
                "chunk_index":   idx,
                "uploaded_at":   uploaded_at,
                "date":          uploaded_at,
                "content_hash":  content_hash(body),
                "user_id":       user_id,
                # Log chunks are self-referential parents
                "parent_id":     cid,
                "parent_text":   body,
            },
        }

    for entry in entries:
        entry_tokens = estimate_tokens(entry)

        if current_tokens + entry_tokens > CHUNK_SIZE and current_entries:
            chunks.append(_make_log_chunk(current_entries, chunk_index))
            chunk_index += 1
            current_entries = []
            current_tokens  = 0

        current_entries.append(entry)
        current_tokens += entry_tokens

    if current_entries:
        chunks.append(_make_log_chunk(current_entries, chunk_index))

    return chunks


# ── Paragraph splitting ───────────────────────────────────────────────────────

def split_into_paragraphs(text: str) -> List[str]:
    parts = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if len(block) > 1600:
            for line in block.split("\n"):
                line = line.strip()
                if line:
                    parts.append(line)
        else:
            parts.append(block)
    return parts


# ── Parent chunker ────────────────────────────────────────────────────────────

def _build_parents(
    paragraphs: List[str],
    page_num: int,
    file_id: str,
) -> List[Dict]:
    """
    Accumulate paragraphs into parent chunks ≤ PARENT_CHUNK_SIZE tokens.
    Returns a list of parent dicts with keys: parent_id, text, page.
    """
    parents: List[Dict] = []
    current_paras: List[str] = []
    current_tokens = 0
    parent_index = 0

    def _flush(paras: List[str], idx: int) -> Dict:
        body = "\n".join(paras)
        return {
            "parent_id":  f"{file_id}_p{page_num}_parent{idx}",
            "text":       body,
            "page":       page_num,
        }

    for para in paragraphs:
        para_tokens = estimate_tokens(para)

        if current_tokens + para_tokens > PARENT_CHUNK_SIZE and current_paras:
            parents.append(_flush(current_paras, parent_index))
            parent_index += 1
            current_paras  = []
            current_tokens = 0

        current_paras.append(para)
        current_tokens += para_tokens

    if current_paras:
        parents.append(_flush(current_paras, parent_index))

    return parents


# ── Child chunker ─────────────────────────────────────────────────────────────

def _build_children(
    parent: Dict,
    file_id: str,
    filename: str,
    file_type: str,
    uploaded_at: str,
    user_id: str,
    child_start_index: int,
) -> List[Dict]:
    """
    Slice a parent chunk into child chunks ≤ CHILD_CHUNK_SIZE tokens.
    Each child carries parent_id and parent_text so the LLM gets full context.
    """
    parent_text = parent["text"]
    parent_id   = parent["parent_id"]
    page_num    = parent["page"]

    # Split parent into sentences / lines for finer child boundaries
    lines = [l.strip() for l in parent_text.replace(". ", ".\n").splitlines() if l.strip()]
    if not lines:
        return []

    children: List[Dict] = []
    current_lines: List[str] = []
    current_tokens = 0
    child_index = child_start_index

    def _flush_child(lines_list: List[str], idx: int) -> Dict:
        body = " ".join(lines_list)
        cid = f"{file_id}_p{page_num}_c{idx}"
        return {
            "chunk_id": cid,
            "text": body,
            "metadata": {
                "doc_id":        file_id,
                "file_id":       file_id,
                "filename":      filename,
                "file_type":     file_type,
                "page":          page_num,
                "section":       f"Page {page_num}",
                "chunk_index":   idx,
                "uploaded_at":   uploaded_at,
                "date":          uploaded_at,
                "content_hash":  content_hash(body),
                "user_id":       user_id,
                "parent_id":     parent_id,
                "parent_text":   parent_text,   # ← full parent for LLM context
            },
        }

    for line in lines:
        line_tokens = estimate_tokens(line)

        if current_tokens + line_tokens > CHILD_CHUNK_SIZE and current_lines:
            children.append(_flush_child(current_lines, child_index))
            child_index += 1
            current_lines  = []
            current_tokens = 0

        current_lines.append(line)
        current_tokens += line_tokens

    if current_lines:
        children.append(_flush_child(current_lines, child_index))

    return children


# ── Page-level entry point ────────────────────────────────────────────────────

def chunk_page(
    page_text: str,
    page_num: int,
    file_id: str,
    filename: str,
    file_type: str,
    user_id: str = "",
) -> List[Dict]:
    """
    Chunk a single page into parent-child pairs.

    If the page looks like a log file, delegates to chunk_log() (flat chunks).
    Otherwise produces a two-level hierarchy:
      1. Parent chunks (≤ PARENT_CHUNK_SIZE tokens) — rich LLM context.
      2. Child chunks (≤ CHILD_CHUNK_SIZE tokens)  — precise search units.
    Returns only child chunks; parent text is embedded in each child's metadata.
    """
    uploaded_at = datetime.now(timezone.utc).isoformat()

    if is_log_file(page_text):
        return chunk_log(page_text, page_num, file_id, filename, file_type, uploaded_at, user_id)

    paragraphs = split_into_paragraphs(page_text)
    if not paragraphs:
        return []

    parents  = _build_parents(paragraphs, page_num, file_id)
    children: List[Dict] = []
    child_counter = 0

    for parent in parents:
        parent_children = _build_children(
            parent, file_id, filename, file_type, uploaded_at, user_id, child_counter
        )
        children.extend(parent_children)
        child_counter += len(parent_children)

    return children


# ── Document-level entry point ────────────────────────────────────────────────

def chunk_document(
    pages: List[Dict],
    file_id: str,
    filename: str,
    file_type: str,
    user_id: str = "",
) -> List[Dict]:
    """
    Chunk all pages of a document into parent-child child chunks.

    Enforces MAX_CHUNKS_PER_FILE cap.
    Deduplicates chunks by SHA-256 hash within this document.
    Returns child chunks only — parent context lives in metadata["parent_text"].
    """
    all_chunks: List[Dict] = []
    seen_hashes: Set[str]  = set()

    for page in pages:
        page_chunks = chunk_page(
            page_text=page["text"],
            page_num=page["page"],
            file_id=file_id,
            filename=filename,
            file_type=file_type,
            user_id=user_id,
        )

        for chunk in page_chunks:
            h = chunk["metadata"].get("content_hash", "")
            if h and h in seen_hashes:
                continue
            if h:
                seen_hashes.add(h)
            all_chunks.append(chunk)

            if len(all_chunks) >= MAX_CHUNKS_PER_FILE:
                return all_chunks

    return all_chunks
