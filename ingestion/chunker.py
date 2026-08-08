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


# ── Table-shaped content detection ────────────────────────────────────────────
# "col=value, col=value" rows — the format parse_csv() and the PDF table
# extractor (ingestion/parser.py) both render tabular data as. Detected from
# shape, same content-sniffing pattern as is_log_file(), not a caller-
# supplied flag — so CSV output and PDF-extracted tables are routed
# consistently through row-aware chunking regardless of which parser
# produced them.
#
# Deliberately plain string ops, not a regex: this runs against arbitrary
# uploaded document content on every ingested page, and a permissive
# "key=value(,key=value)+" pattern with nested unbounded quantifiers is a
# real catastrophic-backtracking (ReDoS) risk against adversarial/malformed
# input. Splitting on "," and checking each part is linear-time by
# construction — no backtracking possible.

def _looks_like_table_row(line: str) -> bool:
    """True if every comma-separated part of *line* is "key=value" with
    non-empty text on both sides of exactly one "=" sign."""
    parts = line.split(",")
    if len(parts) < 2:
        return False
    for part in parts:
        if part.count("=") != 1:
            return False
        key, _, value = part.partition("=")
        if not key.strip() or not value.strip():
            return False
    return True


def is_table_text(text: str, sample_lines: int = 20, threshold: float = 0.6) -> bool:
    lines = [l.strip() for l in text.splitlines() if l.strip()][:sample_lines]
    if not lines:
        return False
    matched = sum(1 for l in lines if _looks_like_table_row(l))
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
    id_prefix: str = "",
) -> List[Dict]:
    """Log-aware chunker; keeps stack traces whole within one chunk.
    Log entries are flat chunks — parent_id == chunk_id (self-referential).

    id_prefix: disambiguates chunk_id when chunk_document() sees more than
    one "page"-dict entry for the same page_num (see chunk_document()'s
    docstring — this is what parse_pdf()'s table extraction produces:
    a prose entry and a table entry both tagged page N).
    """
    entries = _split_log_entries(text)
    chunks: List[Dict] = []
    current_entries: List[str] = []
    current_tokens = 0
    chunk_index = 0

    def _make_log_chunk(entries_list: List[str], idx: int) -> Dict:
        body = "\n".join(entries_list)
        cid = f"{file_id}_p{page_num}{id_prefix}_c{idx}"
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
    id_prefix: str = "",
) -> List[Dict]:
    """
    Accumulate paragraphs into parent chunks ≤ PARENT_CHUNK_SIZE tokens.
    Returns a list of parent dicts with keys: parent_id, text, page.

    id_prefix: see chunk_log()'s docstring for why this exists.
    """
    parents: List[Dict] = []
    current_paras: List[str] = []
    current_tokens = 0
    parent_index = 0

    def _flush(paras: List[str], idx: int) -> Dict:
        body = "\n".join(paras)
        return {
            "parent_id":  f"{file_id}_p{page_num}{id_prefix}_parent{idx}",
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
    row_aware: bool = False,
    id_prefix: str = "",
) -> List[Dict]:
    """
    Slice a parent chunk into child chunks ≤ CHILD_CHUNK_SIZE tokens.
    Each child carries parent_id and parent_text so the LLM gets full context.

    row_aware=True (table-shaped content — see is_table_text()): splits on
    newlines only (never mid-row on ". "), joins child text back together
    with newlines (not spaces) so each row stays on its own readable line,
    and labels the section "Table" instead of just the page number. Used
    for both parse_csv() output and PDF-extracted tables — same "keep the
    whole thing joinable back to one row-per-line block" goal either way.

    id_prefix: see chunk_log()'s docstring for why this exists.
    """
    parent_text = parent["text"]
    parent_id   = parent["parent_id"]
    page_num    = parent["page"]

    if row_aware:
        lines = [l.strip() for l in parent_text.splitlines() if l.strip()]
        child_joiner = "\n"
        section = f"Page {page_num} | Table"
    else:
        # Split parent into sentences / lines for finer child boundaries
        lines = [l.strip() for l in parent_text.replace(". ", ".\n").splitlines() if l.strip()]
        child_joiner = " "
        section = f"Page {page_num}"

    if not lines:
        return []

    children: List[Dict] = []
    current_lines: List[str] = []
    current_tokens = 0
    child_index = child_start_index

    def _flush_child(lines_list: List[str], idx: int) -> Dict:
        body = child_joiner.join(lines_list)
        cid = f"{file_id}_p{page_num}{id_prefix}_c{idx}"
        return {
            "chunk_id": cid,
            "text": body,
            "metadata": {
                "doc_id":        file_id,
                "file_id":       file_id,
                "filename":      filename,
                "file_type":     file_type,
                "page":          page_num,
                "section":       section,
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
    id_prefix: str = "",
) -> List[Dict]:
    """
    Chunk a single page into parent-child pairs.

    If the page looks like a log file, delegates to chunk_log() (flat chunks).
    Otherwise produces a two-level hierarchy:
      1. Parent chunks (≤ PARENT_CHUNK_SIZE tokens) — rich LLM context.
      2. Child chunks (≤ CHILD_CHUNK_SIZE tokens)  — precise search units.
    Returns only child chunks; parent text is embedded in each child's metadata.

    Table-shaped text (is_table_text() — parse_csv() output or a
    PDF-extracted table) reuses this SAME parent-child machinery, just
    with rows standing in for paragraphs: a whole table (up to
    PARENT_CHUNK_SIZE) becomes one parent, so any child chunk matched by
    search still expands back to the FULL table via parent_text — the
    LLM sees every row, not just the one that happened to match, which
    matters for sum/lookup-style questions over the data.

    id_prefix: see chunk_log()'s docstring — chunk_document() supplies
    this when it sees more than one page-dict entry for the same page_num
    (parse_pdf() yields a prose entry AND a table entry both tagged with
    the PDF's real page number), so their chunk_ids don't collide.
    """
    uploaded_at = datetime.now(timezone.utc).isoformat()

    if is_log_file(page_text):
        return chunk_log(page_text, page_num, file_id, filename, file_type, uploaded_at, user_id, id_prefix)

    is_table = is_table_text(page_text)
    if is_table:
        paragraphs = [line.strip() for line in page_text.splitlines() if line.strip()]
    else:
        paragraphs = split_into_paragraphs(page_text)
    if not paragraphs:
        return []

    parents  = _build_parents(paragraphs, page_num, file_id, id_prefix)
    children: List[Dict] = []
    child_counter = 0

    for parent in parents:
        parent_children = _build_children(
            parent, file_id, filename, file_type, uploaded_at, user_id, child_counter,
            row_aware=is_table, id_prefix=id_prefix,
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

    A page NUMBER can appear more than once in *pages* — parse_pdf() yields
    a plain-text entry and a separate table entry for the same PDF page
    when it contains a table. Each repeat past the first gets a distinct
    id_prefix so chunk_page()'s chunk_ids don't collide (two chunk_page()
    calls for "page 1" would otherwise both start counting from
    "..._p1_c0").
    """
    all_chunks: List[Dict] = []
    seen_hashes: Set[str]  = set()
    page_occurrences: Dict[int, int] = {}

    for page in pages:
        page_num = page["page"]
        occurrence = page_occurrences.get(page_num, 0)
        page_occurrences[page_num] = occurrence + 1
        id_prefix = f"_t{occurrence}" if occurrence > 0 else ""

        page_chunks = chunk_page(
            page_text=page["text"],
            page_num=page_num,
            file_id=file_id,
            filename=filename,
            file_type=file_type,
            user_id=user_id,
            id_prefix=id_prefix,
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
