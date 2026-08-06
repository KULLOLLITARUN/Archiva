"""
ingestion/chunker.py — Text chunker with semantic deduplication.

Upgrade (Part 3):
  - SHA-256 content hash stored in every chunk's metadata["content_hash"].
  - chunk_document() maintains a seen-hash set and skips duplicate chunks
    within the same document (prevents redundant ingestion).
  - metadata["doc_id"], metadata["date"] fields added for clarity.
  - Paragraph + log-aware chunking strategy retained.

Fix #5:  No "embedding": [] field (BM25 needs no embeddings).
Fix #13: Log-file detection and stack-trace-aware chunking.
"""

import hashlib
import re
from datetime import datetime, timezone
from typing import Dict, List, Set

from config import CHUNK_SIZE, CHUNK_OVERLAP, MAX_CHUNKS_PER_FILE


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
    """Log-aware chunker; keeps stack traces whole within one chunk."""
    entries = _split_log_entries(text)
    chunks: List[Dict] = []
    current_entries: List[str] = []
    current_tokens = 0
    chunk_index = 0

    def _make_log_chunk(entries_list: List[str], idx: int) -> Dict:
        body = "\n".join(entries_list)
        return {
            "chunk_id": f"{file_id}_p{page_num}_c{idx}",
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
                "user_id":       user_id,      # ← tenant isolation key
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


# ── Paragraph chunking ────────────────────────────────────────────────────────

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


def chunk_page(
    page_text: str,
    page_num: int,
    file_id: str,
    filename: str,
    file_type: str,
    user_id: str = "",
) -> List[Dict]:
    """
    Chunk a single page of text.

    If the page looks like a log file, delegates to chunk_log().
    Otherwise uses paragraph-accumulation with overlap.
    SHA-256 content_hash added to every chunk's metadata.
    """
    uploaded_at = datetime.now(timezone.utc).isoformat()

    if is_log_file(page_text):
        return chunk_log(page_text, page_num, file_id, filename, file_type, uploaded_at, user_id)

    paragraphs   = split_into_paragraphs(page_text)
    chunks: List[Dict] = []
    current_paras: List[str] = []
    current_tokens = 0
    chunk_index = 0

    def make_chunk(paras: List[str], idx: int) -> Dict:
        text = "\n".join(paras)
        return {
            "chunk_id": f"{file_id}_p{page_num}_c{idx}",
            "text": text,
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
                "content_hash":  content_hash(text),
                "user_id":       user_id,      # ← tenant isolation key
            },
        }

    for para in paragraphs:
        para_tokens = estimate_tokens(para)

        if current_tokens + para_tokens > CHUNK_SIZE and current_paras:
            chunks.append(make_chunk(current_paras, chunk_index))
            chunk_index += 1

            overlap_paras: List[str] = []
            overlap_tokens = 0
            for prev_para in reversed(current_paras):
                prev_tokens = estimate_tokens(prev_para)
                overlap_paras.insert(0, prev_para)
                overlap_tokens += prev_tokens
                if overlap_tokens >= CHUNK_OVERLAP:
                    break

            current_paras  = overlap_paras
            current_tokens = overlap_tokens

        current_paras.append(para)
        current_tokens += para_tokens

    if current_paras:
        chunks.append(make_chunk(current_paras, chunk_index))

    return chunks


# ── Document-level entry point ────────────────────────────────────────────────

def chunk_document(
    pages: List[Dict],
    file_id: str,
    filename: str,
    file_type: str,
    user_id: str = "",
) -> List[Dict]:
    """
    Chunk all pages of a document.

    Enforces MAX_CHUNKS_PER_FILE cap.
    Part 3.2: Deduplicates chunks by SHA-256 hash within this document.
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
                # Skip exact duplicate chunk within this document
                continue
            if h:
                seen_hashes.add(h)
            all_chunks.append(chunk)

            if len(all_chunks) >= MAX_CHUNKS_PER_FILE:
                return all_chunks

    return all_chunks
