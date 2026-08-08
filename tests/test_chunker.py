"""Tests for ingestion/chunker.py — token estimation, content hashing,
log-file detection, and the parent-child chunk_document() pipeline."""

from ingestion.chunker import (
    chunk_document,
    content_hash,
    estimate_tokens,
    is_log_file,
)

PROSE_TEXT = (
    "Alpha bravo charlie delta echo foxtrot golf hotel india juliet "
    "kilo lima mike november oscar papa quebec romeo sierra tango."
)

LOG_TEXT = "\n".join([
    "2024-01-01 10:00:00 INFO Starting service",
    "2024-01-01 10:00:01 ERROR Connection failed",
    "2024-01-01 10:00:02 INFO Retrying connection",
])


def test_estimate_tokens_is_chars_over_four():
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 40) == 10
    assert estimate_tokens("") == 0


def test_content_hash_is_deterministic_and_distinguishes_text():
    h1 = content_hash("hello world")
    h2 = content_hash("hello world")
    h3 = content_hash("hello there")
    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 64  # sha256 hex digest


def test_is_log_file_detects_timestamped_lines():
    assert is_log_file(LOG_TEXT) is True


def test_is_log_file_rejects_prose():
    assert is_log_file(PROSE_TEXT) is False


def test_chunk_document_produces_linked_parent_child_metadata():
    pages = [{"text": PROSE_TEXT, "page": 1}]
    chunks = chunk_document(pages=pages, file_id="f1", filename="doc.txt", file_type="txt")

    assert len(chunks) >= 1
    for chunk in chunks:
        meta = chunk["metadata"]
        assert meta["file_id"] == "f1"
        assert meta["filename"] == "doc.txt"
        assert meta["parent_id"]
        assert meta["parent_text"]  # full parent context carried for the LLM
        assert chunk["text"] in meta["parent_text"] or meta["parent_text"].startswith(chunk["text"][:10])


def test_chunk_document_dedupes_identical_content_across_pages():
    pages = [
        {"text": PROSE_TEXT, "page": 1},
        {"text": PROSE_TEXT, "page": 2},  # exact duplicate of page 1
    ]
    chunks = chunk_document(pages=pages, file_id="f1", filename="doc.txt", file_type="txt")

    # Identical paragraph text on both pages hashes the same -> second is deduped.
    hashes = [c["metadata"]["content_hash"] for c in chunks]
    assert len(hashes) == len(set(hashes))


def test_chunk_document_log_chunks_are_self_referential_parents():
    pages = [{"text": LOG_TEXT, "page": 1}]
    chunks = chunk_document(pages=pages, file_id="f1", filename="app.log", file_type="log")

    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk["metadata"]["parent_id"] == chunk["chunk_id"]
