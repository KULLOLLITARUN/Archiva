"""Tests for ingestion/chunker.py — token estimation, content hashing,
log-file detection, and the parent-child chunk_document() pipeline."""

from config import PARENT_CHUNK_SIZE
from ingestion.chunker import (
    _split_into_sentences,
    chunk_document,
    content_hash,
    estimate_tokens,
    is_log_file,
    split_into_paragraphs,
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


def test_split_into_sentences_keeps_abbreviations_intact():
    text = "Dr. Smith noted i.e. growth continued. Prof. Lee agreed."
    sentences = _split_into_sentences(text)
    assert sentences == [
        "Dr. Smith noted i.e. growth continued.",
        "Prof. Lee agreed.",
    ]


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


def test_split_into_paragraphs_bounds_unbroken_blob_by_sentence():
    # No "\n\n" or "\n" at all - common for OCR output / poorly formatted
    # PDF text extraction. Without a fallback, this single "paragraph" would
    # become one oversized parent chunk whose full text is embedded as LLM
    # context for every matching child.
    sentence = "This is a sentence about important topics in the document. "
    blob = sentence * 400  # no newlines anywhere, ~6000 tokens as one unit

    paragraphs = split_into_paragraphs(blob)

    assert len(paragraphs) > 1
    for p in paragraphs:
        assert estimate_tokens(p) <= PARENT_CHUNK_SIZE


def test_split_into_paragraphs_hard_slices_unbroken_text_with_no_punctuation():
    blob = "a" * (PARENT_CHUNK_SIZE * 4 * 3)  # no spaces, no punctuation at all

    paragraphs = split_into_paragraphs(blob)

    assert len(paragraphs) == 3
    for p in paragraphs:
        assert estimate_tokens(p) <= PARENT_CHUNK_SIZE


def test_chunk_document_bounds_parent_text_for_unbroken_input():
    sentence = "This is a sentence about important topics in the document. "
    blob = sentence * 400
    pages = [{"text": blob, "page": 1}]

    chunks = chunk_document(pages=pages, file_id="f1", filename="ocr.txt", file_type="txt")

    # +100 tolerance: estimate_tokens() is a soft chars/4 heuristic, and
    # summing per-sentence estimates during accumulation vs. re-estimating
    # the final "\n"-joined string drifts slightly - this asserts the fix
    # (bounded to roughly PARENT_CHUNK_SIZE), not exact-token precision.
    for chunk in chunks:
        assert estimate_tokens(chunk["metadata"]["parent_text"]) <= PARENT_CHUNK_SIZE + 100


def test_chunk_document_log_chunks_are_self_referential_parents():
    pages = [{"text": LOG_TEXT, "page": 1}]
    chunks = chunk_document(pages=pages, file_id="f1", filename="app.log", file_type="log")

    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk["metadata"]["parent_id"] == chunk["chunk_id"]
