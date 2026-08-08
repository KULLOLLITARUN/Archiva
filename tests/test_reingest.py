"""Tests for ingestion/reingest.py — persisting uploaded files to disk and
replaying parse+chunk against them (the REINGEST healing-action follow-through)."""

import os

import pytest

import retrieval.store as store_module
from ingestion.chunker import chunk_document
from ingestion.parser import compute_hash, parse_file
from ingestion.reingest import (
    persisted_path,
    refresh_all_from_disk,
    reprocess_file,
    save_uploaded_file,
)
from retrieval.store import MultiDocStore

SAMPLE = b"Alpha bravo charlie delta echo foxtrot golf hotel india juliet."


@pytest.fixture(autouse=True)
def _no_background_embeddings(monkeypatch):
    # Keep these tests hermetic/fast — don't spin up the real embedding
    # thread (which would try to load sentence-transformers) on every
    # store.add_file() call.
    monkeypatch.setattr(store_module.MultiDocStore, "_trigger_embedding_precompute", lambda self, chunks: None)


def test_save_uploaded_file_writes_bytes_to_deterministic_path(tmp_path):
    path = save_uploaded_file(SAMPLE, "file123", "txt", str(tmp_path))
    assert path == persisted_path(str(tmp_path), "file123", "txt")
    assert os.path.exists(path)
    with open(path, "rb") as f:
        assert f.read() == SAMPLE


def test_reprocess_file_reparses_and_rechunks(tmp_path):
    path = save_uploaded_file(SAMPLE, "f1", "txt", str(tmp_path))
    chunks = reprocess_file(path, "f1", "doc.txt", "txt")
    assert len(chunks) >= 1
    assert chunks[0]["metadata"]["file_id"] == "f1"
    assert chunks[0]["metadata"]["filename"] == "doc.txt"


def _seed_store(file_id: str, filename: str, content: bytes) -> MultiDocStore:
    store = MultiDocStore()
    pages = parse_file(content, filename)
    chunks = chunk_document(pages=pages, file_id=file_id, filename=filename, file_type="txt")
    store.add_file(
        file_id=file_id, filename=filename,
        content_hash=compute_hash(content), chunks=chunks, file_type="txt",
    )
    return store


def test_refresh_all_from_disk_replaces_chunks_for_persisted_files(tmp_path):
    file_id = "f1"
    save_uploaded_file(SAMPLE, file_id, "txt", str(tmp_path))
    store = _seed_store(file_id, "doc.txt", SAMPLE)

    summary = refresh_all_from_disk(store, str(tmp_path))

    assert summary == {"refreshed": ["doc.txt"], "skipped": []}
    assert file_id in store.files
    assert store.total_chunks() >= 1


def test_refresh_all_from_disk_skips_files_with_no_persisted_copy(tmp_path):
    store = _seed_store("f2", "doc.txt", SAMPLE)  # never persisted to tmp_path

    summary = refresh_all_from_disk(store, str(tmp_path))

    assert summary == {"refreshed": [], "skipped": ["doc.txt"]}
    # Original file/chunks are untouched, not deleted, on a skip.
    assert "f2" in store.files
    assert store.total_chunks() >= 1


def test_refresh_all_from_disk_handles_mixed_persisted_and_missing_files(tmp_path):
    save_uploaded_file(SAMPLE, "f1", "txt", str(tmp_path))
    store = _seed_store("f1", "persisted.txt", SAMPLE)

    other_content = b"Kilo lima mike november oscar papa quebec romeo sierra."
    pages = parse_file(other_content, "missing.txt")
    chunks = chunk_document(pages=pages, file_id="f3", filename="missing.txt", file_type="txt")
    store.add_file(
        file_id="f3", filename="missing.txt",
        content_hash=compute_hash(other_content), chunks=chunks, file_type="txt",
    )

    summary = refresh_all_from_disk(store, str(tmp_path))

    assert summary["refreshed"] == ["persisted.txt"]
    assert summary["skipped"] == ["missing.txt"]
