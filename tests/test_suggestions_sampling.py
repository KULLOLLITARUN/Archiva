"""Tests for main.py's _sample_for_suggestions() — the chunk sampler that
seeds the /suggestions LLM prompt.

Pure function, no Postgres/LLM/ML models involved: importing main.py itself
doesn't touch the database (init_db() only runs inside the app lifespan),
so this doesn't need the Postgres-gated setup test_api_integration.py uses."""

import main


def _chunk(file_id, index, filename="doc.txt"):
    return {
        "text": f"chunk {index} of {file_id}",
        "metadata": {"file_id": file_id, "filename": filename, "page": index, "chunk_index": index},
    }


def _file_chunks(file_id, n, filename="doc.txt"):
    return [_chunk(file_id, i, filename) for i in range(n)]


def test_empty_input_returns_empty_list():
    assert main._sample_for_suggestions([], total=30) == []


def test_returns_all_chunks_when_under_the_total_budget():
    chunks = _file_chunks("f1", 5)
    result = main._sample_for_suggestions(chunks, total=30)
    assert len(result) == 5


def test_caps_at_total_for_a_single_large_file():
    chunks = _file_chunks("f1", 200)
    result = main._sample_for_suggestions(chunks, total=30)
    assert len(result) == 30


def test_single_large_file_is_sampled_across_its_whole_length_not_just_the_start():
    # 200 chunks in one file, only room for 30 in the sample -> the old
    # front-to-back version would return chunks 0..29 (just the intro).
    # The fix should spread picks across the full 0..199 range.
    chunks = _file_chunks("f1", 200)
    result = main._sample_for_suggestions(chunks, total=30)

    indices = [c["metadata"]["chunk_index"] for c in result]
    assert max(indices) > 150          # reaches well into the back half
    assert min(indices) == 0           # still includes the opening
    assert indices == sorted(indices)  # stays in document order


def test_balances_across_multiple_files():
    chunks = _file_chunks("f1", 100) + _file_chunks("f2", 100)
    result = main._sample_for_suggestions(chunks, total=30)

    file_ids = [c["metadata"]["file_id"] for c in result]
    assert file_ids.count("f1") == file_ids.count("f2") == 15


def test_small_file_is_not_padded_beyond_its_own_length():
    # f1 has only 3 chunks total -> all 3 should appear, not be
    # downsampled just because per_file budget (15) exceeds its size.
    chunks = _file_chunks("f1", 3) + _file_chunks("f2", 100)
    result = main._sample_for_suggestions(chunks, total=30)

    file_ids = [c["metadata"]["file_id"] for c in result]
    assert file_ids.count("f1") == 3


def test_many_files_still_respects_total_budget():
    chunks = []
    for i in range(50):
        chunks += _file_chunks(f"f{i}", 3)
    result = main._sample_for_suggestions(chunks, total=30)
    assert len(result) <= 30
