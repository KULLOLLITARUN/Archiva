"""Tests for main.py's _build_sources() / _cited_chunks() — the source list
returned with an answer only names files the answer actually cites.

Pure functions; importing main.py doesn't touch the database (see
test_suggestions_sampling.py)."""

import main

INVOICE = "A4-S1-GST-Invoice-Format_page-0001-scaled.pdf"


def _chunk(filename, page=1, score=0.5):
    return {"text": f"text of {filename}", "score": score,
            "metadata": {"filename": filename, "page": page}}


CHUNKS = [_chunk(INVOICE), _chunk("10.pdf", 9), _chunk("Prompt Engineering for LLMs.pdf", 199)]


def _filenames(answer, chunks=CHUNKS):
    return [s["filename"] for s in main._build_sources(chunks, answer)]


def test_only_cited_file_is_returned():
    # The real case: an invoice answer listed two unrelated PDFs as sources.
    answer = f"The bank is ICICI and the UPI ID is ifox@icici. [Source: {INVOICE}, page 1]"
    assert _filenames(answer) == [INVOICE]


def test_citation_with_look_alike_hyphens_still_matches():
    # Models copy filenames with non-breaking hyphens (U+2011).
    answer = "Total: 4,490.00 [Source: A4‑S1‑GST‑Invoice‑Format_page‑0001‑scaled.pdf | Page 1]"
    assert _filenames(answer) == [INVOICE]


def test_several_cited_files_are_all_kept():
    answer = f"A [Source: {INVOICE}, page 1] and B 【Source: 10.pdf, page 9】"
    assert _filenames(answer) == [INVOICE, "10.pdf"]


def test_every_chunk_of_a_cited_file_is_kept():
    chunks = [_chunk(INVOICE, 1), _chunk(INVOICE, 2), _chunk("10.pdf", 9)]
    assert _filenames(f"x [Source: {INVOICE}, page 1]", chunks) == [INVOICE, INVOICE]


def test_answer_without_citations_keeps_all_sources():
    assert _filenames("An answer with no inline citation.") == [c["metadata"]["filename"] for c in CHUNKS]


def test_unmatchable_citation_keeps_all_sources():
    # Never drop every source because the model garbled a filename.
    assert len(_filenames("x [Source: some-other-file.pdf, page 2]")) == len(CHUNKS)


def test_no_answer_argument_keeps_old_behaviour():
    assert len(main._build_sources(CHUNKS)) == len(CHUNKS)
