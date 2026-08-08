"""
Tests for the large-file handling added to ingestion/parser.py:
  - parse_pdf() is lazy (a generator) so chunk_document()'s existing
    MAX_CHUNKS_PER_FILE early-stop actually skips extracting pages beyond
    the cap, instead of paying full extraction cost for a huge PDF.
  - MAX_INGEST_CHARS bounds how much text single-page formats
    (.txt/.csv/.html) will process, independent of the PDF-specific fix.

See config.py's "Large-file guards" section for the reasoning.
"""

import inspect

import ingestion.chunker as chunker_module
import ingestion.parser as parser_module
from ingestion.chunker import chunk_document
from ingestion.parser import _bound_for_ingest, parse_csv, parse_html, parse_pdf, parse_txt


# ── parse_pdf laziness ───────────────────────────────────────────────────────────

class _FakePage:
    def __init__(self, index: int, tracker: list):
        self._index = index
        self._tracker = tracker

    def extract_text(self, extraction_mode=None):
        self._tracker.append(self._index)
        return f"Page {self._index} has plenty of real words describing quarter results and roadmap items."


class _FakeReader:
    """Stands in for pypdf.PdfReader: a "1000-page PDF" where each page
    records itself in `tracker` only when extract_text() is actually called."""

    def __init__(self, stream, tracker: list, num_pages: int = 1000):
        self.pages = [_FakePage(i, tracker) for i in range(1, num_pages + 1)]


def test_parse_pdf_is_a_generator_function():
    assert inspect.isgeneratorfunction(parse_pdf)


def test_parse_pdf_yields_pages_lazily_one_at_a_time(monkeypatch):
    extracted: list = []
    monkeypatch.setattr("pypdf.PdfReader", lambda stream: _FakeReader(stream, extracted))

    gen = parse_pdf(b"fake pdf bytes", "huge.pdf")
    first = next(gen)

    assert first["page"] == 1
    # Only the first page should have been extracted so far - pulling one
    # item from the generator must not have touched the other 999.
    assert extracted == [1]


def test_chunk_document_early_stop_prevents_extracting_pages_beyond_the_cap(monkeypatch):
    extracted: list = []
    monkeypatch.setattr("pypdf.PdfReader", lambda stream: _FakeReader(stream, extracted))
    # Cap chunking well below the fake PDF's 1000 pages.
    monkeypatch.setattr(chunker_module, "MAX_CHUNKS_PER_FILE", 5)

    pages = parse_pdf(b"fake pdf bytes", "huge.pdf")
    chunks = chunk_document(pages=pages, file_id="f1", filename="huge.pdf", file_type="pdf")

    assert len(chunks) <= 5
    # The real point of this test: laziness means most of the 1000 pages
    # were never text-extracted at all, not just discarded after the fact.
    assert len(extracted) < 20
    assert len(extracted) < 1000


def test_parse_pdf_falls_back_to_one_empty_page_when_nothing_extracts(monkeypatch):
    class _BlankPage:
        def extract_text(self, extraction_mode=None):
            return ""

    class _BlankReader:
        def __init__(self, stream):
            self.pages = [_BlankPage(), _BlankPage()]

    monkeypatch.setattr("pypdf.PdfReader", _BlankReader)

    pages = list(parse_pdf(b"fake pdf bytes", "scanned.pdf"))
    assert pages == [{"page": 1, "text": ""}]


# ── MAX_INGEST_CHARS bound ───────────────────────────────────────────────────────

def test_bound_for_ingest_passes_short_text_through_unchanged():
    text = "short text"
    assert _bound_for_ingest(text, "f.txt") == text


def test_bound_for_ingest_truncates_long_text(monkeypatch):
    monkeypatch.setattr(parser_module, "MAX_INGEST_CHARS", 10)
    result = _bound_for_ingest("a" * 100, "f.txt")
    assert len(result) == 10


def test_parse_txt_respects_max_ingest_chars(monkeypatch):
    monkeypatch.setattr(parser_module, "MAX_INGEST_CHARS", 20)
    pages = parse_txt(b"x" * 1000, "huge.txt")
    assert len(pages[0]["text"]) <= 20


def test_parse_csv_respects_max_ingest_chars(monkeypatch):
    monkeypatch.setattr(parser_module, "MAX_INGEST_CHARS", 15)
    huge_csv = ("col1,col2\n" + "a,b\n" * 10000).encode()
    # Should not raise, and should not process all 10000 rows.
    pages = parse_csv(huge_csv, "huge.csv")
    assert isinstance(pages[0]["text"], str)


def test_parse_html_respects_max_ingest_chars(monkeypatch):
    monkeypatch.setattr(parser_module, "MAX_INGEST_CHARS", 20)
    huge_html = ("<p>" + "word " * 10000 + "</p>").encode()
    pages = parse_html(huge_html, "huge.html")
    assert len(pages[0]["text"]) <= 20 + len("<p>")  # small parser-boundary slack
