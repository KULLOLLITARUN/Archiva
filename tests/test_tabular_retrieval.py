"""
Tests for structured/tabular retrieval: table-shaped content detection
(ingestion/chunker.py), table rendering (ingestion/parser.py), and the
real pypdf/pdfplumber/python-docx extraction paths against small real
fixture files in tests/fixtures/.

Design being tested: a table becomes ONE parent (up to PARENT_CHUNK_SIZE)
so any child chunk matched by search still expands back to the FULL table
via parent_text — see ingestion/chunker.py's chunk_page() docstring.
"""

from pathlib import Path

from ingestion.chunker import chunk_document, chunk_page, is_table_text
from ingestion.parser import _render_table_rows, parse_docx, parse_pdf

FIXTURES_DIR = Path(__file__).parent / "fixtures"

TABLE_TEXT = (
    "Quarter=Q1, Revenue=120000, Costs=80000\n"
    "Quarter=Q2, Revenue=135000, Costs=85000\n"
    "Quarter=Q3, Revenue=150000, Costs=90000\n"
    "Quarter=Q4, Revenue=160000, Costs=95000"
)

PROSE_TEXT = (
    "The roadmap targets Q3 2026 for the mobile app launch, with European "
    "expansion following in Q4 once GDPR compliance review is complete."
)


# ── is_table_text() ─────────────────────────────────────────────────────────────

def test_is_table_text_detects_key_value_rows():
    assert is_table_text(TABLE_TEXT) is True


def test_is_table_text_rejects_prose():
    assert is_table_text(PROSE_TEXT) is False


def test_is_table_text_rejects_prose_containing_equals_signs():
    # "x=y" appearing incidentally in prose must not trigger table
    # detection — needs the comma-separated multi-field row shape.
    text = "The formula is speed = distance / time, which is a basic identity."
    assert is_table_text(text) is False


def test_is_table_text_requires_majority_of_lines_to_match():
    mixed = "Introduction paragraph with no structure at all.\n" + TABLE_TEXT
    # 1 prose line + 4 table lines = 80% table lines, above the 0.6 threshold.
    assert is_table_text(mixed) is True

    mostly_prose = "\n".join(["Just a sentence with no equals signs."] * 4) + "\nQuarter=Q1, Revenue=1"
    assert is_table_text(mostly_prose) is False


def test_is_table_text_empty_string_is_false():
    assert is_table_text("") is False


# ── _render_table_rows() ─────────────────────────────────────────────────────────

def test_render_table_rows_uses_header_as_field_names():
    rows = [
        ["Quarter", "Revenue", "Costs"],
        ["Q1", "120000", "80000"],
        ["Q2", "135000", "85000"],
    ]
    text = _render_table_rows(rows)
    assert text == "Quarter=Q1, Revenue=120000, Costs=80000\nQuarter=Q2, Revenue=135000, Costs=85000"


def test_render_table_rows_skips_empty_cells():
    rows = [["Quarter", "Notes"], ["Q1", ""]]
    text = _render_table_rows(rows)
    assert text == "Quarter=Q1"


def test_render_table_rows_handles_none_cells():
    rows = [["Quarter", "Notes"], ["Q1", None]]
    text = _render_table_rows(rows)
    assert text == "Quarter=Q1"


def test_render_table_rows_returns_empty_for_header_only():
    assert _render_table_rows([["Quarter", "Revenue"]]) == ""


def test_render_table_rows_returns_empty_for_no_rows():
    assert _render_table_rows([]) == ""


# ── chunk_page() routes table-shaped text through the row-aware path ───────────

def test_chunk_page_keeps_whole_table_as_one_parent():
    chunks = chunk_page(TABLE_TEXT, page_num=1, file_id="f1", filename="budget.csv", file_type="csv")

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["metadata"]["section"] == "Page 1 | Table"
    # All four rows present in both the child text and parent_text -
    # the table wasn't fragmented into scattered row chunks.
    for q in ("Q1", "Q2", "Q3", "Q4"):
        assert f"Quarter={q}" in chunk["text"]
        assert f"Quarter={q}" in chunk["metadata"]["parent_text"]


def test_chunk_page_prose_is_unaffected_by_table_routing():
    chunks = chunk_page(PROSE_TEXT, page_num=1, file_id="f1", filename="doc.txt", file_type="txt")
    assert len(chunks) == 1
    assert chunks[0]["metadata"]["section"] == "Page 1"  # not "| Table"


# ── chunk_document() disambiguates chunk_ids across repeated page numbers ──────

def test_chunk_document_disambiguates_repeated_page_numbers():
    # Simulates parse_pdf()'s output for a page with both prose and a table.
    pages = [
        {"page": 1, "text": PROSE_TEXT},
        {"page": 1, "text": TABLE_TEXT},
    ]
    chunks = chunk_document(pages=pages, file_id="f1", filename="budget.pdf", file_type="pdf")

    chunk_ids = [c["chunk_id"] for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids)), f"duplicate chunk_ids: {chunk_ids}"

    sections = {c["metadata"]["section"] for c in chunks}
    assert "Page 1" in sections
    assert "Page 1 | Table" in sections


def test_chunk_document_three_way_repeat_stays_unique():
    pages = [
        {"page": 1, "text": PROSE_TEXT},
        {"page": 1, "text": TABLE_TEXT},
        {"page": 1, "text": "Quarter=Q5, Revenue=1, Costs=1"},
    ]
    chunks = chunk_document(pages=pages, file_id="f1", filename="budget.pdf", file_type="pdf")
    chunk_ids = [c["chunk_id"] for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))


# ── Real-file extraction (pypdf/pdfplumber, python-docx) ────────────────────────

def test_parse_pdf_extracts_both_prose_and_structured_table():
    content = (FIXTURES_DIR / "quarterly_budget.pdf").read_bytes()
    pages = list(parse_pdf(content, "quarterly_budget.pdf"))

    assert len(pages) == 2
    prose_page, table_page = pages
    assert "finance department" in prose_page["text"]
    assert is_table_text(table_page["text"])
    for q in ("Q1", "Q2", "Q3", "Q4"):
        assert f"Quarter={q}" in table_page["text"]


def test_parse_pdf_table_survives_full_chunking_as_one_unit():
    content = (FIXTURES_DIR / "quarterly_budget.pdf").read_bytes()
    pages = parse_pdf(content, "quarterly_budget.pdf")
    chunks = chunk_document(pages=pages, file_id="f1", filename="quarterly_budget.pdf", file_type="pdf")

    table_chunks = [c for c in chunks if c["metadata"]["section"].endswith("| Table")]
    assert len(table_chunks) == 1
    for q in ("Q1", "Q2", "Q3", "Q4"):
        assert f"Quarter={q}" in table_chunks[0]["text"]


def test_parse_docx_extracts_both_prose_and_structured_table():
    content = (FIXTURES_DIR / "quarterly_budget.docx").read_bytes()
    pages = parse_docx(content, "quarterly_budget.docx")

    assert len(pages) == 2
    prose_page, table_page = pages
    assert "finance department" in prose_page["text"]
    assert is_table_text(table_page["text"])
    for q in ("Q1", "Q2", "Q3", "Q4"):
        assert f"Quarter={q}" in table_page["text"]


# ── Graceful degradation when pdfplumber is unavailable/fails ──────────────────

def test_parse_pdf_still_works_if_pdfplumber_import_fails(monkeypatch):
    import sys
    # Setting a module to None in sys.modules makes the import system raise
    # ImportError for it — targeted at "pdfplumber" only, unlike patching
    # __import__ globally which risks intercepting unrelated imports too.
    monkeypatch.setitem(sys.modules, "pdfplumber", None)

    content = (FIXTURES_DIR / "quarterly_budget.pdf").read_bytes()
    pages = list(parse_pdf(content, "quarterly_budget.pdf"))

    # No table entry, but plain-text extraction still works fine.
    assert len(pages) == 1
    assert "finance department" in pages[0]["text"]
