"""Tests for the new ingestion/parser.py formats: Markdown, CSV, HTML."""

from ingestion.parser import SUPPORTED_EXTENSIONS, parse_csv, parse_file, parse_html, parse_md


def test_supported_extensions_includes_new_formats():
    assert {".md", ".csv", ".html", ".htm"} <= SUPPORTED_EXTENSIONS


def test_parse_md_returns_plain_text_page():
    content = b"# Heading\n\nSome **bold** text about the roadmap."
    pages = parse_md(content, "notes.md")
    assert len(pages) == 1
    assert "roadmap" in pages[0]["text"]


def test_parse_csv_renders_column_value_pairs():
    content = b"name,age,city\nAlice,30,Paris\nBob,25,Berlin\n"
    pages = parse_csv(content, "people.csv")
    text = pages[0]["text"]
    assert "name=Alice" in text
    assert "age=30" in text
    assert "city=Paris" in text
    assert "name=Bob" in text


def test_parse_csv_handles_empty_content():
    pages = parse_csv(b"", "empty.csv")
    assert pages == [{"page": 1, "text": ""}]


def test_parse_html_strips_tags_scripts_and_styles():
    content = b"""
    <html><head><style>body { color: red; }</style></head>
    <body>
      <script>alert('hi')</script>
      <h1>Welcome</h1>
      <p>The roadmap targets Q3 2026.</p>
    </body></html>
    """
    pages = parse_html(content, "page.html")
    text = pages[0]["text"]
    assert "roadmap" in text.lower()
    assert "Welcome" in text
    assert "alert" not in text
    assert "color: red" not in text


def test_parse_file_dispatches_to_new_parsers_by_extension():
    md_pages = parse_file(b"# Title\ncontent here", "readme.md")
    assert "content here" in md_pages[0]["text"]

    csv_pages = parse_file(b"a,b\n1,2\n", "data.csv")
    assert "a=1" in csv_pages[0]["text"]

    html_pages = parse_file(b"<p>hello world</p>", "page.htm")
    assert "hello world" in html_pages[0]["text"]


def test_parse_file_raises_on_unsupported_extension():
    import pytest
    with pytest.raises(ValueError):
        parse_file(b"data", "archive.zip")
