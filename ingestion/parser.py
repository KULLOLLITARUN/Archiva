"""
ingestion/parser.py — Document parser supporting TXT, PDF, DOCX, Markdown,
CSV, and HTML.

Fix #9: Added parse_pdf() (via pypdf) and parse_docx() (via python-docx).
        parse_file() dispatcher routes by file extension.
        Returns the same [{page, text}] structure regardless of format.

Format expansion: Added parse_md() / parse_csv() / parse_html() — all
stdlib-only, no new dependencies. Scanned/image PDFs still need OCR
(out of scope here — see README for the manual-conversion workaround).
"""

import csv
import hashlib
import io
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Dict


def compute_hash(content: bytes) -> str:
    """Compute MD5 hash of file content bytes."""
    return hashlib.md5(content).hexdigest()


def clean_text(text: str) -> str:
    """Remove null bytes, collapse whitespace, and strip."""
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── TXT ───────────────────────────────────────────────────────────────────────

def parse_txt(content: bytes, filename: str) -> List[Dict]:
    """
    Parse a TXT file's raw bytes into a list of page dicts.
    Always returns a single page.
    """
    raw = content.decode("utf-8", errors="ignore")
    cleaned = clean_text(raw)
    return [{"page": 1, "text": cleaned}]


# ── PDF ───────────────────────────────────────────────────────────────────────

def parse_pdf(content: bytes, filename: str) -> List[Dict]:
    """
    Parse a PDF file into per-page dicts using pypdf.
    Each PDF page becomes one element with its 1-based page number.
    """
    try:
        from pypdf import PdfReader
        import io
    except ImportError as exc:
        raise ImportError(
            "pypdf is required for PDF support. Run: pip install pypdf"
        ) from exc

    reader = PdfReader(io.BytesIO(content))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        raw = ""
        try:
            raw = page.extract_text() or ""
        except Exception:
            pass
        if not raw.strip():
            try:
                raw = page.extract_text(extraction_mode="layout") or ""
            except Exception:
                pass
        cleaned = clean_text(raw)
        if cleaned:
            pages.append({"page": i, "text": cleaned})

    return pages if pages else [{"page": 1, "text": ""}]


# ── DOCX ──────────────────────────────────────────────────────────────────────

def parse_docx(content: bytes, filename: str) -> List[Dict]:
    """
    Parse a DOCX file from raw bytes into a single-page dict using python-docx.
    Extracts text from paragraphs and tables.
    """
    try:
        from docx import Document
        import io
    except ImportError as exc:
        raise ImportError(
            "python-docx is required for DOCX support. Run: pip install python-docx"
        ) from exc

    doc = Document(io.BytesIO(content))
    elements = []
    for p in doc.paragraphs:
        if p.text.strip():
            elements.append(p.text)
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                elements.append(row_text)

    raw = "\n\n".join(elements)
    cleaned = clean_text(raw)
    return [{"page": 1, "text": cleaned}]



# ── Markdown ──────────────────────────────────────────────────────────────────

def parse_md(content: bytes, filename: str) -> List[Dict]:
    """
    Parse a Markdown file as plain text. Markup characters (#, *, -, etc.)
    are left in place — they're harmless extra tokens for BM25/dense search
    and stripping them isn't worth a parser dependency.
    """
    return parse_txt(content, filename)


# ── CSV ───────────────────────────────────────────────────────────────────────

def parse_csv(content: bytes, filename: str) -> List[Dict]:
    """
    Parse a CSV file into a single readable page: each data row rendered as
    "column=value" pairs using the header row as field names, so BM25/dense
    search can match a cell value together with the column it came from
    (rather than as bare comma-separated text).
    """
    raw = content.decode("utf-8", errors="ignore")
    rows = list(csv.reader(io.StringIO(raw)))
    if not rows:
        return [{"page": 1, "text": ""}]

    header, *data_rows = rows
    lines = []
    for row in data_rows:
        pairs = [f"{h.strip()}={v.strip()}" for h, v in zip(header, row) if h.strip()]
        if pairs:
            lines.append(", ".join(pairs))

    cleaned = clean_text("\n".join(lines))
    return [{"page": 1, "text": cleaned}]


# ── HTML ──────────────────────────────────────────────────────────────────────

class _TextExtractor(HTMLParser):
    """Minimal HTML-to-text extractor: strips tags/scripts/styles, keeps
    visible text. Stdlib-only — no BeautifulSoup dependency."""

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.parts: List[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in ("script", "style"):
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style") and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data.strip():
            self.parts.append(data.strip())


def parse_html(content: bytes, filename: str) -> List[Dict]:
    """Parse an HTML file into plain text, stripping tags/scripts/styles."""
    raw = content.decode("utf-8", errors="ignore")
    extractor = _TextExtractor()
    extractor.feed(raw)
    cleaned = clean_text("\n".join(extractor.parts))
    return [{"page": 1, "text": cleaned}]


# ── Dispatcher ────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".md", ".csv", ".html", ".htm"}

_PARSERS = {
    ".txt":  parse_txt,
    ".pdf":  parse_pdf,
    ".docx": parse_docx,
    ".md":   parse_md,
    ".csv":  parse_csv,
    ".html": parse_html,
    ".htm":  parse_html,
}


def parse_file(content: bytes, filename: str) -> List[Dict]:
    """
    Route to the correct parser based on file extension.

    Fix #9: now supports .txt, .pdf, .docx.
    Format expansion: added .md, .csv, .html/.htm.

    Raises:
        ValueError: if the extension is not supported.
    """
    ext = Path(filename).suffix.lower()
    parser_fn = _PARSERS.get(ext)
    if parser_fn is None:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return parser_fn(content, filename)
