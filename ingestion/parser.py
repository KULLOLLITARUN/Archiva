"""
ingestion/parser.py — Document parser supporting TXT, PDF, DOCX, Markdown,
CSV, and HTML.

Fix #9: Added parse_pdf() (via pypdf) and parse_docx() (via python-docx).
        parse_file() dispatcher routes by file extension.
        Returns the same [{page, text}] structure regardless of format.

Format expansion: Added parse_md() / parse_csv() / parse_html() — all
stdlib-only, no new dependencies. Scanned/image PDFs are handled by
ingestion/ocr.py (background job), not here.
"""

import csv
import hashlib
import io
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, Iterator, List

from config import MAX_INGEST_CHARS


def compute_hash(content: bytes) -> str:
    """Compute MD5 hash of file content bytes."""
    return hashlib.md5(content).hexdigest()


def clean_text(text: str) -> str:
    """Remove null bytes, collapse whitespace, and strip."""
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _bound_for_ingest(text: str, filename: str) -> str:
    """
    Cap text length for single-page formats (.txt/.csv/.html/.docx — anything
    that isn't naturally paginated) so chunking one huge blob does bounded
    work regardless of how large the source file is. See MAX_INGEST_CHARS
    in config.py for why this is separate from the upload-size guard.
    """
    if len(text) <= MAX_INGEST_CHARS:
        return text
    print(
        f"  [WARN]  [parser] {filename!r} truncated at {MAX_INGEST_CHARS:,} chars "
        f"(was {len(text):,}) — only the beginning of the file will be indexed."
    )
    return text[:MAX_INGEST_CHARS]


# ── TXT ───────────────────────────────────────────────────────────────────────

def parse_txt(content: bytes, filename: str) -> List[Dict]:
    """
    Parse a TXT file's raw bytes into a list of page dicts.
    Always returns a single page.
    """
    raw = content.decode("utf-8", errors="ignore")
    cleaned = clean_text(_bound_for_ingest(raw, filename))
    return [{"page": 1, "text": cleaned}]


# ── PDF ───────────────────────────────────────────────────────────────────────

def _render_table_rows(rows: List[List]) -> str:
    """
    Render a pdfplumber-extracted table (list of rows, each a list of cell
    strings) as "col=value, col=value" lines — one line per data row, header
    row supplying field names. Same format parse_csv() already produces, so
    is_table_text()/chunk_page()'s row-aware chunking (ingestion/chunker.py)
    pick this up automatically regardless of which parser produced it.
    """
    rows = [r for r in rows if r and any(c for c in r)]
    if len(rows) < 2:
        return ""

    header = [str(h or "").strip() for h in rows[0]]
    lines = []
    for row in rows[1:]:
        pairs = [
            f"{h}={str(v).strip()}"
            for h, v in zip(header, row)
            if h and v is not None and str(v).strip()
        ]
        if pairs:
            lines.append(", ".join(pairs))
    return "\n".join(lines)


def parse_pdf(content: bytes, filename: str) -> Iterator[Dict]:
    """
    Lazily parse a PDF file into per-page dicts, one page at a time: plain
    text via pypdf (unchanged from before), plus any detected tables via
    pdfplumber, rendered as structured "col=value" rows instead of the
    jumbled left-to-right text pypdf alone would produce for a table.

    Both extractions stay page-synchronized inside one loop so the whole
    function is still lazy — chunk_document()'s MAX_CHUNKS_PER_FILE
    early-stop keeps skipping pages beyond the cap for BOTH passes, not
    just the text one.

    Note: the table's cells also appear (jumbled) in the plain-text yield
    for that page, since pypdf has no notion of "skip the table region."
    That's accepted duplication, not a bug — retrieval reliably prefers
    the clean structured version for exact term matches (e.g. "Costs=90000"),
    and the LLM sees both either way once a chunk from that page is retrieved.
    pdfplumber is optional: if it's not installed, PDFs still parse fine,
    just without the table-structure improvement.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError(
            "pypdf is required for PDF support. Run: pip install pypdf"
        ) from exc

    try:
        import pdfplumber
        plumber_pdf = pdfplumber.open(io.BytesIO(content))
    except Exception:
        plumber_pdf = None

    reader = PdfReader(io.BytesIO(content))
    yielded_any = False
    try:
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
                yielded_any = True
                yield {"page": i, "text": cleaned}

            if plumber_pdf is not None:
                try:
                    plumber_page = plumber_pdf.pages[i - 1]
                    for table_rows in plumber_page.extract_tables():
                        table_text = _render_table_rows(table_rows)
                        if table_text:
                            yielded_any = True
                            yield {"page": i, "text": table_text}
                except Exception:
                    pass
    finally:
        if plumber_pdf is not None:
            plumber_pdf.close()

    if not yielded_any:
        yield {"page": 1, "text": ""}


# ── DOCX ──────────────────────────────────────────────────────────────────────

def parse_docx(content: bytes, filename: str) -> List[Dict]:
    """
    Parse a DOCX file from raw bytes using python-docx.

    Paragraph text becomes one page entry. Each table becomes its OWN page
    entry too, rendered as "col=value" rows (first row as headers) instead
    of raw "|"-joined cells — the same format parse_csv() and parse_pdf()'s
    table extraction already use, so is_table_text() picks it up and
    chunk_page() (ingestion/chunker.py) keeps the whole table together as
    one retrievable unit instead of fragmenting it into scattered row
    chunks mixed in with prose.
    """
    try:
        from docx import Document
    except ImportError as exc:
        raise ImportError(
            "python-docx is required for DOCX support. Run: pip install python-docx"
        ) from exc

    doc = Document(io.BytesIO(content))

    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    raw = "\n\n".join(paragraphs)
    cleaned = clean_text(_bound_for_ingest(raw, filename))

    pages: List[Dict] = []
    if cleaned:
        pages.append({"page": 1, "text": cleaned})

    for table in doc.tables:
        rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
        table_text = _render_table_rows(rows)
        if table_text:
            pages.append({"page": 1, "text": table_text})

    if not pages:
        pages.append({"page": 1, "text": ""})

    return pages



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
    raw = _bound_for_ingest(content.decode("utf-8", errors="ignore"), filename)
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
    raw = _bound_for_ingest(content.decode("utf-8", errors="ignore"), filename)
    extractor = _TextExtractor()
    extractor.feed(raw)
    cleaned = clean_text("\n".join(extractor.parts))
    return [{"page": 1, "text": cleaned}]


# ── Dispatcher ────────────────────────────────────────────────────────────────

# Scanned/image-only PDFs yield no text here. main.py's /upload detects that
# and hands the file to ingestion/ocr_jobs.py, which OCRs it in the background
# (OCR is far too slow to run inside the request). parse_pdf() itself stays
# OCR-free on purpose: it is the fast path for every normal PDF.
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
