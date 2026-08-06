"""
ingestion/parser.py — Document parser supporting TXT, PDF, and DOCX.

Fix #9: Added parse_pdf() (via pypdf) and parse_docx() (via python-docx).
        parse_file() dispatcher routes by file extension.
        Returns the same [{page, text}] structure regardless of format.
"""

import hashlib
import re
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



# ── Dispatcher ────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


def parse_file(content: bytes, filename: str) -> List[Dict]:
    """
    Route to the correct parser based on file extension.

    Fix #9: now supports .txt, .pdf, .docx.

    Raises:
        ValueError: if the extension is not supported.
    """
    ext = Path(filename).suffix.lower()
    if ext == ".txt":
        return parse_txt(content, filename)
    elif ext == ".pdf":
        return parse_pdf(content, filename)
    elif ext == ".docx":
        return parse_docx(content, filename)
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
