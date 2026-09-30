"""
ingestion/ocr.py — OCR for scanned / image-only PDFs.

Engine: RapidOCR (ONNX runtime, models bundled in the wheel) rendering pages
with pypdfium2. Both are plain pip packages — no system binary (unlike
Tesseract), so SETUP stays "pip install -r requirements.txt". Both are
optional at runtime: ocr_available() reports whether they import, and
main.py's /upload keeps returning the explicit "no readable text" error
when they don't.

This module is CPU-heavy (seconds per page) and must never run inline in a
request handler — ingestion/ocr_jobs.py runs it on a dedicated worker
thread. It has no store/Postgres dependency, so it stays unit-testable.

Quality policy: text the engine isn't confident about is dropped, not
indexed (OCR_MIN_LINE_CONFIDENCE). A gap in the index is recoverable; low-
confidence noise that BM25/dense search happily retrieves and the LLM then
quotes is not.
"""

import io
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from config import OCR_MAX_PAGES, OCR_MIN_LINE_CONFIDENCE, OCR_RENDER_SCALE
from ingestion.parser import clean_text


class OcrCancelled(Exception):
    """Raised from the progress callback to abandon a job (document deleted)."""


@dataclass
class OcrStats:
    pages_total: int = 0                 # pages in the PDF
    pages_processed: int = 0             # pages actually run through OCR
    pages_with_text: int = 0             # pages that kept usable text
    lines_dropped: int = 0               # lines below the confidence floor
    truncated: bool = False              # PDF had more than OCR_MAX_PAGES pages
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"{self.pages_with_text}/{self.pages_processed} pages had readable text"]
        if self.truncated:
            parts.append(f"only the first {self.pages_processed} of {self.pages_total} pages were processed")
        if self.lines_dropped:
            parts.append(f"{self.lines_dropped} low-confidence line(s) dropped")
        return "; ".join(parts)


# ── Availability / engine ───────────────────────────────────────────────────────

_engine = None
_engine_lock = threading.Lock()


def ocr_available() -> bool:
    """True if both the OCR engine and the PDF renderer can be imported."""
    try:
        import pypdfium2  # noqa: F401
        import rapidocr_onnxruntime  # noqa: F401
        return True
    except Exception:
        return False


def _get_engine():
    """Lazily build one shared RapidOCR instance (model load takes ~1s)."""
    global _engine
    with _engine_lock:
        if _engine is None:
            from rapidocr_onnxruntime import RapidOCR
            _engine = RapidOCR()
        return _engine


# ── Line assembly ────────────────────────────────────────────────────────────────

def _box_bounds(box) -> tuple:
    xs = [pt[0] for pt in box]
    ys = [pt[1] for pt in box]
    return min(xs), min(ys), max(xs), max(ys)


def assemble_lines(detections, min_confidence: float = OCR_MIN_LINE_CONFIDENCE) -> tuple:
    """
    Turn RapidOCR detections [[box, text, score], ...] into reading-order text.

    Detections that sit on the same visual row (vertical centres within half
    a line height) are joined left-to-right with a space; rows are emitted
    top-to-bottom. Returns (text, dropped_line_count).
    """
    kept = []
    dropped = 0
    for det in detections or []:
        try:
            box, text, score = det[0], str(det[1]).strip(), float(det[2])
        except (IndexError, TypeError, ValueError):
            continue
        if not text:
            continue
        if score < min_confidence:
            dropped += 1
            continue
        x0, y0, x1, y1 = _box_bounds(box)
        kept.append({"x": x0, "yc": (y0 + y1) / 2, "h": max(y1 - y0, 1.0), "text": text})

    kept.sort(key=lambda d: (d["yc"], d["x"]))

    rows: List[List[dict]] = []
    for det in kept:
        if rows:
            prev = rows[-1][0]
            if abs(det["yc"] - prev["yc"]) <= 0.5 * max(det["h"], prev["h"]):
                rows[-1].append(det)
                continue
        rows.append([det])

    lines = [" ".join(d["text"] for d in sorted(row, key=lambda d: d["x"])) for row in rows]
    return "\n".join(lines), dropped


# ── PDF entry point ─────────────────────────────────────────────────────────────

def ocr_pdf(
    content: bytes,
    filename: str = "",
    progress: Optional[Callable[[int, int], None]] = None,
    max_pages: int = OCR_MAX_PAGES,
) -> tuple:
    """
    OCR a PDF's pages. Returns (pages, stats) where pages is the same
    [{"page": n, "text": str}] shape every other parser produces, containing
    only pages that kept usable text.

    *progress(done, total)* is called after each page; it may raise
    OcrCancelled to stop early (e.g. the document was deleted mid-job).
    """
    import numpy as np
    import pypdfium2 as pdfium

    engine = _get_engine()
    pdf = pdfium.PdfDocument(io.BytesIO(content))
    stats = OcrStats(pages_total=len(pdf))
    to_process = min(stats.pages_total, max_pages)
    stats.truncated = stats.pages_total > to_process

    pages: List[Dict] = []
    try:
        for index in range(to_process):
            page = pdf[index]
            try:
                image = page.render(scale=OCR_RENDER_SCALE).to_pil().convert("RGB")
            finally:
                page.close()

            detections, _ = engine(np.array(image))
            text, dropped = assemble_lines(detections)
            text = clean_text(text)

            stats.pages_processed += 1
            stats.lines_dropped += dropped
            if text:
                stats.pages_with_text += 1
                pages.append({"page": index + 1, "text": text})

            if progress is not None:
                progress(stats.pages_processed, to_process)
    finally:
        pdf.close()

    return pages, stats
