"""Tests for ingestion/ocr.py: line assembly (pure) and OCR of a real
image-only PDF (skipped when the OCR dependencies aren't installed)."""

import pytest

from ingestion import ocr
from ingestion.ocr import OcrCancelled, assemble_lines, ocr_available, ocr_pdf
from ingestion.parser import parse_pdf


def _det(x, y, text, score=0.99, w=200, h=30):
    box = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
    return [box, text, score]


# ── assemble_lines (pure) ──────────────────────────────────────────────────────

def test_assemble_orders_rows_top_to_bottom():
    dets = [_det(10, 200, "third"), _det(10, 10, "first"), _det(10, 100, "second")]
    text, dropped = assemble_lines(dets)
    assert text == "first\nsecond\nthird"
    assert dropped == 0


def test_assemble_joins_same_row_left_to_right():
    # Same visual row, slightly different y (typical OCR jitter), out of order.
    dets = [_det(400, 12, "world"), _det(10, 10, "hello")]
    text, _ = assemble_lines(dets)
    assert text == "hello world"


def test_assemble_drops_low_confidence_lines_and_counts_them():
    dets = [_det(10, 10, "solid", score=0.95), _det(10, 100, "garbled", score=0.2)]
    text, dropped = assemble_lines(dets, min_confidence=0.6)
    assert text == "solid"
    assert dropped == 1


def test_assemble_ignores_empty_and_malformed_detections():
    dets = [_det(10, 10, "   "), ["not-a-box"], None, _det(10, 100, "ok")]
    text, dropped = assemble_lines(dets)
    assert text == "ok"
    assert dropped == 0


def test_assemble_handles_no_detections():
    assert assemble_lines(None) == ("", 0)
    assert assemble_lines([]) == ("", 0)


# ── ocr_pdf on a real scanned PDF ───────────────────────────────────────────────

needs_ocr = pytest.mark.skipif(not ocr_available(), reason="OCR dependencies not installed")


@needs_ocr
def test_scanned_pdf_has_no_text_layer_but_ocr_reads_it(make_scanned_pdf):
    pdf = make_scanned_pdf([["Vendor: Acme Corp", "Renewal date: March 2027"]])

    # Precondition: this really is a scan - the normal parser sees nothing.
    assert all(not p["text"].strip() for p in parse_pdf(pdf, "scan.pdf"))

    pages, stats = ocr_pdf(pdf, "scan.pdf")
    assert stats.pages_total == 1 and stats.pages_processed == 1
    assert stats.pages_with_text == 1
    text = pages[0]["text"].lower()
    assert "acme" in text and "renewal" in text
    assert pages[0]["page"] == 1


@needs_ocr
def test_ocr_reports_progress_per_page_and_keeps_page_numbers(make_scanned_pdf):
    pdf = make_scanned_pdf([["First page alpha"], ["Second page bravo"]])
    seen = []
    pages, stats = ocr_pdf(pdf, "scan.pdf", progress=lambda done, total: seen.append((done, total)))

    assert seen == [(1, 2), (2, 2)]
    assert [p["page"] for p in pages] == [1, 2]
    assert stats.pages_with_text == 2


@needs_ocr
def test_ocr_respects_max_pages_and_flags_truncation(make_scanned_pdf):
    pdf = make_scanned_pdf([["Page one"], ["Page two"], ["Page three"]])
    pages, stats = ocr_pdf(pdf, "scan.pdf", max_pages=2)

    assert stats.pages_total == 3
    assert stats.pages_processed == 2
    assert stats.truncated is True
    assert max(p["page"] for p in pages) <= 2
    assert "first 2 of 3" in stats.summary()


@needs_ocr
def test_ocr_can_be_cancelled_from_the_progress_callback(make_scanned_pdf):
    pdf = make_scanned_pdf([["Page one"], ["Page two"]])

    def cancel(done, total):
        raise OcrCancelled()

    with pytest.raises(OcrCancelled):
        ocr_pdf(pdf, "scan.pdf", progress=cancel)


@needs_ocr
def test_blank_page_yields_no_text(make_scanned_pdf):
    pdf = make_scanned_pdf([[" "]])
    pages, stats = ocr_pdf(pdf, "blank.pdf")
    assert pages == []
    assert stats.pages_with_text == 0


def test_ocr_available_is_false_when_engine_import_fails(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "rapidocr_onnxruntime":
            raise ImportError("simulated missing engine")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert ocr.ocr_available() is False
