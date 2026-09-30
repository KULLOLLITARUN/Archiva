"""Shared test fixtures."""

import io
import os
from urllib.parse import urlsplit

import pytest

# Every TestClient(app) start would otherwise warm the embedding + reranker
# models on a background thread - slow and pointless for tests. Set before the
# app (and config) are imported.
os.environ.setdefault("WARM_UP_ON_START", "false")


def _database_is_disposable(url: str) -> bool:
    """
    The DB tests TRUNCATE every table they touch, so they must never run
    against a database holding real data. A database is treated as
    disposable if its name ends in "_test", if it is pinned to an isolated
    schema via search_path, or if ARCHIVA_ALLOW_DESTRUCTIVE_TESTS=1 (used by
    CI, whose Postgres is a throwaway service container).
    """
    if os.environ.get("ARCHIVA_ALLOW_DESTRUCTIVE_TESTS") == "1":
        return True
    parts = urlsplit(url)
    return parts.path.lstrip("/").endswith("_test") or "search_path" in parts.query


def pytest_collection_modifyitems(config, items):
    """Skip the table-wiping Postgres test modules unless the database is disposable."""
    for item in items:
        pg = getattr(item.module, "pg", None)
        if pg is None or not hasattr(item.module, "_postgres_available"):
            continue   # not a Postgres test module
        if not _database_is_disposable(pg.DATABASE_URL):
            item.add_marker(pytest.mark.skip(reason=(
                "DATABASE_URL does not look disposable and these tests TRUNCATE tables. "
                "Point it at a database named *_test (or set ARCHIVA_ALLOW_DESTRUCTIVE_TESTS=1 "
                "if you are sure) - see README, Testing."
            )))


@pytest.fixture(autouse=True)
def _isolate_sessions_file(monkeypatch, tmp_path):
    """
    Never let a test touch the real sessions.json. The app's startup imports
    that file into Postgres and renames it, and ConversationMemory reads and
    rewrites it - all of which would mutate the developer's actual data.
    """
    import chatbot.memory as memory_module
    monkeypatch.setattr(memory_module, "_SESSIONS_FILE", str(tmp_path / "sessions.json"))
    monkeypatch.setattr(memory_module, "_SESSIONS_FILE_TMP", str(tmp_path / "sessions.json.tmp"))


def _render_text_image(lines, width=1000, line_height=90):
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (width, line_height * len(lines) + 60), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=44)   # bundled with Pillow: no OS font needed
    for i, line in enumerate(lines):
        draw.text((40, 30 + i * line_height), line, fill="black", font=font)
    return image


@pytest.fixture()
def make_scanned_pdf():
    """
    Build a genuinely image-only PDF (one rasterised page per entry in
    *pages*, each a list of text lines). pypdf/pdfplumber find no text layer
    in it, exactly like a real scan - only OCR can read it.
    """
    def _make(pages):
        images = [_render_text_image(lines) for lines in pages]
        buf = io.BytesIO()
        images[0].save(buf, format="PDF", save_all=True, append_images=images[1:])
        return buf.getvalue()
    return _make
