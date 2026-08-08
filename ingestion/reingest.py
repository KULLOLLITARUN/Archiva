"""
ingestion/reingest.py — Persists uploaded file bytes to disk and re-runs the
parse -> chunk -> index pipeline against them.

Closes the loop on the healer's REINGEST signal (agents/healer.py): today
that signal is only ever written to logs/reingestion_queue.jsonl and never
consumed. This module lets it actually trigger something.

Limitation, stated plainly: refreshing re-runs today's parser/chunker
against the ORIGINAL uploaded bytes. It cannot fetch newer external data —
if the source content itself is stale, a human still needs to upload a
newer version. What this buys you:
  (a) uploaded files now survive a restart (they previously only existed
      in memory for the duration of the upload request), and
  (b) chunking/parsing can be refreshed for all documents after a code
      change, without asking every user to re-upload.
"""

import os
from typing import Dict, List

from ingestion.chunker import chunk_document
from ingestion.parser import parse_file


def persisted_path(dir_path: str, file_id: str, file_type: str) -> str:
    """Deterministic on-disk path for a given file_id's persisted upload."""
    return os.path.join(dir_path, f"{file_id}.{file_type}")


def save_uploaded_file(content: bytes, file_id: str, file_type: str, dir_path: str) -> str:
    """Persist raw upload bytes to disk so they can be reprocessed later."""
    os.makedirs(dir_path, exist_ok=True)
    path = persisted_path(dir_path, file_id, file_type)
    with open(path, "wb") as f:
        f.write(content)
    return path


def reprocess_file(path: str, file_id: str, filename: str, file_type: str) -> List[Dict]:
    """Re-run parse + chunk against a persisted file's bytes on disk."""
    with open(path, "rb") as f:
        content = f.read()
    pages = parse_file(content, filename)
    return chunk_document(pages=pages, file_id=file_id, filename=filename, file_type=file_type)


def refresh_all_from_disk(store, dir_path: str) -> Dict:
    """
    Re-parse + re-chunk every active file in *store* from its persisted copy
    on disk, replacing its chunks in place. Files with no persisted copy
    (e.g. uploaded before this feature existed) are skipped and reported,
    not treated as an error.

    Returns: {"refreshed": [...filenames], "skipped": [...filenames]}
    """
    refreshed: List[str] = []
    skipped: List[str] = []

    for file_id, record in list(store.files.items()):
        filename = record["filename"]
        file_type = record["file_type"]
        path = persisted_path(dir_path, file_id, file_type)

        if not os.path.exists(path):
            skipped.append(filename)
            continue

        try:
            new_chunks = reprocess_file(path, file_id, filename, file_type)
        except Exception as exc:
            print(f"  [WARN]  [reingest] Failed to reprocess {filename!r}: {exc}")
            skipped.append(filename)
            continue

        content_hash = record["hash"]
        store.delete_file(file_id)
        store.add_file(
            file_id=file_id,
            filename=filename,
            content_hash=content_hash,
            chunks=new_chunks,
            file_type=file_type,
        )
        refreshed.append(filename)

    return {"refreshed": refreshed, "skipped": skipped}
