"""
load_docs.py — Run once before starting the API server.

Usage:
    python load_docs.py

Fix #3:  failed counter now increments on actual parse/chunk errors.
Fix #9:  Globs *.txt, *.pdf, *.docx (in that priority order).
Persistence: Postgres (db/postgres.py / db/store_sync.py), not a pickle
file — matches main.py, which no longer reads store_state.pkl at all.
"""

import sys
import uuid
from pathlib import Path

from db.postgres import init_db
from db.store_sync import load_store_from_postgres, sync_file_to_postgres
from ingestion.chunker import chunk_document
from ingestion.parser import compute_hash, parse_file, SUPPORTED_EXTENSIONS
from retrieval.store import MultiDocStore

DOCS_DIR  = Path("test_docs")


def load_store() -> MultiDocStore:
    init_db()
    try:
        store = load_store_from_postgres()
        print(f"[+] Loaded existing store: {len(store.files)} file(s), {store.total_chunks()} chunks")
        return store
    except Exception as exc:
        print(f"[!] Failed to load existing store ({exc}). Rebuilding from source files.")
        return MultiDocStore()


def main() -> None:
    if not DOCS_DIR.exists():
        DOCS_DIR.mkdir(parents=True)
        print(f"[Dir] Created {DOCS_DIR}/ -- add .txt/.pdf/.docx files there and re-run.")
        sys.exit(0)

    # Fix #9: collect all supported file types
    doc_files = sorted(
        p for p in DOCS_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not doc_files:
        exts = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        print(f"[!] No supported files ({exts}) found in {DOCS_DIR}/")
        sys.exit(0)

    store = load_store()
    loaded = 0
    failed = 0  # Fix #3: will now actually increment on errors
    stopped_early = False
    newly_added_ids: list = []

    for filepath in doc_files:
        filename = filepath.name
        try:
            content = filepath.read_bytes()
        except Exception as exc:
            print(f"  [!] {filename} -- could not read file: {exc}")
            failed += 1   # Fix #3
            continue

        content_hash = compute_hash(content)

        if content_hash in store.file_hash_map:
            print(f"  [Skip] {filename} -- skipped (duplicate hash)")
            continue

        # Fix #9: parse_file() dispatches to the correct parser by extension
        try:
            pages = parse_file(content, filename)
        except Exception as exc:
            print(f"  [!] {filename} -- parse error: {exc}")
            failed += 1   # Fix #3
            continue

        file_id = str(uuid.uuid4())
        file_type = filepath.suffix.lower().lstrip(".")

        try:
            chunks = chunk_document(
                pages=pages,
                file_id=file_id,
                filename=filename,
                file_type=file_type,
            )
        except Exception as exc:
            print(f"  [!] {filename} -- chunking error: {exc}")
            failed += 1   # Fix #3
            continue

        if not chunks:
            print(f"  [!] {filename} -- no chunks produced, skipping")
            failed += 1   # Fix #3: empty result is an error
            continue

        _, status = store.add_file(
            file_id=file_id,
            filename=filename,
            content_hash=content_hash,
            chunks=chunks,
            file_type=file_type,
        )

        if status == "ok":
            print(f"  [OK] {filename} -- {len(chunks)} chunks indexed")
            loaded += 1
            newly_added_ids.append(file_id)

        elif status == "duplicate":
            print(f"  [Skip] {filename} -- skipped (duplicate)")
        elif status == "limit":
            print(f"  [X] {filename} -- skipped (MAX_TOTAL_CHUNKS reached)")
            stopped_early = True
            break

    # Fix #2 (corrected): each add_file() spawns its own background thread, but
    # self._embed_thread only tracks the LAST one — so wait_for_embeddings() only
    # joined the final file's thread, leaving the earlier files unembedded.
    #
    # Solution: after the loop, run precompute_embeddings() synchronously on ALL
    # chunks that still lack an embedding. diskcache makes already-computed chunks
    # instant (cache hit), so only genuinely missing ones are re-encoded.
    if loaded > 0:
        print("\n[Embed] Pre-computing embeddings for all chunks before saving…")
        try:
            from retrieval.dense import precompute_embeddings
            precompute_embeddings(store.chunks)
            embedded = sum(1 for c in store.chunks if "_embedding" in c)
            print(f"[Embed] Done — {embedded}/{store.total_chunks()} chunks have embeddings.")
        except Exception as exc:
            print(f"[Warn] Embedding precompute failed: {exc} — saving without embeddings.")

        print("[DB] Syncing newly-added files to Postgres…")
        for file_id in newly_added_ids:
            try:
                sync_file_to_postgres(store, file_id)
            except Exception as exc:
                print(f"[Warn] Failed to sync {file_id!r} to Postgres: {exc}")


    print()
    print("==================================")
    if failed:
        print("[Warn] DocChat store updated with warnings")
    else:
        print("[Success] DocChat Ready (BM25 mode)")
    print(f"Files added   : {loaded}")
    print(f"Files failed  : {failed}")   # Fix #3: now reflects real failures
    print(f"Total files   : {len(store.files)}")
    print(f"Total chunks  : {store.total_chunks()}")
    if stopped_early:
        print("Resume later: python load_docs.py")
    else:
        print("Run: uvicorn main:app --reload")
    print("==================================")

    if store.is_empty():
        print("\n[Error] No usable store is available yet.")
        sys.exit(1)


if __name__ == "__main__":
    main()
