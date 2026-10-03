"""
retrieval/store.py — In-memory document store with integrated BM25 index.

Fix #2 (embedding race): wait_for_embeddings() lets callers join the background
  embedding thread before pickling so the store is always saved with embeddings
  baked into chunk dicts.

Cleanup: ScopedDocStore and multi-tenant scoping removed (auth fully removed).
"""

import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

from rank_bm25 import BM25Okapi

from config import MAX_TOTAL_CHUNKS
from ingestion.embedder import build_bm25_index


class MultiDocStore:
    """
    In-memory document store with BM25 and (optional) dense search support.
    Persisted to / restored from store_state.pkl.
    """

    CURRENT_VERSION: str = "3.1"

    def __init__(self):
        self.STORE_VERSION: str           = "3.1"
        self.files: Dict[str, dict]       = {}
        self.chunks: List[dict]           = []
        self.file_hash_map: Dict[str, str] = {}   # content_hash → file_id
        self._bm25: Optional[BM25Okapi]  = None
        self._chunk_hashes: Set[str]     = set()  # SHA-256 of each chunk text
        # Documents accepted but not yet searchable (background OCR). Kept
        # apart from `files`/`chunks` so retrieval, BM25 and get_files_summary()
        # never see a document with nothing indexed. See add_pending().
        self.pending: Dict[str, dict]    = {}
        # Fix #2: track the background embedding thread so we can join it
        self._embed_thread: Optional[threading.Thread] = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_embed_thread"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._embed_thread = None
        self.__dict__.setdefault("pending", {})

    # ── Private ────────────────────────────────────────────────────────────────

    def _rebuild_index(self) -> None:
        if self.chunks:
            self._bm25 = build_bm25_index(self.chunks)
        else:
            self._bm25 = None

    def _trigger_embedding_precompute(self, new_chunks: List[dict]) -> None:
        """
        Start embedding precompute in a background thread.

        Fix #2: stores the thread handle in self._embed_thread so that
        wait_for_embeddings() can join it before the store is pickled,
        preventing the race where save_store() runs before embeddings
        are written into the chunk dicts.
        """
        def _run():
            try:
                from retrieval.dense import precompute_embeddings
                precompute_embeddings(new_chunks)
            except Exception as exc:
                print(f"  [WARN]  [store] Embedding precompute failed: {exc}")

        t = threading.Thread(target=_run, daemon=True, name="embed-precompute")
        self._embed_thread = t
        t.start()

    def wait_for_embeddings(self, timeout: float = 120.0) -> None:
        """
        Block until the most recent embedding precompute thread finishes.

        Call this before pickling the store so embeddings are baked in.
        Safe to call even when no thread has started (no-op).
        """
        if self._embed_thread is not None and self._embed_thread.is_alive():
            print("  [store] Waiting for embedding precompute to finish…")
            self._embed_thread.join(timeout=timeout)
            if self._embed_thread.is_alive():
                print("  [WARN]  [store] Embedding thread timed out — saving without all embeddings.")
            else:
                print("  [store] Embeddings ready.")
        self._embed_thread = None

    # ── Write ──────────────────────────────────────────────────────────────────

    def add_file(
        self,
        file_id: str,
        filename: str,
        content_hash: str,
        chunks: List[dict],
        file_type: str = "txt",
    ) -> Tuple[Optional[str], str]:
        """
        Add a file and its chunks; rebuild BM25 index; trigger embedding precompute.

        Returns:
            (file_id, "ok")         – success
            (None,    "duplicate")  – hash already present
            (None,    "limit")      – would exceed MAX_TOTAL_CHUNKS
        """
        if content_hash in self.file_hash_map:
            return None, "duplicate"

        if len(self.chunks) + len(chunks) > MAX_TOTAL_CHUNKS:
            return None, "limit"

        uploaded_at = datetime.now(timezone.utc).isoformat()

        file_record = {
            "file_id":     file_id,
            "filename":    filename,
            "file_type":   file_type,
            "hash":        content_hash,
            "chunk_count": len(chunks),
            "uploaded_at": uploaded_at,
            "status":      "active",
        }

        self.files[file_id] = file_record
        self.chunks.extend(chunks)
        self.file_hash_map[content_hash] = file_id

        # Track per-chunk content hashes for global cross-doc dedup
        for chunk in chunks:
            h = chunk.get("metadata", {}).get("content_hash", "")
            if h:
                self._chunk_hashes.add(h)

        self._rebuild_index()

        # Background embedding precompute (thread tracked for wait_for_embeddings)
        self._trigger_embedding_precompute(chunks)

        return file_id, "ok"

    def delete_file(self, file_id: str) -> bool:
        if file_id not in self.files:
            return False

        file_record  = self.files.pop(file_id)
        content_hash = file_record["hash"]
        self.file_hash_map.pop(content_hash, None)

        # Remove per-chunk hashes belonging to this file
        removed_hashes = {
            c["metadata"].get("content_hash", "")
            for c in self.chunks
            if c["metadata"]["file_id"] == file_id
        }
        self._chunk_hashes -= {h for h in removed_hashes if h}

        self.chunks = [
            c for c in self.chunks
            if c["metadata"]["file_id"] != file_id
        ]
        self._rebuild_index()
        return True

    def adopt(self, fresh: "MultiDocStore") -> None:
        """
        Replace this store's contents with *fresh*'s, IN PLACE.

        Used when another worker process changed the documents and this
        one reloads from Postgres. In place (rather than rebinding the
        module-level `store`) so every holder of this object - the OCR job
        runner, background tasks - keeps a valid reference.

        Pending records for documents that are STILL pending keep their
        local entry, so a job running in this process keeps its page
        progress across a reload.
        """
        merged_pending = {}
        for file_id, record in fresh.pending.items():
            local = self.pending.get(file_id)
            merged_pending[file_id] = local if local is not None and local["status"] == record["status"] else record

        self.files          = fresh.files
        self.chunks         = fresh.chunks
        self.file_hash_map  = fresh.file_hash_map
        self._chunk_hashes  = fresh._chunk_hashes
        self._bm25          = fresh._bm25
        self.pending        = merged_pending

    # ── Pending (not-yet-searchable) documents ─────────────────────────────────

    def add_pending(
        self,
        file_id: str,
        filename: str,
        content_hash: str,
        file_type: str,
        status: str = "processing",
        message: str = "",
    ) -> dict:
        """
        Register a document whose text isn't extracted yet (a scanned PDF
        queued for OCR) or whose extraction failed. It holds its content
        hash so the same file can't be uploaded twice while it is in flight,
        but contributes nothing to retrieval.
        """
        record = {
            "file_id":     file_id,
            "filename":    filename,
            "file_type":   file_type,
            "hash":        content_hash,
            "chunk_count": 0,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "status":      status,
            "message":     message,
            "progress":    None,   # (pages_done, pages_total) while processing
        }
        self.pending[file_id] = record
        return record

    def set_pending_state(self, file_id: str, status: str, message: str = "") -> None:
        record = self.pending.get(file_id)
        if record is not None:
            record["status"] = status
            record["message"] = message

    def remove_pending(self, file_id: str) -> bool:
        return self.pending.pop(file_id, None) is not None

    def find_pending_by_hash(self, content_hash: str) -> Optional[dict]:
        for record in self.pending.values():
            if record["hash"] == content_hash:
                return record
        return None

    def get_pending(self) -> List[dict]:
        return list(self.pending.values())

    # ── Read ───────────────────────────────────────────────────────────────────

    def get_bm25_index(self) -> Optional[BM25Okapi]:
        return self._bm25

    def get_all_chunks(self) -> List[dict]:
        return list(self.chunks)

    def get_file_chunks(self, file_id: str) -> List[dict]:
        return [c for c in self.chunks if c["metadata"]["file_id"] == file_id]

    def get_files(self) -> List[dict]:
        return list(self.files.values())

    def get_files_summary(self) -> str:
        """Return a formatted text summary of all active documents in the store."""
        if not self.files:
            return "No documents currently uploaded."
        lines = [f"[Loaded System Documents ({len(self.files)} total)]:"]
        for f in self.files.values():
            filename = f.get("filename", "Unknown")
            chunks = f.get("chunk_count", 0)
            lines.append(f"- {filename} ({chunks} chunks)")
        return "\n".join(lines)

    def get_chunk_hashes(self) -> Set[str]:
        """Return all content hashes present in the store (for cross-doc dedup)."""
        return set(self._chunk_hashes)

    def total_chunks(self) -> int:
        return len(self.chunks)

    def is_empty(self) -> bool:
        return len(self.chunks) == 0

    def get_parent_text(self, parent_id: str) -> str:
        """
        Return the full parent_text for a given parent_id.
        Falls back to empty string if the parent_id is not found.
        Used by the agent loop to expand child chunks to rich LLM context.
        """
        for chunk in self.chunks:
            if chunk.get("metadata", {}).get("parent_id") == parent_id:
                return chunk["metadata"].get("parent_text", chunk.get("text", ""))
        return ""

