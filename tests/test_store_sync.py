"""Tests for db/store_sync.py's StoreSynchronizer and MultiDocStore.adopt:
the logic that keeps one worker's in-memory store in step with Postgres when
other worker processes change the documents. Postgres is faked here; the
real-database and real-multi-process behaviour is covered separately."""

import asyncio

import pytest

import retrieval.store as store_module
from db import store_sync
from db.store_sync import StoreSynchronizer
from retrieval.store import MultiDocStore


@pytest.fixture(autouse=True)
def _no_background_embeddings(monkeypatch):
    monkeypatch.setattr(store_module.MultiDocStore, "_trigger_embedding_precompute", lambda self, chunks: None)


def _chunk(file_id, text, idx=0):
    return {
        "chunk_id": f"{file_id}_c{idx}", "text": text,
        "metadata": {"file_id": file_id, "filename": f"{file_id}.txt", "page": 1,
                     "chunk_index": idx, "content_hash": f"{file_id}-{idx}"},
    }


def _store_with(*file_ids):
    store = MultiDocStore()
    for fid in file_ids:
        store.add_file(fid, f"{fid}.txt", f"hash-{fid}", [_chunk(fid, f"text of {fid}")], "txt")
    return store


def run(coro):
    return asyncio.run(coro)


class FakeRemote:
    """Stands in for Postgres: a version number and the store a reload returns."""

    def __init__(self, version, store):
        self.version, self.store, self.loads = version, store, 0

    def install(self, monkeypatch):
        monkeypatch.setattr(store_sync.pg, "db_get_store_version", lambda: self.version)

        def load():
            self.loads += 1
            return self.store
        monkeypatch.setattr(store_sync, "load_store_from_postgres", load)


# ── note_local_write ────────────────────────────────────────────────────────────

def test_own_consecutive_write_advances_local_version():
    sync = StoreSynchronizer()
    sync.attach(MultiDocStore(), version=5)
    sync.note_local_write(6)
    assert sync.local_version == 6


def test_write_that_skipped_a_version_does_not_advance():
    # Somebody else bumped 6 between our read (5) and our own bump (7): our
    # copy lacks their change, so we must NOT claim to be at 7.
    sync = StoreSynchronizer()
    sync.attach(MultiDocStore(), version=5)
    sync.note_local_write(7)
    assert sync.local_version == 5


def test_local_write_fires_on_change():
    fired = []
    sync = StoreSynchronizer()
    sync.attach(MultiDocStore(), version=0, on_change=lambda: fired.append(1))
    sync.note_local_write(1)
    assert fired == [1]


def test_failing_on_change_does_not_break_a_write():
    def boom():
        raise RuntimeError("cache exploded")
    sync = StoreSynchronizer()
    sync.attach(MultiDocStore(), version=0, on_change=boom)
    sync.note_local_write(1)   # must not raise
    assert sync.local_version == 1


# ── check_once ──────────────────────────────────────────────────────────────────

def test_no_reload_when_version_is_unchanged(monkeypatch):
    remote = FakeRemote(3, _store_with("a"))
    remote.install(monkeypatch)
    sync = StoreSynchronizer()
    sync.attach(MultiDocStore(), version=3)

    assert run(sync.check_once()) is False
    assert remote.loads == 0


def test_reload_when_another_process_changed_the_documents(monkeypatch):
    local = _store_with("a")
    remote = FakeRemote(4, _store_with("a", "b"))
    remote.install(monkeypatch)
    fired = []
    sync = StoreSynchronizer()
    sync.attach(local, version=3, on_change=lambda: fired.append(1))

    assert run(sync.check_once()) is True
    assert sorted(local.files) == ["a", "b"]          # adopted IN PLACE
    assert local.total_chunks() == 2
    assert sync.local_version == 4 and sync.reload_count == 1
    assert fired == [1]


def test_reload_picks_up_a_deletion_made_elsewhere(monkeypatch):
    local = _store_with("a", "b")
    FakeRemote(9, _store_with("a")).install(monkeypatch)
    sync = StoreSynchronizer()
    sync.attach(local, version=8)

    run(sync.check_once())
    assert list(local.files) == ["a"] and local.total_chunks() == 1


def test_snapshot_is_discarded_if_a_local_write_raced_the_load(monkeypatch):
    # A local upload finishing while the (off-loop) snapshot loads would be
    # erased by adopting that older snapshot. The epoch check must refuse it.
    local = _store_with("a")
    sync = StoreSynchronizer()
    sync.attach(local, version=3)

    def racing_load():
        local.add_file("new", "new.txt", "hash-new", [_chunk("new", "fresh upload")], "txt")
        sync.note_local_write(None)            # the local write lands mid-load
        return _store_with("a")                # stale snapshot without "new"

    monkeypatch.setattr(store_sync.pg, "db_get_store_version", lambda: 4)
    monkeypatch.setattr(store_sync, "load_store_from_postgres", racing_load)

    assert run(sync.check_once()) is False
    assert "new" in local.files                # the local write survived
    assert sync.local_version == 3             # still behind -> retried next tick


def test_reload_is_held_off_while_a_write_guard_is_open(monkeypatch):
    remote = FakeRemote(4, _store_with("a", "b"))
    remote.install(monkeypatch)
    sync = StoreSynchronizer()
    sync.attach(_store_with("a"), version=3)

    with sync.write_guard():
        assert run(sync.check_once()) is False
        assert remote.loads == 0
    # Guard released (and it counts as a local write) - the next tick proceeds.
    assert run(sync.check_once()) is True


def test_check_once_is_a_noop_before_attach():
    assert run(StoreSynchronizer().check_once()) is False


def test_run_survives_a_failing_tick_and_keeps_polling(monkeypatch):
    sync = StoreSynchronizer()
    sync.attach(MultiDocStore(), version=0)
    ticks = []

    def flaky_version():
        ticks.append(1)
        if len(ticks) == 1:
            raise RuntimeError("postgres blip")
        return 0
    monkeypatch.setattr(store_sync.pg, "db_get_store_version", flaky_version)

    async def go():
        task = asyncio.create_task(sync.run(interval=0.01))
        await asyncio.sleep(0.15)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    run(go())
    assert len(ticks) >= 2     # kept going after the first tick raised


def test_on_tick_runs_every_poll(monkeypatch):
    monkeypatch.setattr(store_sync.pg, "db_get_store_version", lambda: 0)
    ticks = []
    sync = StoreSynchronizer()
    sync.attach(MultiDocStore(), version=0, on_tick=lambda: ticks.append(1))

    async def go():
        task = asyncio.create_task(sync.run(interval=0.01))
        await asyncio.sleep(0.12)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    run(go())
    assert len(ticks) >= 3


# ── MultiDocStore.adopt ─────────────────────────────────────────────────────────

def test_adopt_replaces_documents_chunks_and_index():
    local = _store_with("a")
    fresh = _store_with("b", "c")
    local.adopt(fresh)

    assert sorted(local.files) == ["b", "c"]
    assert local.total_chunks() == 2
    assert local.file_hash_map == {"hash-b": "b", "hash-c": "c"}
    assert local.get_bm25_index() is fresh.get_bm25_index()


def test_adopt_keeps_local_progress_for_still_pending_documents():
    local = MultiDocStore()
    local.add_pending("p1", "scan.pdf", "h", "pdf")
    local.pending["p1"]["progress"] = (3, 10)

    fresh = MultiDocStore()
    fresh.add_pending("p1", "scan.pdf", "h", "pdf")     # remote view: no progress info

    local.adopt(fresh)
    assert local.pending["p1"]["progress"] == (3, 10)


def test_adopt_takes_remote_state_when_status_changed():
    local = MultiDocStore()
    local.add_pending("p1", "scan.pdf", "h", "pdf")

    fresh = MultiDocStore()
    fresh.add_pending("p1", "scan.pdf", "h", "pdf", status="failed", message="no text")

    local.adopt(fresh)
    assert local.pending["p1"]["status"] == "failed"
    assert local.pending["p1"]["message"] == "no text"


def test_adopt_drops_pending_documents_that_finished_elsewhere():
    local = MultiDocStore()
    local.add_pending("p1", "scan.pdf", "h", "pdf")
    local.adopt(_store_with("p1"))                     # now a normal, ready file remotely

    assert local.pending == {}
    assert "p1" in local.files
