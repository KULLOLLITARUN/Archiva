"""
Tests for agents/loop.py's _select_context_chunks.

Found live: for "What region and availability zone must Disk-Clone-1 be in?"
retrieval DID surface the chunk naming Disk-Clone-1, but the context builder
kept only 2 chunks per document (a hardcoded cap meant to stop one document
crowding out others) and dropped it. With a single document loaded there is
nothing to balance, so the model saw a context that never mentioned
Disk-Clone-1 and - correctly - answered "Not found in the document."
"""

from agents.loop import _select_context_chunks


def _chunk(cid, doc, parent=None):
    return {"chunk_id": cid, "metadata": {"filename": doc, "parent_id": parent or f"p-{cid}"}}


def ids(chunks):
    return [c["chunk_id"] for c in chunks]


def test_single_document_uses_every_slot_instead_of_stopping_at_the_cap():
    top5 = [_chunk(f"c{i}", "plan.docx") for i in range(5)]
    assert ids(_select_context_chunks(top5, max_per_doc=2, limit=5)) == ["c0", "c1", "c2", "c3", "c4"]


def test_the_chunk_that_names_the_answer_is_not_dropped_for_being_third():
    # Ranks 1-2 are generic callouts; the chunk naming the disk is 3rd.
    top = [_chunk("region-callout", "plan.docx"), _chunk("step-2-2", "plan.docx"),
           _chunk("step-1-2-names-the-disk", "plan.docx")]
    assert "step-1-2-names-the-disk" in ids(_select_context_chunks(top, max_per_doc=2, limit=5))


def test_diversity_comes_first_when_several_documents_compete():
    # Doc A supplies ranks 1-4, doc B rank 5. Limit 3: B must not be shut out.
    chunks = [_chunk("a1", "A"), _chunk("a2", "A"), _chunk("a3", "A"), _chunk("a4", "A"), _chunk("b1", "B")]
    selected = ids(_select_context_chunks(chunks, max_per_doc=2, limit=3))
    assert selected == ["a1", "a2", "b1"]


def test_backfill_happens_in_rank_order_after_the_diverse_picks():
    chunks = [_chunk("a1", "A"), _chunk("a2", "A"), _chunk("a3", "A"), _chunk("a4", "A"), _chunk("b1", "B")]
    assert ids(_select_context_chunks(chunks, max_per_doc=2, limit=5)) == ["a1", "a2", "b1", "a3", "a4"]


def test_never_exceeds_the_limit():
    chunks = [_chunk(f"c{i}", "A") for i in range(10)]
    assert len(_select_context_chunks(chunks, max_per_doc=2, limit=5)) == 5


def test_chunks_sharing_a_parent_section_are_included_once():
    chunks = [_chunk("c1", "A", parent="sec-1"), _chunk("c2", "A", parent="sec-1"), _chunk("c3", "A", parent="sec-2")]
    assert ids(_select_context_chunks(chunks, max_per_doc=2, limit=5)) == ["c1", "c3"]


def test_duplicate_parent_is_not_resurrected_by_the_backfill():
    # c3 shares a parent with c1; it must not come back via the overflow pass.
    chunks = [_chunk("c1", "A", parent="s1"), _chunk("c2", "A", parent="s2"),
              _chunk("c3", "A", parent="s1"), _chunk("c4", "A", parent="s3")]
    selected = ids(_select_context_chunks(chunks, max_per_doc=2, limit=5))
    assert selected == ["c1", "c2", "c4"]


def test_empty_input():
    assert _select_context_chunks([]) == []


def test_chunks_without_a_parent_id_fall_back_to_their_chunk_id():
    chunks = [{"chunk_id": "x1", "metadata": {"filename": "A"}}, {"chunk_id": "x2", "metadata": {"filename": "A"}}]
    assert ids(_select_context_chunks(chunks, max_per_doc=2, limit=5)) == ["x1", "x2"]
