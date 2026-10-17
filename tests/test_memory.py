"""The memory tiers against the in-memory adapters — no database, runs in CI.

Pins the behaviours the design turns on: semantic is exact-match-first and
workspace-shadows-global; episodic recalls only your own pending/rectified gaps and
never a rejected one; and no memory may carry a raw id.
"""

from __future__ import annotations

import pytest

from agentmem import memory
from agentmem.memory import guards
from agentmem.memory.adapters.inmemory import (
    InMemoryDocs,
    InMemoryEmbedder,
    InMemoryVectors,
    StdLog,
    StubChat,
)
from agentmem.memory.schemas import DomainFact, EntryType, EpisodeKind, ReviewStatus


@pytest.fixture
def configured():
    docs, vectors = InMemoryDocs(), InMemoryVectors()
    memory.configure(
        docs=docs, vectors=vectors, embedder=InMemoryEmbedder(), chat=StubChat(), log=StdLog()
    )
    return docs, vectors


SCOPE = memory.MemoryScope(tenant="acme", scope="doc-42", user="u1")


def _seed_fact(docs, **kw):
    fact = DomainFact(**kw)
    docs.insert_one("semantic_fact", fact.model_dump(mode="json"))


async def test_semantic_exact_match_beats_knn(configured):
    docs, _ = configured
    _seed_fact(docs, fact_id="f1", entry_type=EntryType.GLOSSARY, term="AE",
               abbreviation="AE", definition="adverse event", tenant="acme")
    hits = await memory.semantic.recall("what does AE mean here?", scope=SCOPE, k=5)
    assert [f.term for f in hits] == ["AE"]
    assert "adverse event" in memory.semantic.render(hits)


async def test_semantic_workspace_shadows_global(configured):
    docs, _ = configured
    _seed_fact(docs, fact_id="g", entry_type=EntryType.GLOSSARY, term="dose",
               definition="global meaning", visibility="global", tenant="acme")
    _seed_fact(docs, fact_id="w", entry_type=EntryType.GLOSSARY, term="dose",
               definition="local meaning", visibility="workspace", scope="doc-42", tenant="acme")
    hits = await memory.semantic.recall("what is the dose here?", scope=SCOPE, k=5)
    assert [f.definition for f in hits] == ["local meaning"]  # workspace wins


async def test_semantic_workspace_entry_does_not_leak_to_another_scope(configured):
    docs, _ = configured
    _seed_fact(docs, fact_id="w", entry_type=EntryType.GLOSSARY, term="dose",
               definition="local meaning", visibility="workspace", scope="doc-99", tenant="acme")
    other = memory.MemoryScope(tenant="acme", scope="doc-42", user="u1")
    hits = await memory.semantic.recall("dose", scope=other, k=5)
    assert hits == []  # a workspace entry applies only to its own scope


async def test_episodic_records_and_recalls_own_pending(configured):
    await memory.episodic.record(
        kind=EpisodeKind.ASSUMPTION, statement="dosing is in mg/kg here",
        what_i_did="proceeded", scope=SCOPE,
    )
    got = await memory.episodic.recall("dosing units", scope=SCOPE, k=5)
    assert len(got) == 1 and got[0].kind == EpisodeKind.ASSUMPTION


async def test_episodic_does_not_recall_another_users_episode(configured):
    await memory.episodic.record(
        kind=EpisodeKind.ASSUMPTION, statement="mine", what_i_did="x", scope=SCOPE,
    )
    other = memory.MemoryScope(tenant="acme", scope="doc-42", user="u2")
    assert await memory.episodic.recall("mine", scope=other, k=5) == []


async def test_episodic_never_recalls_a_rejected_episode(configured):
    docs, vectors = configured
    ep = await memory.episodic.record(
        kind=EpisodeKind.ASSUMPTION, statement="wrong guess", what_i_did="x", scope=SCOPE,
    )
    # A reviewer rejected it — mark it in both stores.
    doc = docs.find_one("episode", {"episode_id": ep.episode_id})
    doc["review_status"] = ReviewStatus.REJECTED.value
    vectors.index("episode_knn", ep.episode_id, [1.0] * 32, doc)
    assert await memory.episodic.recall("wrong guess", scope=SCOPE, k=5) == []


def test_a_raw_id_is_refused_on_write():
    with pytest.raises(guards.RawIdInMemory):
        guards.assert_no_raw_ids("authored from element 3f2504e0-4f89-41d3-9a0c-0305e82c3301")


def test_a_tier_before_configure_fails_loudly():
    import importlib
    from agentmem.memory import runtime
    importlib.reload(runtime)  # reset the injected slots
    with pytest.raises(runtime.NotConfigured):
        runtime.docs()
