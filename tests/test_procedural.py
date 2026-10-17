"""Procedural memory: opaque named procedures, versioned with one active per name.

What is worth pinning: a read returns the active payload verbatim (the library never
interprets it), a re-``put`` activates the new version and retires the prior one
(kept, not deleted), ``list_kind`` returns every active procedure of a kind, and a
staged (``activate=False``) version does not become live.
"""

from __future__ import annotations

import pytest

from agentmem import memory
from agentmem.memory import procedural
from agentmem.memory.adapters.inmemory import (
    InMemoryDocs,
    InMemoryEmbedder,
    InMemoryVectors,
    StdLog,
    StubChat,
)


@pytest.fixture(autouse=True)
def configured():
    memory.configure(
        docs=InMemoryDocs(),
        vectors=InMemoryVectors(),
        embedder=InMemoryEmbedder(),
        chat=StubChat(),
        log=StdLog(),
    )


def test_get_returns_the_active_payload_verbatim():
    procedural.put("prompt", "answer", {"system_prompt": "You answer.", "model_name": "m"})
    assert procedural.get("prompt", "answer") == {"system_prompt": "You answer.", "model_name": "m"}


def test_get_is_none_when_absent():
    assert procedural.get("prompt", "nope") is None


def test_reput_activates_the_new_version_and_retires_the_prior():
    procedural.put("prompt", "answer", {"system_prompt": "v1"})
    entry2 = procedural.put("prompt", "answer", {"system_prompt": "v2"})

    assert entry2["version"] == 2
    assert entry2["is_active"] is True
    # A read returns only the live version.
    assert procedural.get("prompt", "answer") == {"system_prompt": "v2"}


def test_list_kind_returns_every_active_of_a_kind():
    procedural.put("card", "orchestrator", {"card_id": "orchestrator"})
    procedural.put("card", "context_analyst", {"card_id": "context_analyst"})
    procedural.put("workflow", "answer_question", {"workflow_id": "answer_question"})

    ids = sorted(c["card_id"] for c in procedural.list_kind("card"))
    assert ids == ["context_analyst", "orchestrator"]
    # a different kind is not mixed in
    assert [w["workflow_id"] for w in procedural.list_kind("workflow")] == ["answer_question"]


def test_a_staged_version_is_not_live():
    procedural.put("prompt", "answer", {"system_prompt": "live"})
    procedural.put("prompt", "answer", {"system_prompt": "draft"}, activate=False)
    # The draft did not take over; the live version still answers.
    assert procedural.get("prompt", "answer") == {"system_prompt": "live"}


def test_tenants_are_isolated():
    procedural.put("prompt", "answer", {"system_prompt": "acme"}, tenant="acme")
    procedural.put("prompt", "answer", {"system_prompt": "globex"}, tenant="globex")
    assert procedural.get("prompt", "answer", tenant="acme") == {"system_prompt": "acme"}
    assert procedural.get("prompt", "answer", tenant="globex") == {"system_prompt": "globex"}
    # the default (un-tenanted) space is its own, separate from named tenants
    assert procedural.get("prompt", "answer") is None
