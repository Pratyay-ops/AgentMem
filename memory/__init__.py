"""Public API for the memory layer.

    from agentmem import memory
    memory.configure(docs=..., vectors=..., embedder=..., chat=..., log=...)
    facts = await memory.semantic.recall("what does AE mean?", scope=scope)
    await memory.episodic.record(kind=..., statement=..., what_i_did=..., scope=scope)

``working`` (a checkpointer) is the host's tier — it needs a graph runtime this
library deliberately does not depend on. This library owns the other three:
``semantic`` (curated, read-only), ``episodic`` (agent-written, reviewed) and
``procedural`` (named prompts / agent configs / workflows, versioned, edited via API).
"""

from __future__ import annotations

from agentmem.memory import episodic, procedural, semantic
from agentmem.memory.runtime import NotConfigured, configure
from agentmem.memory.schemas import (
    DomainFact,
    Episode,
    EpisodeKind,
    EntryType,
    MemoryScope,
    ReviewStatus,
)

__all__ = [
    "configure",
    "NotConfigured",
    "semantic",
    "episodic",
    "procedural",
    "MemoryScope",
    "DomainFact",
    "Episode",
    "EpisodeKind",
    "EntryType",
    "ReviewStatus",
]
