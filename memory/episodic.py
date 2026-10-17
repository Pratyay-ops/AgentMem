"""Episodic memory: the agent's assumptions and contradictions, human-reviewed.

The agent writes an entry when it makes an ASSUMPTION (needed a fact the tools did
not give it) or hits a CONTRADICTION (two sources disagree). It is a
knowledge-acquisition inbox, NOT a history log: a developer reviews it, rejects the
wrong guesses, and promotes the durable ones into semantic memory — the only path
from one user's working context to shared knowledge.

Two rules are load-bearing:
* **Recall filters ``user`` INSIDE the kNN query**, never after — a post-filter
  silently returns fewer than ``k`` and reads as "no history".
* **Only ``pending`` and ``rectified`` are recalled.** A ``rejected`` entry is a
  recorded wrong answer; feeding it back is worse than having no memory at all.
"""

from __future__ import annotations

import uuid
from typing import List

from agentmem.memory import guards, runtime
from agentmem.memory.schemas import Episode, EpisodeKind, MemoryScope, ReviewStatus

EPISODE_COLLECTION = "episode"
EPISODE_INDEX = "episode_knn"

_RECALLABLE = [ReviewStatus.PENDING.value, ReviewStatus.RECTIFIED.value]


async def record(
    *,
    kind: EpisodeKind,
    statement: str,
    what_i_did: str,
    scope: MemoryScope,
    source: str = "",
    handle: str = "",
) -> Episode:
    """Record one gap. Rejects any field carrying a raw id (see :mod:`.guards`)."""
    for text in (statement, what_i_did, handle):
        guards.assert_no_raw_ids(text)

    episode = Episode(
        episode_id=uuid.uuid4().hex,
        kind=kind,
        statement=statement,
        what_i_did=what_i_did,
        source=source,
        handle=handle,
        tenant=scope.tenant,
        scope=scope.scope,
        user=scope.user,
    )
    doc = episode.model_dump(mode="json")
    runtime.docs().insert_one(EPISODE_COLLECTION, doc)
    vector = (await runtime.embedder().embed([statement]))[0]
    runtime.vectors().index(EPISODE_INDEX, episode.episode_id, vector, doc)
    runtime.log().info("episode recorded | kind=%s user=%s", kind.value, scope.user)
    return episode


async def recall(question: str, *, scope: MemoryScope, k: int = 5) -> List[Episode]:
    """This user's own pending/rectified episodes relevant to ``question``.

    The ``user`` and the recallable-status set are part of the kNN FILTER, so a
    rejected entry or another user's entry is never a neighbour to begin with.
    """
    vector = (await runtime.embedder().embed([question]))[0]
    hits = runtime.vectors().knn(
        EPISODE_INDEX,
        vector,
        k=k,
        filters={
            "tenant": scope.tenant,
            "user": scope.user,
            "review_status": {"$in": _RECALLABLE},
        },
    )
    return [Episode(**h) for h in hits]
