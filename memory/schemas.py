"""The shapes the memory tiers store — all CAF-free, all scope-generic.

``MemoryScope`` is the ONLY vocabulary the library exposes: ``tenant`` / ``scope``
/ ``user``. A host maps its own ids onto these (e.g. a report id becomes ``scope``)
before calling in; the library never sees or interprets a domain id. That single
rule is what keeps a generic library from silently becoming a CAF one.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryScope(BaseModel):
    """Who and where a memory belongs to — the host's opaque keys.

    ``tenant`` isolates one customer/org; ``scope`` narrows within it (a document, a
    workspace — whatever the host decides); ``user`` attributes episodic writes. A
    blank ``scope``/``user`` is legal (tenant-wide, or unattributed reads).
    """

    model_config = {"frozen": True}

    tenant: str = ""
    scope: str = ""
    user: str = ""


class EntryType(str, Enum):
    GLOSSARY = "glossary"
    DATA_DEFINITION = "data_definition"
    DOMAIN_RULE = "domain_rule"


class DomainFact(BaseModel):
    """One curated, human-authored, read-only piece of domain knowledge (semantic).

    Three shapes share one collection so a single question retrieves whatever
    answers it; ``entry_type`` is what lets each RENDER correctly (a glossary pair
    flattened to prose loses the term/definition pairing that made it useful).

    ``visibility`` is ``"global"`` or ``"workspace"``: a workspace entry SHADOWS a
    global one for the same term, and the shadowing is surfaced so a silently
    different definition cannot be used without anyone noticing.
    """

    model_config = {"extra": "ignore"}

    fact_id: str = ""
    entry_type: EntryType = EntryType.DOMAIN_RULE
    # glossary / data_definition
    term: str = ""
    synonyms: List[str] = Field(default_factory=list)
    abbreviation: str = ""
    definition: str = ""
    # data_definition
    unit: str = ""
    fmt: str = ""
    allowed_values: List[str] = Field(default_factory=list)
    # domain_rule
    statement: str = ""
    applies_when: str = ""
    # scoping
    visibility: str = "global"  # "global" | "workspace"
    tenant: str = ""
    scope: str = ""  # set only when visibility == "workspace"
    is_active: bool = True
    created_at: datetime = Field(default_factory=_now)


class EpisodeKind(str, Enum):
    ASSUMPTION = "assumption"
    CONTRADICTION = "contradiction"


class ReviewStatus(str, Enum):
    PENDING = "pending"
    RECTIFIED = "rectified"
    REJECTED = "rejected"
    PROMOTED = "promoted"


class Episode(BaseModel):
    """One gap the agent recorded: an assumption it made, or a contradiction it hit.

    NOT a transcript of successful turns — the loop only terminates because the log
    records what the agent did NOT know. Scoped to the ``user`` while ``pending``;
    a human review promotes the durable ones into semantic memory (the only path to
    shared knowledge). ``handle`` is a human-addressable reference ("4.2 Safety"),
    NEVER a raw id — see :mod:`.guards`.
    """

    model_config = {"extra": "ignore"}

    episode_id: str = ""
    kind: EpisodeKind = EpisodeKind.ASSUMPTION
    statement: str = ""
    what_i_did: str = ""
    source: str = ""  # a tool name, or "no tool returned this"
    handle: str = ""  # a human handle, never an id
    tenant: str = ""
    scope: str = ""
    user: str = ""
    review_status: ReviewStatus = ReviewStatus.PENDING
    reviewed_by: str = ""
    promoted_to: str = ""  # a DomainFact id, once promoted
    created_at: datetime = Field(default_factory=_now)
