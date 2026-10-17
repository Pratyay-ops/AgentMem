"""Procedural memory: the agent's named procedures — prompts, agent/tool configs,
workflows — stored as OPAQUE documents. Read often, written by a deploy or an edit.

The fourth CoALA tier. Where semantic holds curated facts and episodic holds the
agent's assumptions, procedural memory is HOW the agent is configured to act: its
system prompts, which capabilities exist, the steps a flow takes. The library stores
each as an opaque ``payload`` keyed by ``(tenant, kind, name)`` and **never interprets
it** — a "prompt" is a dict here, a "card" is a dict, a "workflow" is a dict; the host
owns their shapes, exactly as it owns nothing the DocStore holds. That is what lets
one storage-agnostic tier serve every kind of procedure and stay domain-free.

Versioned and activatable like a deployment: many versions of a ``(kind, name)`` may
exist, exactly ONE is active, and a read returns the active one. So editing a prompt
is a :func:`put` of a new active version and the prior one is retired but kept — "edit
via API" and an audit trail, uniform across every procedure.

``tenant`` is the only scope key: procedures are configuration for a whole
tenant/deployment, not per-document, so unlike :mod:`.semantic` / :mod:`.episodic`
this tier takes a plain ``tenant`` (``""`` = the default, un-tenanted deployment)
rather than a per-record :class:`~.schemas.MemoryScope`.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agentmem.memory import runtime

PROCEDURAL_COLLECTION = "procedural_entry"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get(kind: str, name: str, *, tenant: str = "") -> Optional[Dict[str, Any]]:
    """The active payload for one named procedure, or ``None`` if there is none."""
    row = runtime.docs().find_one(
        PROCEDURAL_COLLECTION,
        {"tenant": tenant, "kind": kind, "name": name, "is_active": True},
    )
    return dict(row.get("payload") or {}) if row else None


def list_kind(kind: str, *, tenant: str = "") -> List[Dict[str, Any]]:
    """Every active payload of one ``kind`` — e.g. all cards, or all workflows."""
    rows = runtime.docs().find_many(
        PROCEDURAL_COLLECTION,
        {"tenant": tenant, "kind": kind, "is_active": True},
    )
    return [dict(r.get("payload") or {}) for r in rows]


def put(
    kind: str,
    name: str,
    payload: Dict[str, Any],
    *,
    tenant: str = "",
    activate: bool = True,
) -> Dict[str, Any]:
    """Store a procedure payload as a new version. Returns the stored entry.

    When ``activate`` (the default), the new version becomes THE active one for
    ``(kind, name)`` and every prior active version is retired — kept, not deleted,
    so "what was the prompt yesterday" stays answerable. ``activate=False`` stages a
    version without making it live (a draft).

    Retirement is a loop of single-document updates rather than one bulk update: the
    DocStore port is deliberately narrow (no ``update_many``), and procedures are
    written rarely enough that the loop costs nothing.
    """
    existing = runtime.docs().find_many(
        PROCEDURAL_COLLECTION, {"tenant": tenant, "kind": kind, "name": name}
    )
    version = 1 + max([int(e.get("version") or 0) for e in existing], default=0)

    if activate:
        for prior in existing:
            if prior.get("is_active"):
                runtime.docs().update_one(
                    PROCEDURAL_COLLECTION,
                    {"entry_id": prior.get("entry_id")},
                    {"$set": {"is_active": False}},
                )

    entry = {
        "entry_id": uuid.uuid4().hex,
        "tenant": tenant,
        "kind": kind,
        "name": name,
        "payload": dict(payload),
        "version": version,
        "is_active": bool(activate),
        "created_at": _now(),
    }
    runtime.docs().insert_one(PROCEDURAL_COLLECTION, entry)
    runtime.log().info(
        "procedure stored | kind=%s name=%s version=%d active=%s",
        kind,
        name,
        version,
        bool(activate),
    )
    return entry
