"""Semantic memory: retrieve curated domain knowledge. READ-ONLY.

Nothing here writes — semantic knowledge is authored and versioned by humans, so
there is no ADD/UPDATE/reconcile problem and no staleness the library has to
manage. Retrieval is **exact term/synonym match FIRST, then kNN**: "what does AE
mean here" must hit the ``AE`` abbreviation directly, not depend on embedding
similarity ranking it above six adjacent terms.

A **workspace** entry shadows a **global** one for the same term, and the shadowing
is marked on the result — a silently different definition of a term in one scope is
worse than either definition, because nobody can tell which one was used.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from agentmem.memory import runtime
from agentmem.memory.schemas import DomainFact, MemoryScope

SEMANTIC_COLLECTION = "semantic_fact"
SEMANTIC_INDEX = "semantic_fact_knn"

# Past this a retrieved block costs more prompt space than it saves; the caller
# should never render the whole base.
DEFAULT_K = 8


def _terms(fact: DomainFact) -> List[str]:
    out = [fact.term, fact.abbreviation, *fact.synonyms]
    return [t.strip().lower() for t in out if t and t.strip()]


def _mentions(fact: DomainFact, question: str) -> bool:
    ql = f" {question.lower()} "
    return any(re.search(rf"\b{re.escape(t)}\b", ql) for t in _terms(fact))


def _dedupe_with_shadow(facts: List[DomainFact], scope: MemoryScope) -> List[DomainFact]:
    """Workspace shadows global for the same term; keep first-seen order otherwise."""
    chosen: Dict[str, DomainFact] = {}
    order: List[str] = []
    for fact in facts:
        key = (fact.term or fact.statement or fact.fact_id).strip().lower()
        current = chosen.get(key)
        if current is None:
            chosen[key] = fact
            order.append(key)
        elif fact.visibility == "workspace" and fact.scope == scope.scope:
            chosen[key] = fact  # the local override wins
    return [chosen[k] for k in order]


async def recall(question: str, *, scope: MemoryScope, k: int = DEFAULT_K) -> List[DomainFact]:
    """Curated facts relevant to ``question``, exact matches first, capped at ``k``."""
    rows = runtime.docs().find_many(
        SEMANTIC_COLLECTION,
        {"tenant": scope.tenant, "is_active": True},
    )
    candidates = [DomainFact(**row) for row in rows]
    # A workspace entry applies only to its own scope; a global entry always applies.
    visible = [
        f for f in candidates
        if f.visibility == "global" or (f.visibility == "workspace" and f.scope == scope.scope)
    ]

    exact = [f for f in visible if _mentions(f, question)]
    if len(_dedupe_with_shadow(exact, scope)) >= k:
        return _dedupe_with_shadow(exact, scope)[:k]

    # kNN fallback for the remainder — the exact hits stay first.
    vector = (await runtime.embedder().embed([question]))[0]
    hits = runtime.vectors().knn(
        SEMANTIC_INDEX, vector, k=k, filters={"tenant": scope.tenant, "is_active": True}
    )
    knn_facts = [DomainFact(**h) for h in hits]
    return _dedupe_with_shadow(exact + knn_facts, scope)[:k]


def render(facts: List[DomainFact]) -> str:
    """The prompt block for a set of curated facts — each type rendered in its own
    shape so a glossary pair keeps its term/definition pairing."""
    lines: List[str] = []
    for fact in facts:
        if fact.entry_type.value == "glossary":
            syn = f" (a.k.a. {', '.join(fact.synonyms)})" if fact.synonyms else ""
            lines.append(f"- {fact.term}{syn}: {fact.definition}")
        elif fact.entry_type.value == "data_definition":
            unit = f" [{fact.unit}]" if fact.unit else ""
            lines.append(f"- {fact.term}{unit}: {fact.definition}")
        else:
            lines.append(f"- {fact.statement}")
    return "\n".join(lines)
