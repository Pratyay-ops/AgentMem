"""In-memory implementations of the five ports — zero dependencies, no database.

Real enough to exercise the tiers end to end: the DocStore matches queries, the
VectorStore does a genuine cosine-ranked kNN with the SAME in-query filtering the
real stores must do, and the Embedder is deterministic so a test is reproducible.
Not meant for production — meant so the library is provable without cloud.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional

_DIM = 32


def _vector(text: str) -> List[float]:
    """A deterministic bag-of-characters vector. Not semantic — reproducible."""
    vec = [0.0] * _DIM
    for ch in (text or "").lower():
        vec[ord(ch) % _DIM] += 1.0
    return vec


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _matches(payload: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    for field, expected in (filters or {}).items():
        actual = payload.get(field)
        if isinstance(expected, dict) and "$in" in expected:
            if actual not in expected["$in"]:
                return False
        elif actual != expected:
            return False
    return True


class InMemoryDocs:
    """A DocStore backed by ``{collection: [docs]}``."""

    def __init__(self) -> None:
        self.collections: Dict[str, List[Dict[str, Any]]] = {}

    def _rows(self, collection: str) -> List[Dict[str, Any]]:
        return self.collections.setdefault(collection, [])

    def find_one(self, collection, query, projection=None):
        for row in self._rows(collection):
            if _matches(row, query):
                return copy.deepcopy(row)
        return None

    def find_many(self, collection, query, projection=None, *, sort=None, limit=0):
        hits = [copy.deepcopy(r) for r in self._rows(collection) if _matches(r, query)]
        return hits[:limit] if limit else hits

    def insert_one(self, collection, doc):
        self._rows(collection).append(copy.deepcopy(doc))

    def update_one(self, collection, query, update):
        for row in self._rows(collection):
            if _matches(row, query):
                row.update(update.get("$set", update))
                return copy.deepcopy(row)
        return {}


class InMemoryVectors:
    """A VectorStore with real cosine kNN and IN-QUERY filtering."""

    def __init__(self) -> None:
        self.indexes: Dict[str, List[Dict[str, Any]]] = {}

    def index(self, index_name, doc_id, vector, payload):
        entries = self.indexes.setdefault(index_name, [])
        entries[:] = [e for e in entries if e["id"] != doc_id]
        entries.append({"id": doc_id, "vector": list(vector), "payload": copy.deepcopy(payload)})

    def knn(self, index_name, vector, *, k, filters):
        scored = [
            (_cosine(vector, e["vector"]), e["payload"])
            for e in self.indexes.get(index_name, [])
            if _matches(e["payload"], filters)  # filter INSIDE the query
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [copy.deepcopy(payload) for _score, payload in scored[:k]]


class InMemoryEmbedder:
    async def embed(self, texts: List[str]) -> List[List[float]]:
        return [_vector(t) for t in texts]


class StubChat:
    async def complete(self, system: str, user: str, *, model: str) -> str:
        return ""


class StdLog:
    def info(self, msg, *args):
        pass

    def warning(self, msg, *args, **kwargs):
        pass
