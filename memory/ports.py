"""The storage seam: Protocols the host implements and injects.

The library declares WHAT it needs; a consumer supplies the implementations and
calls :func:`~agentmem.memory.configure` once at startup. This is the whole
reason the library is storage-agnostic and testable with no database — the eval
harness and the in-memory reference adapters implement the same Protocols.

Nothing here interprets the payloads: a ``dict`` is a ``dict``, a ``collection`` is
an opaque name. The library never learns what a "report" is.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class DocStore(Protocol):
    """A document store: Mongo/DocumentDB/Postgres-JSON, anything with these five."""

    def find_one(
        self, collection: str, query: Dict[str, Any], projection: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]: ...

    def find_many(
        self,
        collection: str,
        query: Dict[str, Any],
        projection: Optional[Dict[str, Any]] = None,
        *,
        sort: Optional[List] = None,
        limit: int = 0,
    ) -> List[Dict[str, Any]]: ...

    def insert_one(self, collection: str, doc: Dict[str, Any]) -> None: ...

    def update_one(
        self, collection: str, query: Dict[str, Any], update: Dict[str, Any]
    ) -> Dict[str, Any]: ...


@runtime_checkable
class VectorStore(Protocol):
    """A vector index: OpenSearch/pgvector/Elasticsearch, anything with kNN.

    ``knn`` MUST apply ``filters`` INSIDE the query, never after: a post-filter over
    a top-k neighbour set silently returns fewer results than asked, and the
    shortfall reads as "no relevant memory" rather than "filtered out".
    """

    def index(self, index_name: str, doc_id: str, vector: List[float], payload: Dict[str, Any]) -> None: ...

    def knn(
        self, index_name: str, vector: List[float], *, k: int, filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]: ...


@runtime_checkable
class Embedder(Protocol):
    async def embed(self, texts: List[str]) -> List[List[float]]: ...


@runtime_checkable
class Chat(Protocol):
    async def complete(self, system: str, user: str, *, model: str) -> str: ...


@runtime_checkable
class Log(Protocol):
    def info(self, msg: str, *args: Any) -> None: ...

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None: ...
