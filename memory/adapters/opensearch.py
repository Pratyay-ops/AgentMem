"""OpenSearch reference adapter for the :class:`~agentmem.memory.ports.VectorStore`.

Point it at an OpenSearch (or Elasticsearch-compatible) cluster you already run.
``opensearch-py`` is an OPTIONAL dependency — install ``agentmem[opensearch]`` — and
it is imported lazily, so ``import agentmem`` never needs it.

Two contract details are load-bearing and implemented here:

* **Filters are applied INSIDE the kNN query.** The port forbids post-filtering a
  top-k neighbour set: it silently returns fewer than ``k`` and reads as "no
  relevant memory". A scalar filter becomes a ``term``, ``{"$in": [...]}`` a
  ``terms``.
* **The index owns its mapping.** ``agentmem`` creates the kNN indexes, so every
  string field maps to ``keyword`` (a ``term`` on a default ``text`` field matches
  nothing) and ``embedding`` is a ``knn_vector`` whose dimension is taken from the
  first vector — no embedding-model dimension is hardcoded.
"""

from __future__ import annotations

from typing import Any, Dict, List


def _to_os_filters(filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Translate the plain filter dict into OpenSearch ``term``/``terms`` clauses."""
    clauses: List[Dict[str, Any]] = []
    for field, expected in (filters or {}).items():
        if isinstance(expected, dict) and "$in" in expected:
            clauses.append({"terms": {field: list(expected["$in"])}})
        else:
            clauses.append({"term": {field: expected}})
    return clauses


def _knn_index_body(dimension: int) -> Dict[str, Any]:
    """Index body for a kNN index: keyword-mapped strings + a ``knn_vector`` field."""
    return {
        "settings": {"index": {"knn": True}},
        "mappings": {
            "dynamic_templates": [
                {
                    "strings_as_keyword": {
                        "match_mapping_type": "string",
                        "mapping": {"type": "keyword"},
                    }
                }
            ],
            "properties": {"embedding": {"type": "knn_vector", "dimension": dimension}},
        },
    }


class OpenSearchVectorStore:
    """A :class:`VectorStore` over an ``opensearchpy.OpenSearch`` client.

    Args:
        client: A live ``opensearchpy.OpenSearch`` (or compatible) client.
        refresh: Whether to refresh the index on each write so a freshly recorded
            memory is immediately searchable. Defaults to True — memory writes are
            rare and "record then can't recall in the same turn" is a confusing bug.
    """

    def __init__(self, client: Any, *, refresh: bool = True) -> None:
        self._client = client
        self._refresh = refresh

    @classmethod
    def from_hosts(
        cls, hosts: Any, *, refresh: bool = True, **client_kwargs: Any
    ) -> "OpenSearchVectorStore":
        """Build the underlying client from connection settings (lazy import).

        ``hosts`` and ``client_kwargs`` are passed straight to
        ``opensearchpy.OpenSearch`` — e.g. ``http_auth``, ``use_ssl``,
        ``connection_class``, an AWS SigV4 signer. Nothing here assumes a cloud.
        """
        from opensearchpy import OpenSearch

        return cls(OpenSearch(hosts=hosts, **client_kwargs), refresh=refresh)

    def index(
        self, index_name: str, doc_id: str, vector: List[float], payload: Dict[str, Any]
    ) -> None:
        self._ensure_index(index_name, len(vector))
        body = {**payload, "embedding": list(vector)}
        self._client.index(index=index_name, id=doc_id, body=body, refresh=self._refresh)

    def knn(
        self, index_name: str, vector: List[float], *, k: int, filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        from opensearchpy.exceptions import NotFoundError

        body = {
            "size": k,
            "_source": {"excludes": ["embedding"]},
            "query": {
                "bool": {
                    "filter": _to_os_filters(filters),
                    "must": {"knn": {"embedding": {"vector": list(vector), "k": k}}},
                }
            },
        }
        try:
            response = self._client.search(index=index_name, body=body)
        except NotFoundError:
            # The index does not exist yet — a normal state before the first write,
            # so "no neighbours", not an error.
            return []
        hits = response.get("hits", {}).get("hits", [])
        return [hit.get("_source", {}) for hit in hits]

    def _ensure_index(self, index_name: str, dimension: int) -> None:
        if not self._client.indices.exists(index=index_name):
            self._client.indices.create(index=index_name, body=_knn_index_body(dimension))
