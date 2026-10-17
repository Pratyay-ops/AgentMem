"""MongoDB / AWS DocumentDB reference adapter for the DocStore port.

Point it at a MongoDB deployment or an AWS DocumentDB cluster (DocumentDB speaks the
MongoDB wire protocol, so the same adapter serves both). ``pymongo`` is an OPTIONAL
dependency — install ``agentmem[mongodb]`` — imported lazily, so ``import agentmem``
never needs it.

This is the thinnest adapter of the three because the port's vocabulary IS Mongo's:
the filter dicts (equality, ``$in``) and update documents (``$set``, ``$inc``,
``$addToSet``) pass straight through to ``pymongo`` with no translation. That is not
laziness — a translation layer here would be a second dialect to keep in step with
the one the port already speaks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class MongoDocStore:
    """A :class:`DocStore` over a ``pymongo`` database handle.

    Args:
        database: A ``pymongo.database.Database`` — one logical database whose
            collection names are the ``collection`` arguments the tiers pass.
    """

    def __init__(self, database: Any) -> None:
        self._db = database

    @classmethod
    def from_uri(cls, uri: str, database: str, **client_kwargs: Any) -> "MongoDocStore":
        """Connect from a URI and database name (lazy import).

        ``client_kwargs`` reach ``pymongo.MongoClient`` untouched — TLS material, a
        DocumentDB CA bundle, ``retryWrites=false`` (DocumentDB requires it), read
        preference, and so on. The adapter assumes nothing about the deployment.
        """
        from pymongo import MongoClient

        return cls(MongoClient(uri, **client_kwargs)[database])

    def find_one(
        self, collection: str, query: Dict[str, Any], projection: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        return self._db[collection].find_one(query, projection)

    def find_many(
        self,
        collection: str,
        query: Dict[str, Any],
        projection: Optional[Dict[str, Any]] = None,
        *,
        sort: Optional[List] = None,
        limit: int = 0,
    ) -> List[Dict[str, Any]]:
        cursor = self._db[collection].find(query, projection)
        if sort:
            cursor = cursor.sort(list(sort))
        if limit:
            cursor = cursor.limit(int(limit))
        return list(cursor)

    def insert_one(self, collection: str, doc: Dict[str, Any]) -> None:
        # Copy so pymongo's injected ``_id`` never mutates the caller's dict.
        self._db[collection].insert_one(dict(doc))

    def update_one(
        self, collection: str, query: Dict[str, Any], update: Dict[str, Any]
    ) -> Dict[str, Any]:
        result = self._db[collection].update_one(query, update)
        return {
            "matched_count": result.matched_count,
            "modified_count": result.modified_count,
        }
