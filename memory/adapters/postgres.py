"""PostgreSQL (JSONB) reference adapter for the DocStore port.

Store the memory documents as ``jsonb`` rows in a Postgres you already run — RDS,
Aurora, a container, anything. ``psycopg`` (v3) is an OPTIONAL dependency — install
``agentmem[postgres]`` — imported lazily, so ``import agentmem`` never needs it.

**Documents live in one table, ``agentmem_documents(collection text, doc jsonb)``.**
A GIN index on ``doc`` makes the containment lookups the tiers issue fast. The
adapter supports exactly the query surface ``agentmem`` uses and refuses the rest,
because a store that silently matches everything on an operator it does not
understand is worse than one that raises:

* **filters** — equality on top-level fields (via ``@>`` containment) and ``$in``
  (via ``= ANY``). ``$in`` compares the field's TEXT form, which is what the
  episodic status filter needs; it is not a general numeric-range operator.
* **updates** — ``$set`` only (merged with ``||``). ``$inc`` / ``$addToSet`` raise.

Each call opens its own connection (from the DSN, or the ``connect`` callable you
pass — hand it a pool's ``connection`` for production), so the adapter is safe to
call concurrently from the worker threads a host offloads these sync methods to.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_ident(name: str) -> str:
    """A table name is interpolated into SQL, so it must be a bare identifier."""
    if not _IDENT.match(name or ""):
        raise ValueError(f"unsafe table name: {name!r}")
    return name


def _where(query: Dict[str, Any]) -> Tuple[List[str], List[Any]]:
    """Build the WHERE fragments (excluding ``collection``) and their params.

    Equality becomes a JSONB containment test; ``{"$in": [...]}`` becomes a text
    ``= ANY`` over the extracted field. Kept pure so it is testable without a
    database — the SQL surface is the part worth pinning.
    """
    clauses: List[str] = []
    params: List[Any] = []
    for field, expected in (query or {}).items():
        if isinstance(expected, dict) and "$in" in expected:
            clauses.append("doc->>%s = ANY(%s)")
            params.append(field)
            params.append([str(v) for v in expected["$in"]])
        elif isinstance(expected, dict):
            raise NotImplementedError(
                f"PostgresDocStore: unsupported filter operator(s) {list(expected)!r}"
            )
        else:
            clauses.append("doc @> %s::jsonb")
            params.append(json.dumps({field: expected}))
    return clauses, params


def _set_payload(update: Dict[str, Any]) -> str:
    """The ``$set`` body as JSON, or raise on any other update operator."""
    extra = set(update or {}) - {"$set"}
    if extra:
        raise NotImplementedError(
            f"PostgresDocStore.update_one supports only $set, got {sorted(extra)!r}"
        )
    return json.dumps(update.get("$set", {}))


class PostgresDocStore:
    """A :class:`DocStore` over a Postgres ``jsonb`` table.

    Args:
        connect: A zero-arg callable returning a context-manager connection (e.g. a
            psycopg connection or a pool's ``connection()``). Mutually exclusive with
            ``dsn``.
        dsn: A libpq connection string; the adapter opens a fresh psycopg connection
            per call from it. Mutually exclusive with ``connect``.
        table: The documents table name (validated as a bare identifier).
    """

    def __init__(
        self,
        connect: Optional[Callable[[], Any]] = None,
        *,
        dsn: Optional[str] = None,
        table: str = "agentmem_documents",
    ) -> None:
        if (connect is None) == (dsn is None):
            raise ValueError("provide exactly one of connect= or dsn=")
        self._connect = connect
        self._dsn = dsn
        self._table = _validate_ident(table)
        self._schema_ready = False

    @classmethod
    def from_dsn(cls, dsn: str, *, table: str = "agentmem_documents") -> "PostgresDocStore":
        """Open connections per call from a libpq DSN (lazy import)."""
        return cls(dsn=dsn, table=table)

    def _conn(self) -> Any:
        if self._connect is not None:
            return self._connect()
        import psycopg

        return psycopg.connect(self._dsn)

    def _ensure_schema(self, cur: Any) -> None:
        if self._schema_ready:
            return
        cur.execute(
            f"CREATE TABLE IF NOT EXISTS {self._table} "
            f"(collection text NOT NULL, doc jsonb NOT NULL)"
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS {self._table}_collection_idx "
            f"ON {self._table} (collection)"
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS {self._table}_doc_gin "
            f"ON {self._table} USING gin (doc)"
        )
        self._schema_ready = True

    def find_one(
        self, collection: str, query: Dict[str, Any], projection: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        rows = self.find_many(collection, query, projection, limit=1)
        return rows[0] if rows else None

    def find_many(
        self,
        collection: str,
        query: Dict[str, Any],
        projection: Optional[Dict[str, Any]] = None,
        *,
        sort: Optional[List] = None,
        limit: int = 0,
    ) -> List[Dict[str, Any]]:
        clauses, params = _where(query)
        sql = f"SELECT doc FROM {self._table} WHERE collection = %s"
        args: List[Any] = [collection]
        if clauses:
            sql += " AND " + " AND ".join(clauses)
            args += params
        if sort:
            order_parts: List[str] = []
            for field, direction in sort:
                order_parts.append("doc->>%s " + ("DESC" if int(direction) < 0 else "ASC"))
                args.append(field)
            sql += " ORDER BY " + ", ".join(order_parts)
        if limit:
            sql += " LIMIT %s"
            args.append(int(limit))
        with self._conn() as conn, conn.cursor() as cur:
            self._ensure_schema(cur)
            cur.execute(sql, args)
            return [row[0] for row in cur.fetchall()]

    def insert_one(self, collection: str, doc: Dict[str, Any]) -> None:
        with self._conn() as conn, conn.cursor() as cur:
            self._ensure_schema(cur)
            cur.execute(
                f"INSERT INTO {self._table} (collection, doc) VALUES (%s, %s::jsonb)",
                [collection, json.dumps(dict(doc))],
            )

    def update_one(
        self, collection: str, query: Dict[str, Any], update: Dict[str, Any]
    ) -> Dict[str, Any]:
        payload = _set_payload(update)
        clauses, params = _where(query)
        where = "collection = %s" + ("" if not clauses else " AND " + " AND ".join(clauses))
        sql = (
            f"UPDATE {self._table} SET doc = doc || %s::jsonb "
            f"WHERE ctid = (SELECT ctid FROM {self._table} WHERE {where} LIMIT 1)"
        )
        with self._conn() as conn, conn.cursor() as cur:
            self._ensure_schema(cur)
            cur.execute(sql, [payload, collection, *params])
            affected = cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
        return {"matched_count": affected, "modified_count": affected}
