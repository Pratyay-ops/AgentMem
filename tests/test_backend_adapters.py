"""The backend reference adapters, exercised with injected fakes — no live DB.

Each adapter imports its driver lazily, so every test here constructs the adapter
with a fake connection/client and never touches psycopg / pymongo / opensearch-py —
except the OpenSearch kNN path, which needs the driver's ``NotFoundError`` type and
is skipped when the ``opensearch`` extra is not installed. What is pinned is the part
that could silently be wrong: the SQL the Postgres adapter builds, the Mongo
passthrough, and the OpenSearch query body + index mapping.
"""

from __future__ import annotations

import json
import types

import pytest

from agentmem.memory.adapters import mongodb, opensearch, postgres


# ============================================================ Postgres


def test_validate_ident_rejects_injection():
    with pytest.raises(ValueError):
        postgres._validate_ident("docs; drop table x")
    assert postgres._validate_ident("agentmem_documents") == "agentmem_documents"


def test_where_equality_is_containment_and_in_is_any():
    clauses, params = postgres._where(
        {"tenant": "t", "review_status": {"$in": ["pending", "rectified"]}}
    )
    assert clauses == ["doc @> %s::jsonb", "doc->>%s = ANY(%s)"]
    assert params == [json.dumps({"tenant": "t"}), "review_status", ["pending", "rectified"]]


def test_where_rejects_an_unknown_operator():
    with pytest.raises(NotImplementedError):
        postgres._where({"score": {"$gt": 1}})


def test_set_payload_only_accepts_set():
    assert postgres._set_payload({"$set": {"a": 1}}) == json.dumps({"a": 1})
    with pytest.raises(NotImplementedError):
        postgres._set_payload({"$inc": {"n": 1}})


class _PgCursor:
    def __init__(self, rows):
        self.executed = []
        self._rows = rows
        self.rowcount = 0

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        if sql.strip().upper().startswith("UPDATE"):
            self.rowcount = 1

    def fetchall(self):
        return self._rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _PgConn:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _last(cur, verb):
    return next(e for e in reversed(cur.executed) if e[0].strip().upper().startswith(verb))


def test_postgres_find_many_builds_filtered_sorted_limited_sql():
    cur = _PgCursor(rows=[({"tenant": "t", "statement": "x"},)])
    store = postgres.PostgresDocStore(connect=lambda: _PgConn(cur))

    out = store.find_many(
        "episode",
        {"tenant": "t", "review_status": {"$in": ["pending"]}},
        sort=[("created_at", -1)],
        limit=5,
    )

    assert out == [{"tenant": "t", "statement": "x"}]  # unwrapped from the row tuple
    sql, args = _last(cur, "SELECT")
    assert "WHERE collection = %s" in sql
    assert "doc @> %s::jsonb" in sql
    assert "doc->>%s = ANY(%s)" in sql
    assert "ORDER BY doc->>%s DESC" in sql
    assert sql.endswith("LIMIT %s")
    assert args == ["episode", json.dumps({"tenant": "t"}), "review_status", ["pending"], "created_at", 5]


def test_postgres_insert_writes_collection_and_json():
    cur = _PgCursor(rows=[])
    store = postgres.PostgresDocStore(connect=lambda: _PgConn(cur))

    store.insert_one("episode", {"episode_id": "e1", "statement": "x"})

    sql, args = _last(cur, "INSERT")
    assert "INSERT INTO agentmem_documents (collection, doc) VALUES (%s, %s::jsonb)" in sql
    assert args == ["episode", json.dumps({"episode_id": "e1", "statement": "x"})]


def test_postgres_update_one_merges_set_and_reports_match():
    cur = _PgCursor(rows=[])
    store = postgres.PostgresDocStore(connect=lambda: _PgConn(cur))

    result = store.update_one(
        "episode", {"episode_id": "e1"}, {"$set": {"review_status": "rejected"}}
    )

    sql, args = _last(cur, "UPDATE")
    assert "SET doc = doc || %s::jsonb" in sql
    assert args == [
        json.dumps({"review_status": "rejected"}),
        "episode",
        json.dumps({"episode_id": "e1"}),
    ]
    assert result == {"matched_count": 1, "modified_count": 1}


def test_postgres_requires_exactly_one_of_connect_or_dsn():
    with pytest.raises(ValueError):
        postgres.PostgresDocStore()
    with pytest.raises(ValueError):
        postgres.PostgresDocStore(connect=lambda: None, dsn="postgresql://x")


# ============================================================ MongoDB / DocumentDB


class _MongoCursor:
    def __init__(self, docs):
        self._docs = list(docs)
        self.sorted = None
        self.limited = None

    def sort(self, spec):
        self.sorted = spec
        return self

    def limit(self, n):
        self.limited = n
        self._docs = self._docs[:n]
        return self

    def __iter__(self):
        return iter(self._docs)


class _MongoCollection:
    def __init__(self):
        self.docs = []
        self.last_cursor = None

    def find_one(self, query, projection=None):
        return self.docs[0] if self.docs else None

    def find(self, query, projection=None):
        self.last_cursor = _MongoCursor(self.docs)
        return self.last_cursor

    def insert_one(self, doc):
        self.docs.append(doc)

    def update_one(self, query, update):
        return types.SimpleNamespace(matched_count=1, modified_count=1)


class _MongoDB:
    def __init__(self):
        self.collections = {}

    def __getitem__(self, name):
        return self.collections.setdefault(name, _MongoCollection())


def test_mongo_passthrough_insert_find_and_update():
    db = _MongoDB()
    store = mongodb.MongoDocStore(db)

    store.insert_one("episode", {"episode_id": "e1", "statement": "x"})
    assert store.find_one("episode", {"episode_id": "e1"})["statement"] == "x"

    rows = store.find_many("episode", {}, sort=[("created_at", -1)], limit=3)
    assert rows == [{"episode_id": "e1", "statement": "x"}]
    # sort/limit reached the cursor untranslated.
    assert db["episode"].last_cursor.sorted == [("created_at", -1)]
    assert db["episode"].last_cursor.limited == 3

    assert store.update_one("episode", {"episode_id": "e1"}, {"$set": {"review_status": "rejected"}}) == {
        "matched_count": 1,
        "modified_count": 1,
    }


def test_mongo_insert_copies_so_pymongo_id_cannot_leak_back():
    db = _MongoDB()
    store = mongodb.MongoDocStore(db)
    original = {"episode_id": "e1"}
    store.insert_one("episode", original)
    # The stored dict is a copy; a pymongo-injected _id would not appear on the caller's.
    db["episode"].docs[0]["_id"] = "injected"
    assert "_id" not in original


# ============================================================ OpenSearch


def test_os_filter_translation_and_index_body():
    assert opensearch._to_os_filters({"tenant": "t", "review_status": {"$in": ["pending"]}}) == [
        {"term": {"tenant": "t"}},
        {"terms": {"review_status": ["pending"]}},
    ]
    body = opensearch._knn_index_body(1536)
    assert body["settings"]["index"]["knn"] is True
    assert body["mappings"]["properties"]["embedding"] == {"type": "knn_vector", "dimension": 1536}


class _OSIndices:
    def __init__(self):
        self.existing = set()
        self.created = []

    def exists(self, index):
        return index in self.existing

    def create(self, index, body):
        self.created.append((index, body))
        self.existing.add(index)


class _OSClient:
    def __init__(self, response=None, raise_exc=None):
        self.indices = _OSIndices()
        self.indexed = []
        self.searched = None
        self._response = response or {"hits": {"hits": []}}
        self._raise = raise_exc

    def index(self, index, id, body, refresh):
        self.indexed.append((index, id, body, refresh))
        self.indices.existing.add(index)

    def search(self, index, body):
        self.searched = body
        if self._raise is not None:
            raise self._raise
        return self._response


def test_os_index_creates_mapping_then_writes_vector():
    client = _OSClient()
    store = opensearch.OpenSearchVectorStore(client, refresh=True)

    store.index("episode_knn", "e1", [0.1, 0.2], {"tenant": "t", "statement": "x"})

    idx, body = client.indices.created[0]
    assert idx == "episode_knn"
    assert body["mappings"]["properties"]["embedding"]["dimension"] == 2
    assert client.indexed[0] == (
        "episode_knn",
        "e1",
        {"tenant": "t", "statement": "x", "embedding": [0.1, 0.2]},
        True,
    )


def test_os_knn_filters_in_query_and_excludes_vector():
    pytest.importorskip("opensearchpy")
    client = _OSClient(response={"hits": {"hits": [{"_source": {"tenant": "t"}}]}})
    store = opensearch.OpenSearchVectorStore(client)

    hits = store.knn("episode_knn", [0.1], k=5, filters={"tenant": "t", "review_status": {"$in": ["pending"]}})

    assert hits == [{"tenant": "t"}]
    body = client.searched
    assert body["_source"]["excludes"] == ["embedding"]
    assert body["query"]["bool"]["filter"] == [
        {"term": {"tenant": "t"}},
        {"terms": {"review_status": ["pending"]}},
    ]


def test_os_knn_missing_index_reads_as_no_neighbours():
    pytest.importorskip("opensearchpy")
    from opensearchpy.exceptions import NotFoundError

    client = _OSClient(raise_exc=NotFoundError(404, "index_not_found", {}))
    store = opensearch.OpenSearchVectorStore(client)
    assert store.knn("episode_knn", [0.1], k=3, filters={"tenant": "t"}) == []
