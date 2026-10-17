# AgentMem — tiered memory for LLM agents

A **storage-agnostic** memory layer for LLM agents. The library owns the memory
*model*; **you** own the databases. Point it at infrastructure you already run and
audit — nothing is hosted, nothing leaves your deployment, and the core installs with
a single dependency (`pydantic`).

```python
from agentmem import memory

memory.configure(docs=..., vectors=..., embedder=..., chat=..., log=...)
scope = memory.MemoryScope(tenant="acme", scope="doc-42", user="u1")

facts = await memory.semantic.recall("what does AE mean here?", scope=scope)
await memory.episodic.record(
    kind=memory.EpisodeKind.ASSUMPTION,
    statement="dosing is stated in mg/kg here",
    what_i_did="proceeded on that",
    scope=scope,
)
```

## Why this exists

Most "agent memory" is either a hosted black box or a thin wrapper that pins you to
one vector database and one embedding provider. This library inverts that: it defines
the memory **behaviour** — what a semantic hit is, how an episodic assumption is
scoped and reviewed, why a memory may never hold a raw id — and leaves every byte of
storage to you. If you already run Postgres, DocumentDB/MongoDB and OpenSearch, you
wire those in and keep your data where it is (see
[Connect your own storage](#connect-your-own-storage)).

## The four tiers (CoALA — [arXiv:2309.02427](https://arxiv.org/abs/2309.02427))

| Tier | What it holds | Who writes it |
|---|---|---|
| **working** | per-conversation scratch | the host (a checkpointer) |
| **semantic** | curated domain knowledge — glossaries, data definitions, rules | **humans**, read-only to the agent |
| **episodic** | the agent's own **assumptions and contradictions** — the gaps, not a transcript | the agent, per user, human-reviewed |
| **procedural** | the agent/workflow registry | the host |

This library **implements `semantic` and `episodic`**. `working` (a checkpointer) and
`procedural` (your agent/workflow registry) are host concerns — the same ports and
scope model apply, but the library does not own them.

## Design positions

Deliberate divergences from the common "just embed everything" approach:

1. **Semantic memory is curated, not learned.** Auto-extraction produces a store that
   contradicts itself and goes stale. Domain knowledge is authored and versioned by
   humans; the agent reads it. Retrieval is **exact term/synonym match first, then
   kNN** — "what does AE mean" must hit the `AE` entry directly, not hope embedding
   similarity ranks it first.
2. **The agent records what it did NOT know.** Episodic memory captures assumptions
   and contradictions, not successful turns. The loop terminates: the log empties as
   the knowledge base fills.
3. **Promotion is the only path to shared knowledge, and it passes a human.** An
   episodic entry is scoped to the user who wrote it and stays `pending` until a
   reviewer rejects it or promotes it into semantic memory. A `rejected` entry is
   never recalled again — a recorded wrong answer fed back is worse than no memory.
4. **No memory may contain a raw identifier.** Entries store human-addressable
   handles ("4.2 Safety"), never a database id, and the host re-resolves them per
   turn. A memory that embedded an id could silently point at the wrong record in
   another tenant. The guard is enforced on write.

## The scope model

The library's ONLY vocabulary for "who and where" is `MemoryScope(tenant, scope,
user)` — three opaque strings. It never learns what a "document", "report" or
"workspace" is; the host maps its own ids onto these keys before calling in.

- `tenant` isolates one customer/org (hard boundary — filtered inside every query).
- `scope` narrows within a tenant (a document, a workspace — your choice).
- `user` attributes episodic writes, so one person's unreviewed assumptions never
  leak into another's recall.

A **workspace-visibility** semantic entry shadows a **global** one for the same term,
and the shadowing is surfaced on the result — a silently different definition of a
term is worse than either definition alone.

## Install

```bash
pip install agentmem                 # core: pydantic only
pip install "agentmem[postgres]"     # + PostgresDocStore     (psycopg 3)
pip install "agentmem[mongodb]"      # + MongoDocStore        (pymongo; MongoDB / DocumentDB)
pip install "agentmem[opensearch]"   # + OpenSearchVectorStore (opensearch-py)
pip install "agentmem[postgres,opensearch]"   # mix and match
```

## Quickstart (no database)

The in-memory reference adapters implement every port with real cosine-ranked kNN and
in-query filtering, so the quickstart, your tests, and an eval harness all run with
nothing to stand up:

```python
from agentmem import memory
from agentmem.memory import EpisodeKind
from agentmem.memory.adapters.inmemory import (
    InMemoryDocs, InMemoryVectors, InMemoryEmbedder, StubChat, StdLog,
)

memory.configure(
    docs=InMemoryDocs(),
    vectors=InMemoryVectors(),
    embedder=InMemoryEmbedder(),
    chat=StubChat(),
    log=StdLog(),
)

scope = memory.MemoryScope(tenant="acme", scope="doc-42", user="u1")

# semantic (curated, read-only): exact term/synonym match first, then kNN
hits = await memory.semantic.recall("what does AE mean here?", scope=scope, k=5)

# episodic (agent-written): record a gap, later recall your own past assumptions
await memory.episodic.record(
    kind=EpisodeKind.ASSUMPTION,
    statement="dosing is stated in mg/kg here",
    what_i_did="proceeded on that",
    scope=scope,
)
prior = await memory.episodic.recall("dosing units", scope=scope, k=5)
```

## Connect your own storage

Everything is injected. The library declares five Protocols in
[`memory/ports.py`](memory/ports.py) — `DocStore`, `VectorStore`, `Embedder`, `Chat`,
`Log` — and you supply implementations and call `configure()` once at startup. It
ships reference adapters for the three backends most agent stacks already operate, so
connecting to hosted Postgres, DocumentDB/MongoDB or OpenSearch is a constructor call,
not a rewrite:

```python
from agentmem import memory
from agentmem.memory.adapters.postgres import PostgresDocStore
from agentmem.memory.adapters.opensearch import OpenSearchVectorStore

# Documents in a Postgres you already run (JSONB; the table is created on first use)
docs = PostgresDocStore.from_dsn("postgresql://user:pass@host:5432/appdb")

# ...or in MongoDB / AWS DocumentDB (same wire protocol, same adapter):
#   from agentmem.memory.adapters.mongodb import MongoDocStore
#   docs = MongoDocStore.from_uri("mongodb://host:27017", "appdb",
#                                 tls=True, tlsCAFile="global-bundle.pem", retryWrites=False)

# kNN over your OpenSearch cluster (auth/TLS/SigV4 pass straight through)
vectors = OpenSearchVectorStore.from_hosts(
    [{"host": "search.example.internal", "port": 443}],
    http_auth=("user", "pass"), use_ssl=True,
)

memory.configure(
    docs=docs,
    vectors=vectors,
    embedder=my_embedder,   # any object with `async embed(texts) -> list[list[float]]`
    chat=my_chat,           # any object with `async complete(system, user, *, model) -> str`
    log=my_logger,          # anything with info()/warning()
)
```

What each adapter guarantees, so behaviour never drifts by backend:

- **`OpenSearchVectorStore`** applies filters INSIDE the kNN query (never as a
  post-filter, which silently returns fewer than `k`), and creates each index with a
  `knn_vector` mapping whose dimension it reads from your first vector — no embedding
  model dimension is hardcoded.
- **`MongoDocStore`** passes filters and updates straight through — the port's
  vocabulary already IS Mongo's (`$in`, `$set`, …) — so there is no second dialect to
  keep in step. Serves MongoDB and AWS DocumentDB alike.
- **`PostgresDocStore`** stores documents as `jsonb` in one GIN-indexed table,
  translating equality to `@>` containment and `$in` to `= ANY`. It supports exactly
  the query surface the tiers use and **raises on anything else** rather than
  silently matching wrong.

`Embedder` and `Chat` are yours to provide — that is where provider choice
(OpenAI, Bedrock, a local model) belongs, and why the library depends on none of them.

### Writing an adapter

An adapter is a plain class satisfying a Protocol — no base class, no registration.
The `VectorStore` contract has one rule worth repeating: **filter inside the query.**

```python
class MyVectorStore:
    def index(self, index_name, doc_id, vector, payload): ...
    def knn(self, index_name, vector, *, k, filters):
        # `filters` is e.g. {"tenant": "acme", "review_status": {"$in": ["pending"]}}
        # apply it as part of the ANN query — a post-filter returns < k and reads as
        # "no relevant memory".
        ...
```

## Testing

```bash
pip install -e ".[test]" && pytest
```

The suite runs entirely on the in-memory adapters — no database, no network, and
deterministic embeddings — so it is fast and reproducible in CI.

## Status

`semantic` and `episodic` are implemented and tested; the in-memory, Postgres,
MongoDB/DocumentDB and OpenSearch adapters ship in the box. `working` and
`procedural` are host tiers (a checkpointer and your agent registry) that consume the
same scope model. The public API is `configure()` + `MemoryScope` + `semantic` +
`episodic`; everything else is an implementation detail.

## License

See [LICENSE](LICENSE).
