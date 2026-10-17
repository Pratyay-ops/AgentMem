"""Reference adapters.

``inmemory`` implements every port with no database, so the quickstart and the test
suite (and the eval harness) run with nothing to stand up.

The backend adapters connect to infrastructure you already run. Each imports its
driver LAZILY and ships behind an optional extra, so importing this package (or
``agentmem`` at all) pulls in no database driver — install only what you use:

* ``opensearch`` → :class:`.opensearch.OpenSearchVectorStore`  (``agentmem[opensearch]``)
* ``mongodb``    → :class:`.mongodb.MongoDocStore`             (``agentmem[mongodb]``)
* ``postgres``   → :class:`.postgres.PostgresDocStore`         (``agentmem[postgres]``)

They are intentionally NOT re-exported here: importing this package must never fail
because a driver you do not use is absent. Import the one you want directly, e.g.
``from agentmem.memory.adapters.opensearch import OpenSearchVectorStore``.
"""
