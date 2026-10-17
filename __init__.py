"""agentmem — a tiered, storage-agnostic memory layer for LLM agents.

The library owns the memory *model* (working / semantic / episodic / procedural);
you own the storage. It declares Protocols for a document store, a vector index, an
embedder, a chat model and a logger, and you inject implementations at startup — so
it runs on infrastructure you already operate and never hosts anything itself.

The import name maps to the repo root, so it resolves both as a `pip install` and as
a vendored submodule directory named ``agentmem/`` with no install step. See
``memory/`` for the public API and the README for the design.
"""

__version__ = "0.1.0"
