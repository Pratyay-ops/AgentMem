"""The injected storage, held once. ``configure()`` sets it; the tiers read it.

Kept in its own module so ``semantic`` and ``episodic`` can reach the adapters
without importing the package ``__init__`` (which imports them) — the usual
circular-import sidestep. Nothing here is CAF-specific; it is five slots.
"""

from __future__ import annotations

from typing import Optional

from agentmem.memory.ports import Chat, DocStore, Embedder, Log, VectorStore


class NotConfigured(RuntimeError):
    """A tier was used before ``memory.configure(...)`` supplied its storage."""


_docs: Optional[DocStore] = None
_vectors: Optional[VectorStore] = None
_embedder: Optional[Embedder] = None
_chat: Optional[Chat] = None
_log: Optional[Log] = None


def configure(*, docs: DocStore, vectors: VectorStore, embedder: Embedder, chat: Chat, log: Log) -> None:
    """Inject the storage the memory tiers run on. Call once at startup."""
    global _docs, _vectors, _embedder, _chat, _log
    _docs, _vectors, _embedder, _chat, _log = docs, vectors, embedder, chat, log


def _require(value, name: str):
    if value is None:
        raise NotConfigured(f"memory.configure(...) has not supplied '{name}'")
    return value


def docs() -> DocStore:
    return _require(_docs, "docs")


def vectors() -> VectorStore:
    return _require(_vectors, "vectors")


def embedder() -> Embedder:
    return _require(_embedder, "embedder")


def chat() -> Chat:
    return _require(_chat, "chat")


def log() -> Log:
    return _require(_log, "log")
