"""Vector store interface.

Backends implement this ABC so Chroma (M3) can later be swapped for pgvector
(stretch goal) without touching retrieval or generation code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_assistant.types import Chunk


class VectorStore(ABC):
    """Minimal storage contract for the assistant."""

    @abstractmethod
    def add(self, chunks: list[Chunk]) -> None:
        """Upsert chunks. Chunk ids are deterministic, so this is idempotent."""

    @abstractmethod
    def query(self, text: str, k: int = 5) -> list[Chunk]:
        """Return the k most similar chunks to the query text."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of stored chunks."""
