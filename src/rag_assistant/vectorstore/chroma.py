"""Persistent Chroma backend for the VectorStore interface. (Milestone M3)"""

from __future__ import annotations

from rag_assistant.types import Chunk
from rag_assistant.vectorstore.base import VectorStore


class ChromaVectorStore(VectorStore):
    """Local, persistent Chroma collection.

    Data lives in ``persist_dir`` (see CHROMA_PERSIST_DIR) — no server needed,
    so anyone can clone the repo and run the demo.
    """

    def __init__(self, persist_dir: str = "./.chroma", collection: str = "rag") -> None:
        raise NotImplementedError("M3: implement with langchain-chroma")

    def add(self, chunks: list[Chunk]) -> None:
        raise NotImplementedError("M3")

    def query(self, text: str, k: int = 5) -> list[Chunk]:
        raise NotImplementedError("M3")

    def count(self) -> int:
        raise NotImplementedError("M3")
