"""Chunking: split Documents into retrieval-ready Chunks. (Milestone M2)"""

from __future__ import annotations

from rag_assistant.types import Chunk, Document


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Chunk]:
    """Split documents with a recursive character splitter.

    Source/page metadata is preserved on every chunk. Chunk ids are
    deterministic (``f"{source}#{ordinal}"``) so re-ingestion is idempotent.
    """
    raise NotImplementedError("M2: implement with langchain-text-splitters")
