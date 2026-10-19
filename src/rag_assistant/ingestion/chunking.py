"""Chunking: split Documents into retrieval-ready Chunks."""

from __future__ import annotations

from collections import defaultdict

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_assistant.types import Chunk, Document


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Chunk]:
    """Split documents into overlapping chunks with a recursive splitter.

    Source/page metadata is preserved on every chunk, plus ``chunk_index``.
    Chunk ids are deterministic (``f"{source}#{ordinal}"``, ordinal counted
    per source file) so re-ingestion is idempotent.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks: list[Chunk] = []
    ordinals: dict[str, int] = defaultdict(int)
    for document in documents:
        for piece in splitter.split_text(document.text):
            ordinal = ordinals[document.source]
            ordinals[document.source] += 1
            chunks.append(
                Chunk(
                    text=piece,
                    source=document.source,
                    chunk_id=f"{document.source}#{ordinal}",
                    metadata={**document.metadata, "chunk_index": ordinal},
                )
            )
    return chunks
