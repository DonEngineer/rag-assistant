"""Persistent Chroma backend for the VectorStore interface."""

from __future__ import annotations

from langchain_chroma import Chroma

from rag_assistant.embeddings import OllamaEmbeddings
from rag_assistant.types import Chunk
from rag_assistant.vectorstore.base import VectorStore


class ChromaVectorStore(VectorStore):
    """Local, persistent Chroma collection.

    Data lives in ``persist_dir`` (see CHROMA_PERSIST_DIR) — no server needed,
    so anyone can clone the repo and run the demo. Chunk ids and sources are
    stored in metadata, so ``query`` returns fully-formed Chunks.
    """

    def __init__(
        self,
        persist_dir: str = "./.chroma",
        collection: str = "rag",
        embeddings: OllamaEmbeddings | None = None,
    ) -> None:
        self._embeddings = embeddings or OllamaEmbeddings()
        self._store = Chroma(
            collection_name=collection,
            embedding_function=self._embeddings,
            persist_directory=persist_dir,
        )

    def add(self, chunks: list[Chunk]) -> None:
        """Upsert chunks. Ids are deterministic, so re-adding is idempotent."""
        self._store.add_texts(
            texts=[chunk.text for chunk in chunks],
            metadatas=[
                {"chunk_id": chunk.chunk_id, "source": chunk.source, **chunk.metadata}
                for chunk in chunks
            ],
            ids=[chunk.chunk_id for chunk in chunks],
        )

    def query(self, text: str, k: int = 5) -> list[Chunk]:
        """Return the k most similar chunks to the query text."""
        results = []
        for doc in self._store.similarity_search(text, k=k):
            metadata = dict(doc.metadata)
            chunk_id = str(metadata.pop("chunk_id", ""))
            source = str(metadata.pop("source", ""))
            results.append(
                Chunk(
                    text=doc.page_content,
                    source=source,
                    chunk_id=chunk_id,
                    metadata=metadata,
                )
            )
        return results

    def count(self) -> int:
        """Return the number of stored chunks."""
        return len(self._store.get()["ids"])
