"""Tests for the Chroma vector store backend (fully local)."""

from __future__ import annotations

from rag_assistant.types import Chunk
from rag_assistant.vectorstore import ChromaVectorStore

_KEYWORDS = ("cat", "dog", "mat", "night")


class FakeEmbeddings:
    """Deterministic keyword-count embeddings; no server needed."""

    def _vector(self, text: str) -> list[float]:
        words = text.lower().split()
        return [float(words.count(keyword)) for keyword in _KEYWORDS]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def _chunks() -> list[Chunk]:
    return [
        Chunk(
            text="the cat sat on the mat",
            source="a.md",
            chunk_id="a.md#0",
            metadata={"page": 1},
        ),
        Chunk(
            text="dogs bark loudly at night",
            source="a.md",
            chunk_id="a.md#1",
            metadata={"page": 2},
        ),
    ]


def test_add_query_and_count(tmp_path):
    store = ChromaVectorStore(
        persist_dir=str(tmp_path / "chroma"), embeddings=FakeEmbeddings()
    )
    assert store.count() == 0
    store.add(_chunks())
    assert store.count() == 2

    results = store.query("cat", k=1)
    assert len(results) == 1
    assert results[0].chunk_id == "a.md#0"
    assert results[0].source == "a.md"
    assert results[0].metadata["page"] == 1
    assert "cat" in results[0].text


def test_readd_is_idempotent(tmp_path):
    store = ChromaVectorStore(
        persist_dir=str(tmp_path / "chroma"), embeddings=FakeEmbeddings()
    )
    store.add(_chunks())
    store.add(_chunks())
    assert store.count() == 2


def test_persists_across_instances(tmp_path):
    persist_dir = str(tmp_path / "chroma")
    store = ChromaVectorStore(persist_dir=persist_dir, embeddings=FakeEmbeddings())
    store.add(_chunks())

    reopened = ChromaVectorStore(persist_dir=persist_dir, embeddings=FakeEmbeddings())
    assert reopened.count() == 2
    assert reopened.query("night", k=1)[0].chunk_id == "a.md#1"
