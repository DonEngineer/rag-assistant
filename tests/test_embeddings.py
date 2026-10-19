"""Tests for the Ollama embeddings wrapper (no server needed)."""

from unittest.mock import MagicMock

from rag_assistant.embeddings import OllamaEmbeddings


def test_init_stores_config():
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text", base_url="http://localhost:11434"
    )
    assert embeddings.model == "nomic-embed-text"
    assert embeddings.base_url == "http://localhost:11434"


def test_embed_delegates_to_client():
    embeddings = OllamaEmbeddings()
    embeddings._client = MagicMock()
    embeddings._client.embed_documents.return_value = [[2.0] * 3, [4.0] * 3]
    assert embeddings.embed(["ab", "abcd"]) == [[2.0] * 3, [4.0] * 3]
    embeddings._client.embed_documents.assert_called_once_with(["ab", "abcd"])


def test_embed_query_delegates_to_client():
    embeddings = OllamaEmbeddings()
    embeddings._client = MagicMock()
    embeddings._client.embed_query.return_value = [1.0, 2.0]
    assert embeddings.embed_query("hello") == [1.0, 2.0]
    embeddings._client.embed_query.assert_called_once_with("hello")
