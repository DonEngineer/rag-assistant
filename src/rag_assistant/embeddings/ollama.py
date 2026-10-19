"""Local embedding client backed by Ollama."""

from __future__ import annotations

from langchain_ollama import OllamaEmbeddings as _LangChainOllamaEmbeddings


class OllamaEmbeddings:
    """Batch embedding client against a local Ollama server.

    Default model ``nomic-embed-text``. No API keys, no network beyond
    localhost. Also exposes the LangChain ``embed_documents``/``embed_query``
    interface so it plugs straight into LangChain vector stores.
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self.base_url = base_url
        self._client = _LangChainOllamaEmbeddings(model=model, base_url=base_url)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, returning one vector per text."""
        return self.embed_documents(texts)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents (LangChain interface)."""
        return self._client.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string (LangChain interface)."""
        return self._client.embed_query(text)
