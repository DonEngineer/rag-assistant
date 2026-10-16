"""Local embedding client backed by Ollama. (Milestone M3)"""

from __future__ import annotations


class OllamaEmbeddings:
    """Batch embedding client against a local Ollama server.

    Default model ``nomic-embed-text``. No API keys, no network beyond localhost.
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
    ) -> None:
        raise NotImplementedError("M3: implement with langchain-ollama or httpx")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, returning one vector per text."""
        raise NotImplementedError("M3")
