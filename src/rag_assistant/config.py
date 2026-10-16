"""Runtime configuration: environment variables with sane local defaults.

Copy `.env.example` to `.env` to customize. Every setting has a default so the
project runs out of the box with a local Ollama server.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    """All tunables for the assistant."""

    ollama_base_url: str = field(
        default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    llm_model: str = field(default_factory=lambda: _env("OLLAMA_LLM_MODEL", "llama3.1"))
    embed_model: str = field(
        default_factory=lambda: _env("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    )
    chroma_persist_dir: str = field(
        default_factory=lambda: _env("CHROMA_PERSIST_DIR", "./.chroma")
    )
    api_host: str = field(default_factory=lambda: _env("API_HOST", "127.0.0.1"))
    api_port: int = field(default_factory=lambda: int(_env("API_PORT", "8000")))
    chunk_size: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "800")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "120")))
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "5")))
    max_rewrites: int = field(default_factory=lambda: int(_env("MAX_REWRITES", "2")))


settings = Settings()
