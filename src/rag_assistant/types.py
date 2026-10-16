"""Shared domain types used across ingestion, retrieval, and generation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Document:
    """A loaded source document before chunking."""

    text: str
    source: str  # file path or URI the document was loaded from
    metadata: dict = field(default_factory=dict)  # e.g. {"page": 3} or {"heading": "Intro"}


@dataclass
class Chunk:
    """A retrievable unit of text with provenance."""

    text: str
    source: str
    chunk_id: str  # deterministic: f"{source}#{ordinal}"
    metadata: dict = field(default_factory=dict)


@dataclass
class Citation:
    """A cited chunk supporting (part of) an answer."""

    chunk_id: str
    source: str
    excerpt: str
