"""Smoke tests for shared domain types."""

from rag_assistant.types import Chunk, Citation, Document


def test_document_defaults():
    doc = Document(text="hello", source="a.md")
    assert doc.metadata == {}


def test_chunk_id_format():
    chunk = Chunk(text="t", source="a.md", chunk_id="a.md#0")
    assert chunk.chunk_id == "a.md#0"


def test_citation_fields():
    cite = Citation(chunk_id="a.md#0", source="a.md", excerpt="hello")
    assert cite.source == "a.md"
