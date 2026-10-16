"""Document ingestion: loading (PDF/Markdown) and chunking. Milestone M2."""

from rag_assistant.ingestion.chunking import chunk_documents
from rag_assistant.ingestion.loaders import load_markdown, load_path, load_pdf

__all__ = ["chunk_documents", "load_markdown", "load_path", "load_pdf"]
