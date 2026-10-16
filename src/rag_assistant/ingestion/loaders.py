"""Document loaders: PDF and Markdown files -> list[Document]. (Milestone M2)"""

from __future__ import annotations

from pathlib import Path

from rag_assistant.types import Document


def load_pdf(path: str | Path) -> list[Document]:
    """Load a PDF, returning one Document per page.

    Page numbers go into ``metadata["page"]`` so citations can point at them.
    """
    raise NotImplementedError("M2: implement with pypdf")


def load_markdown(path: str | Path) -> list[Document]:
    """Load a Markdown file, returning one Document per top-level section.

    Section headings go into ``metadata["heading"]``.
    """
    raise NotImplementedError("M2: implement with section-aware splitting")


def load_path(path: str | Path) -> list[Document]:
    """Dispatch to the right loader based on file extension (.pdf, .md)."""
    raise NotImplementedError("M2")
