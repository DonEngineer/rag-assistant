"""Document loaders: PDF and Markdown files -> list[Document]."""

from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

from rag_assistant.types import Document

SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown"}

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def load_pdf(path: str | Path) -> list[Document]:
    """Load a PDF, returning one Document per non-empty page.

    The 1-based page number is stored in ``metadata["page"]`` so citations
    can point at it.
    """
    reader = PdfReader(str(path))
    documents = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            documents.append(
                Document(text=text, source=str(path), metadata={"page": number})
            )
    return documents


def load_markdown(path: str | Path) -> list[Document]:
    """Load a Markdown file, returning one Document per section.

    Sections split on ATX headings (``#`` … ``######``); the heading text is
    stored in ``metadata["heading"]``. Text before the first heading becomes
    a section with an empty heading. Empty sections are dropped.
    """
    full_text = Path(path).read_text(encoding="utf-8")
    sections: list[tuple[str, list[str]]] = []
    heading = ""
    body: list[str] = []
    for line in full_text.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            if body or heading:
                sections.append((heading, body))
            heading = match.group(2).strip()
            body = []
        else:
            body.append(line)
    if body or heading:
        sections.append((heading, body))
    return [
        Document(
            text="\n".join(lines).strip(),
            source=str(path),
            metadata={"heading": text},
        )
        for text, lines in sections
        if "\n".join(lines).strip()
    ]


def load_path(path: str | Path) -> list[Document]:
    """Load a file or a directory of files.

    Directories are traversed recursively (sorted); files dispatch by
    extension. Supported: ``.pdf``, ``.md``, ``.markdown``. Anything else
    raises ``ValueError``; missing paths raise ``FileNotFoundError``.
    """
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"No such file or directory: {target}")
    if target.is_dir():
        documents: list[Document] = []
        for child in sorted(target.rglob("*")):
            if child.is_file() and child.suffix.lower() in SUPPORTED_SUFFIXES:
                documents.extend(load_path(child))
        return documents
    suffix = target.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(target)
    if suffix in {".md", ".markdown"}:
        return load_markdown(target)
    raise ValueError(f"Unsupported file type: {target}")
