"""Tests for ingestion: loaders, chunking, and path dispatch."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_assistant.ingestion import chunk_documents, load_markdown, load_path
from rag_assistant.ingestion.loaders import load_pdf
from rag_assistant.types import Document


def _write_minimal_pdf(path: Path, page_texts: list[str]) -> None:
    """Write a minimal single-font PDF whose text pypdf can extract."""
    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}

    def emit(num: int, body: bytes) -> None:
        offsets[num] = len(out)
        out.extend(f"{num} 0 obj\n".encode("latin-1"))
        out.extend(body)
        out.extend(b"\nendobj\n")

    page_nums = [3 + 2 * i for i in range(len(page_texts))]
    content_nums = [4 + 2 * i for i in range(len(page_texts))]
    font_num = 3 + 2 * len(page_texts)

    emit(1, b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{n} 0 R" for n in page_nums)
    emit(2, f"<< /Type /Pages /Kids [{kids}] /Count {len(page_texts)} >>".encode("latin-1"))
    for page_num, content_num, text in zip(page_nums, content_nums, page_texts):
        emit(
            page_num,
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {content_num} 0 R "
            f"/Resources << /Font << /F1 {font_num} 0 R >> >> >>".encode("latin-1"),
        )
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({safe}) Tj ET".encode("latin-1")
        emit(
            content_num,
            b"<< /Length " + str(len(stream)).encode("latin-1") + b" >>\nstream\n" + stream + b"\nendstream",
        )
    emit(font_num, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    xref_at = len(out)
    out.extend(f"xref\n0 {font_num + 1}\n".encode("latin-1"))
    out.extend(b"0000000000 65535 f \n")
    for n in range(1, font_num + 1):
        out.extend(f"{offsets[n]:010d} 00000 n \n".encode("latin-1"))
    out.extend(
        f"trailer\n<< /Size {font_num + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_at}\n%%EOF".encode("latin-1")
    )
    path.write_bytes(bytes(out))


def test_load_pdf_one_document_per_page(tmp_path: Path):
    pdf = tmp_path / "doc.pdf"
    _write_minimal_pdf(pdf, ["First page text", "Second page text"])
    docs = load_pdf(pdf)
    assert len(docs) == 2
    assert "First page text" in docs[0].text
    assert docs[0].metadata["page"] == 1
    assert docs[1].metadata["page"] == 2


def test_load_pdf_skips_empty_pages(tmp_path: Path):
    pdf = tmp_path / "doc.pdf"
    _write_minimal_pdf(pdf, ["Some text", "   "])
    docs = load_pdf(pdf)
    assert len(docs) == 1


def test_load_markdown_splits_on_headings(tmp_path: Path):
    md = tmp_path / "doc.md"
    md.write_text("# Alpha\n\ntext one\n\n## Beta\n\ntext two\n", encoding="utf-8")
    docs = load_markdown(md)
    assert [d.metadata["heading"] for d in docs] == ["Alpha", "Beta"]
    assert "text one" in docs[0].text


def test_load_markdown_without_headings_is_one_document(tmp_path: Path):
    md = tmp_path / "doc.md"
    md.write_text("just some text\n", encoding="utf-8")
    docs = load_markdown(md)
    assert len(docs) == 1
    assert docs[0].metadata["heading"] == ""


def test_load_path_dispatch_and_directory(tmp_path: Path):
    (tmp_path / "a.md").write_text("# T\n\nhello\n", encoding="utf-8")
    _write_minimal_pdf(tmp_path / "b.pdf", ["pdf text"])
    (tmp_path / "ignore.txt").write_text("nope", encoding="utf-8")
    docs = load_path(tmp_path)
    assert {d.source for d in docs} == {str(tmp_path / "a.md"), str(tmp_path / "b.pdf")}


def test_load_path_rejects_unknown_suffix(tmp_path: Path):
    target = tmp_path / "data.csv"
    target.write_text("a,b\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_path(target)


def test_load_path_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_path(tmp_path / "nope.md")


def test_chunk_documents_ids_metadata_and_overlap():
    text = " ".join(f"token{i}" for i in range(300))
    docs = [Document(text=text, source="s.md", metadata={"heading": "H"})]
    chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1
    assert [c.chunk_id for c in chunks] == [f"s.md#{i}" for i in range(len(chunks))]
    assert all(c.metadata["heading"] == "H" for c in chunks)
    assert all(c.metadata["chunk_index"] == i for i, c in enumerate(chunks))
    # overlap: trailing tokens of one chunk reappear at the start of the next
    first_tail = chunks[0].text.split()[-3:]
    assert all(token in chunks[1].text for token in first_tail)


def test_chunk_documents_deterministic_ids():
    docs = [Document(text="hello world " * 300, source="s.md")]
    first = [c.chunk_id for c in chunk_documents(docs)]
    second = [c.chunk_id for c in chunk_documents(docs)]
    assert first == second
