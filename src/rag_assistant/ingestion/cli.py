"""Command-line ingestion: load documents, chunk them, report stats.

Examples:
    python -m rag_assistant.ingestion.cli ingest ./data/sample_docs
    python -m rag_assistant.ingestion.cli ingest ./docs --output chunks.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag_assistant.config import settings
from rag_assistant.ingestion import chunk_documents, load_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-ingest",
        description="Ingest documents into chunks.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest", help="Load and chunk a file or directory.")
    ingest.add_argument("path", help="File or directory to ingest.")
    ingest.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    ingest.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap)
    ingest.add_argument(
        "--output",
        default=None,
        help="Optional JSONL file to write the chunks to.",
    )
    return parser


def _cmd_ingest(args: argparse.Namespace) -> int:
    try:
        documents = load_path(args.path)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    chunks = chunk_documents(
        documents,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    per_source: dict[str, int] = {}
    for chunk in chunks:
        per_source[chunk.source] = per_source.get(chunk.source, 0) + 1
    print(f"documents: {len(documents)}")
    print(f"chunks: {len(chunks)}")
    for source in sorted(per_source):
        print(f"  {source}: {per_source[source]} chunks")
    if args.output:
        out = Path(args.output)
        with out.open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(
                    json.dumps(
                        {
                            "chunk_id": chunk.chunk_id,
                            "source": chunk.source,
                            "text": chunk.text,
                            "metadata": chunk.metadata,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns a process exit code."""
    args = _build_parser().parse_args(argv)
    if args.command == "ingest":
        return _cmd_ingest(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
