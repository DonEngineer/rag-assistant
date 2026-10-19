"""Run the eval harness against a question set.

Example:
    python -m rag_assistant.eval.cli evals/data/sample_eval.jsonl

Requires a local Ollama server (used both for answers and as judge).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag_assistant.api.main import services
from rag_assistant.eval import run_eval


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns a process exit code."""
    parser = argparse.ArgumentParser(prog="rag-eval", description="Run the eval harness.")
    parser.add_argument("eval_file", help="JSONL file with {question, expected_points}.")
    args = parser.parse_args(argv)
    if not Path(args.eval_file).exists():
        print(f"error: no such file: {args.eval_file}", file=sys.stderr)
        return 1
    svc = services()
    run_eval(args.eval_file, svc["agent"], svc["llm"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
