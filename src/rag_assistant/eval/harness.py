"""Lightweight eval harness. (Milestone M6)

Runs a question set from ``evals/data/*.jsonl`` through the pipeline and scores:

- **faithfulness** — is every claim in the answer grounded in retrieved chunks?
- **answer relevance** — does the answer actually address the question? (1–5)

Judged by the same local LLM, so the whole loop stays free and offline.
Deliberately dependency-light: no ragas, no external services.
"""

from __future__ import annotations

from pathlib import Path


def run_eval(eval_file: str | Path, agent) -> dict:
    """Run one eval file; return ``{"faithfulness": float, "relevance": float}``.

    Prints a Markdown report with per-question scores.
    """
    raise NotImplementedError("M6")
