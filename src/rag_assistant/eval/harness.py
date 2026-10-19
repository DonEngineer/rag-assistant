"""Lightweight eval harness: faithfulness + answer relevance.

Runs a question set (JSONL lines of ``{"question", "expected_points"}``)
through the retrieval agent and scores each answer with an LLM judge:

- **faithfulness** — fraction of the answer's claims supported by the
  retrieved chunks (0..1). An abstention ("I don't have enough information")
  makes no claims and scores 1.0.
- **answer relevance** — how well the answer addresses the question and
  covers the expected points (1..5).

Deliberately dependency-light: no ragas, no external services. The judge is
just another ``invoke(prompt) -> str`` LLM, so the whole loop stays local.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from rag_assistant.retrieval import ask

FAITHFULNESS_PROMPT = """You are a faithfulness judge. Given an answer and the \
context chunks it was generated from, decide whether every factual claim in the \
answer is supported by the chunks.

Answer:
{answer}

Chunks:
{chunks}

Instructions:
- List each factual claim in the answer on its own line.
- Mark each claim SUPPORTED or UNSUPPORTED using only the chunks.
- If the answer states it does not have enough information, there are no claims.
- End with exactly one line: SCORE: <supported>/<total> (e.g. SCORE: 3/4)
"""

RELEVANCE_PROMPT = """You are an answer relevance judge. Score how well the answer \
addresses the question (1-5). A 5 fully answers it and covers every expected \
point; a 1 ignores the question.

Question: {question}

Expected key points:
{points}

Answer:
{answer}

End with exactly one line: SCORE: <n>/5 (e.g. SCORE: 4/5)
"""

_SCORE_RE = re.compile(r"SCORE:\s*(\d+)\s*/\s*(\d+)", re.IGNORECASE)


@dataclass
class QuestionResult:
    question: str
    answer: str
    faithfulness: float
    relevance: float


def _parse_fraction(text: str) -> float | None:
    """Parse 'SCORE: supported/total' into a 0..1 fraction.

    Returns 1.0 for 'SCORE: 0/0' (no claims made, e.g. an abstention) and
    None when the judge did not emit a parseable score.
    """
    match = _SCORE_RE.search(text)
    if not match:
        return None
    supported, total = int(match.group(1)), int(match.group(2))
    if total == 0:
        return 1.0
    return supported / total


def judge_faithfulness(answer: str, chunk_texts: list[str], judge) -> tuple[float, str]:
    """Score claim grounding. Returns (score 0..1, raw judge output)."""
    context = "\n\n".join(f"[{i + 1}] {text}" for i, text in enumerate(chunk_texts))
    raw = judge.invoke(FAITHFULNESS_PROMPT.format(answer=answer, chunks=context))
    score = _parse_fraction(raw)
    return (score if score is not None else 0.0, raw)


def judge_relevance(
    question: str, expected_points: list[str], answer: str, judge
) -> tuple[float, str]:
    """Score answer relevance 1..5. Returns (score, raw judge output)."""
    points = "\n".join(f"- {point}" for point in expected_points) or "(none given)"
    raw = judge.invoke(RELEVANCE_PROMPT.format(question=question, points=points, answer=answer))
    match = _SCORE_RE.search(raw)
    score = float(match.group(1)) if match else 0.0
    return (score, raw)


def run_eval(eval_file: str | Path, agent, judge) -> dict:
    """Run one eval file, print a Markdown report, and return summary scores.

    ``agent`` is the compiled retrieval agent; ``judge`` is an
    ``invoke(prompt) -> str`` LLM. Returns
    ``{"faithfulness": float, "relevance": float, "n": int}`` with corpus
    averages.
    """
    entries = [
        json.loads(line)
        for line in Path(eval_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    results: list[QuestionResult] = []
    for entry in entries:
        question = entry["question"]
        outcome = ask(question, agent)
        chunks = outcome.get("chunks", [])
        faithfulness, _ = judge_faithfulness(
            outcome["answer"], [chunk.text for chunk in chunks], judge
        )
        relevance, _ = judge_relevance(
            question, entry.get("expected_points", []), outcome["answer"], judge
        )
        results.append(
            QuestionResult(
                question=question,
                answer=outcome["answer"],
                faithfulness=faithfulness,
                relevance=relevance,
            )
        )

    avg_faith = sum(r.faithfulness for r in results) / len(results) if results else 0.0
    avg_rel = sum(r.relevance for r in results) / len(results) if results else 0.0

    lines = [f"# Eval report — {Path(eval_file).name}", ""]
    lines.append("| question | faithfulness | relevance (1-5) |")
    lines.append("|---|---|---|")
    for result in results:
        short = result.question
        if len(short) > 60:
            short = short[:57] + "..."
        lines.append(f"| {short} | {result.faithfulness:.2f} | {result.relevance:.1f} |")
    lines += [
        "",
        (
            f"**Averages ({len(results)} questions):** "
            f"faithfulness {avg_faith:.2f}, relevance {avg_rel:.2f}"
        ),
    ]
    print("\n".join(lines))
    return {"faithfulness": avg_faith, "relevance": avg_rel, "n": len(results)}
