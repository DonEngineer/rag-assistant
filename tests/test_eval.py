"""Tests for the eval harness (judge and agent are faked)."""

from __future__ import annotations

from rag_assistant.eval.harness import (
    _parse_fraction,
    judge_faithfulness,
    judge_relevance,
    run_eval,
)
from rag_assistant.types import Chunk, Citation


class FakeJudge:
    def __init__(self, replies: list[str]):
        self.replies = replies
        self.calls = 0

    def invoke(self, prompt: str) -> str:
        reply = self.replies[self.calls]
        self.calls += 1
        return reply


class FakeAgent:
    """Mimics the compiled LangGraph agent's invoke(state) contract."""

    def __init__(self, answer: str, chunks: list[Chunk]):
        self.answer = answer
        self.chunks = chunks

    def invoke(self, state: dict) -> dict:
        return {
            "question": state["question"],
            "chunks": self.chunks,
            "rewrites": 0,
            "answer": self.answer,
            "citations": [
                Citation(chunk_id=c.chunk_id, source=c.source, excerpt=c.text[:50])
                for c in self.chunks
            ],
        }


def _chunk(text: str) -> Chunk:
    return Chunk(text=text, source="a.md", chunk_id="a.md#0", metadata={})


def test_parse_fraction():
    assert _parse_fraction("blah\nSCORE: 3/4") == 0.75
    assert _parse_fraction("SCORE: 0/0") == 1.0
    assert _parse_fraction("no score here") is None


def test_judge_faithfulness_uses_score():
    judge = FakeJudge(["- claim one SUPPORTED\n- claim two UNSUPPORTED\nSCORE: 1/2"])
    score, raw = judge_faithfulness("the answer", ["chunk one"], judge)
    assert score == 0.5
    assert "SCORE: 1/2" in raw


def test_judge_relevance_uses_score():
    judge = FakeJudge(["Reasonable coverage.\nSCORE: 4/5"])
    score, _ = judge_relevance("q?", ["p1"], "the answer", judge)
    assert score == 4.0


def test_run_eval_averages_scores(tmp_path):
    eval_file = tmp_path / "eval.jsonl"
    eval_file.write_text(
        '{"question": "q1?", "expected_points": ["p1"]}\n'
        '{"question": "q2?", "expected_points": ["p2"]}\n',
        encoding="utf-8",
    )
    agent = FakeAgent("The answer.", [_chunk("RELEVANT text")])
    judge = FakeJudge(
        [
            "SCORE: 2/2",  # q1 faithfulness
            "SCORE: 5/5",  # q1 relevance
            "SCORE: 1/2",  # q2 faithfulness
            "SCORE: 3/5",  # q2 relevance
        ]
    )
    summary = run_eval(eval_file, agent, judge)
    assert summary == {"faithfulness": 0.75, "relevance": 4.0, "n": 2}
    assert judge.calls == 4


def test_run_eval_unparseable_judge_scores_zero(tmp_path):
    eval_file = tmp_path / "eval.jsonl"
    eval_file.write_text('{"question": "q?", "expected_points": []}\n', encoding="utf-8")
    agent = FakeAgent("The answer.", [_chunk("text")])
    judge = FakeJudge(["I refuse to score.", "Still no score."])
    summary = run_eval(eval_file, agent, judge)
    assert summary["faithfulness"] == 0.0
    assert summary["relevance"] == 0.0
