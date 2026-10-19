"""Tests for the LangGraph retrieval agent (LLM and store are faked)."""

from __future__ import annotations

from rag_assistant.retrieval import ask, build_agent
from rag_assistant.retrieval.agent import NO_CONTEXT_ANSWER
from rag_assistant.types import Chunk


class FakeVectorStore:
    """Returns canned chunks; serves different results per question."""

    def __init__(self, by_question: dict[str, list[Chunk]]):
        self.by_question = by_question
        self.seen: list[str] = []

    def query(self, text: str, k: int = 5) -> list[Chunk]:
        self.seen.append(text)
        return self.by_question.get(text, [])


class FakeLLM:
    """Scripted LLM: grades by marker, rewrites once, answers canned."""

    def __init__(self, answer: str = "The final answer."):
        self.answer = answer

    def invoke(self, prompt: str) -> str:
        if "relevance grader" in prompt:
            return "relevant" if "RELEVANT" in prompt else "irrelevant"
        if "search query optimizer" in prompt:
            return "rewritten question"
        return self.answer


def _chunk(text: str, chunk_id: str = "a.md#0") -> Chunk:
    return Chunk(text=text, source="a.md", chunk_id=chunk_id, metadata={})


def test_happy_path_answers_with_citations():
    store = FakeVectorStore({"q": [_chunk("RELEVANT chunk about cats")]})
    agent = build_agent(store, FakeLLM(), max_rewrites=2, top_k=5)
    result = ask("q", agent)
    assert result["answer"] == "The final answer."
    assert result["chunks_used"] == 1
    assert len(result["citations"]) == 1
    assert result["citations"][0].chunk_id == "a.md#0"


def test_irrelevant_chunks_trigger_rewrite():
    store = FakeVectorStore(
        {
            "original": [_chunk("nothing useful here")],
            "rewritten question": [_chunk("RELEVANT chunk about dogs")],
        }
    )
    agent = build_agent(store, FakeLLM(), max_rewrites=2, top_k=5)
    result = ask("original", agent)
    assert result["answer"] == "The final answer."
    assert store.seen == ["original", "rewritten question"]
    assert result["chunks_used"] == 1


def test_exhausted_rewrites_answers_without_context():
    store = FakeVectorStore({"q": [_chunk("nothing useful")]})
    agent = build_agent(store, FakeLLM(), max_rewrites=1, top_k=5)
    result = ask("q", agent)
    assert result["answer"] == NO_CONTEXT_ANSWER
    assert result["citations"] == []
    assert result["chunks_used"] == 0
    assert store.seen == ["q", "rewritten question"]
