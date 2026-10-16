"""Agentic retrieval loop built with LangGraph. (Milestone M4)

Graph: ``retrieve -> grade_documents -> [rewrite_query -> retrieve | generate]``

Instead of answering from whatever top-k comes back, the agent grades each
retrieved chunk for relevance. If nothing relevant survives, it rewrites the
query from the miss and retries (up to ``max_rewrites``) rather than
hallucinating an answer.
"""

from __future__ import annotations

from typing import TypedDict

from rag_assistant.types import Chunk, Citation


class AgentState(TypedDict):
    """State threaded through the retrieval graph."""

    question: str
    chunks: list[Chunk]
    rewrites: int
    answer: str
    citations: list[Citation]


def build_agent(vector_store, llm, max_rewrites: int = 2):
    """Compile the retrieval agent graph.

    Nodes: ``retrieve``, ``grade_documents``, ``rewrite_query``, ``generate``.
    """
    raise NotImplementedError("M4: implement with langgraph")


def ask(question: str, agent) -> dict:
    """Run the agent on a question.

    Returns ``{"answer": str, "citations": list[Citation], "chunks_used": int}``.
    """
    raise NotImplementedError("M4")
