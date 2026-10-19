"""Agentic retrieval loop built with LangGraph.

Graph: ``retrieve -> grade_documents -> [rewrite_query -> retrieve | generate]``

Instead of answering from whatever top-k comes back, the agent grades each
retrieved chunk for relevance. If nothing relevant survives, it rewrites the
query from the miss and retries (up to ``max_rewrites``) rather than
hallucinating an answer.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from rag_assistant.generation.prompts import (
    GRADE_PROMPT,
    QA_SYSTEM_PROMPT,
    REWRITE_PROMPT,
    format_citations,
)
from rag_assistant.types import Chunk, Citation

NO_CONTEXT_ANSWER = "I don't have enough information in the provided documents."


class AgentState(TypedDict):
    """State threaded through the retrieval graph."""

    question: str
    chunks: list[Chunk]
    rewrites: int
    answer: str
    citations: list[Citation]


def build_agent(vector_store, llm, max_rewrites: int = 2, top_k: int = 5):
    """Compile the retrieval agent graph.

    ``vector_store`` must implement ``query(text, k) -> list[Chunk]``;
    ``llm`` must implement ``invoke(prompt) -> str``.
    """

    def retrieve(state: AgentState) -> dict:
        return {"chunks": vector_store.query(state["question"], k=top_k)}

    def grade_documents(state: AgentState) -> dict:
        kept = []
        for chunk in state["chunks"]:
            verdict = (
                llm.invoke(
                    GRADE_PROMPT.format(question=state["question"], chunk=chunk.text)
                )
                .strip()
                .lower()
            )
            if "relevant" in verdict and "irrelevant" not in verdict:
                kept.append(chunk)
        return {"chunks": kept}

    def rewrite_query(state: AgentState) -> dict:
        new_question = llm.invoke(REWRITE_PROMPT.format(question=state["question"])).strip()
        return {"question": new_question, "rewrites": state["rewrites"] + 1}

    def generate(state: AgentState) -> dict:
        if not state["chunks"]:
            return {"answer": NO_CONTEXT_ANSWER, "citations": []}
        context = "\n\n".join(
            f"[{i + 1}] {chunk.text}\n(source: {chunk.source})"
            for i, chunk in enumerate(state["chunks"])
        )
        prompt = (
            f"{QA_SYSTEM_PROMPT}\n\nContext:\n{context}\n\n"
            f"Question: {state['question']}\nAnswer:"
        )
        return {
            "answer": llm.invoke(prompt).strip(),
            "citations": format_citations(state["chunks"]),
        }

    def decide(state: AgentState) -> str:
        if state["chunks"]:
            return "generate"
        if state["rewrites"] < max_rewrites:
            return "rewrite"
        return "generate"

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("generate", generate)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents", decide, {"generate": "generate", "rewrite": "rewrite_query"}
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", END)
    return graph.compile()


def ask(question: str, agent) -> dict:
    """Run the agent on a question.

    Returns ``{"answer": str, "citations": list[Citation], "chunks_used": int}``.
    """
    final = agent.invoke(
        {"question": question, "chunks": [], "rewrites": 0, "answer": "", "citations": []}
    )
    return {
        "answer": final["answer"],
        "citations": final["citations"],
        "chunks_used": len(final["chunks"]),
    }
