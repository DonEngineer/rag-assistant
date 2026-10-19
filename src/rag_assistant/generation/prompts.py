"""Prompts and citation formatting for grounded answers."""

from __future__ import annotations

from rag_assistant.types import Chunk, Citation

QA_SYSTEM_PROMPT = """You are a precise research assistant. Answer the user's question \
USING ONLY the context chunks provided below.

Rules:
- Every factual claim must be cited inline as [n], where n is the chunk number.
- If the context does not contain the answer, say "I don't have enough information \
in the provided documents." — do not guess.
- Be concise. Prefer short paragraphs or bullets over long prose.
"""

GRADE_PROMPT = """You are a relevance grader. Given the question and one retrieved chunk, \
answer with a single word: "relevant" or "irrelevant".

Question: {question}

Chunk:
{chunk}
"""

REWRITE_PROMPT = """You are a search query optimizer. The previous retrieval found no \
relevant chunks for the question below. Rewrite it as a better standalone search \
query (different keywords, no question mark needed). Reply with the query only.

Question: {question}
"""


def format_citations(chunks: list[Chunk]) -> list[Citation]:
    """Build citations for the chunks used in an answer.

    ``source`` carries a human-readable reference such as ``"guide.md, p.3"``
    or ``"guide.md § Install"``; ``excerpt`` holds the first 200 characters
    of the chunk.
    """
    citations = []
    for chunk in chunks:
        page = chunk.metadata.get("page")
        heading = chunk.metadata.get("heading")
        if page is not None:
            reference = f"{chunk.source}, p.{page}"
        elif heading:
            reference = f"{chunk.source} § {heading}"
        else:
            reference = chunk.source
        excerpt = chunk.text[:200]
        if len(chunk.text) > 200:
            excerpt += "…"
        citations.append(
            Citation(chunk_id=chunk.chunk_id, source=reference, excerpt=excerpt)
        )
    return citations
