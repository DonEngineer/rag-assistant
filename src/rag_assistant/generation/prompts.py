"""Prompts and citation formatting for grounded answers. (Milestone M4)"""

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
    """Build [n]-style citations from the chunks used in an answer. (M4)"""
    raise NotImplementedError("M4")
