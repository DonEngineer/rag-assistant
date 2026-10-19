"""Prompts and chat clients for grounded answers. Milestone M4."""

from rag_assistant.generation.llm import OllamaChat
from rag_assistant.generation.prompts import (
    GRADE_PROMPT,
    QA_SYSTEM_PROMPT,
    REWRITE_PROMPT,
    format_citations,
)

__all__ = [
    "GRADE_PROMPT",
    "QA_SYSTEM_PROMPT",
    "REWRITE_PROMPT",
    "OllamaChat",
    "format_citations",
]
