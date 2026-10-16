"""FastAPI service exposing the assistant. (Milestone M4)

Endpoints:
- ``GET /health`` — liveness probe.
- ``POST /ingest`` — upload documents into the vector store.
- ``POST /ask`` — ask a question; returns answer + citations.
"""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from rag_assistant.config import settings

app = FastAPI(title="RAG Assistant", version="0.1.0")


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    citations: list[str]
    chunks_used: int


@app.get("/health")
def health() -> dict:
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/ingest")
def ingest() -> dict:
    """Ingest uploaded documents. (M4)"""
    raise NotImplementedError("M4")


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest) -> AskResponse:
    """Answer a question with citations. (M4)"""
    raise NotImplementedError("M4")


def run() -> None:
    """Entry point for the ``rag-assistant`` console script."""
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
