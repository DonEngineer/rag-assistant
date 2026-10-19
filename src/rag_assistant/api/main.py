"""FastAPI service exposing the assistant.

Endpoints:
- ``GET /health`` — liveness probe.
- ``POST /ingest`` — upload PDF/Markdown files into the vector store.
- ``POST /ask`` — ask a question; returns answer + citations.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from rag_assistant.config import settings
from rag_assistant.embeddings import OllamaEmbeddings
from rag_assistant.generation.llm import OllamaChat
from rag_assistant.ingestion import chunk_documents, load_path
from rag_assistant.retrieval import ask, build_agent
from rag_assistant.vectorstore import ChromaVectorStore

app = FastAPI(title="RAG Assistant", version="0.1.0")


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    citations: list[str]
    chunks_used: int


class IngestResponse(BaseModel):
    documents: int
    chunks: int


_services: dict | None = None


def _build_services() -> dict:
    embeddings = OllamaEmbeddings(
        model=settings.embed_model, base_url=settings.ollama_base_url
    )
    vector_store = ChromaVectorStore(
        persist_dir=settings.chroma_persist_dir, embeddings=embeddings
    )
    llm = OllamaChat(model=settings.llm_model, base_url=settings.ollama_base_url)
    agent = build_agent(
        vector_store, llm, max_rewrites=settings.max_rewrites, top_k=settings.top_k
    )
    return {"vector_store": vector_store, "llm": llm, "agent": agent}


def services() -> dict:
    """Process-wide service singletons (vector store, LLM, agent)."""
    global _services
    if _services is None:
        _services = _build_services()
    return _services


@app.get("/health")
def health() -> dict:
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest_endpoint(
    files: list[UploadFile] = File(...),  # noqa: B008 — idiomatic FastAPI declaration
) -> IngestResponse:
    """Ingest uploaded PDF/Markdown files into the vector store."""
    documents = []
    with tempfile.TemporaryDirectory() as tmp:
        for upload in files:
            safe_name = Path(upload.filename or "upload").name
            dest = Path(tmp) / safe_name
            dest.write_bytes(upload.file.read())
            documents.extend(load_path(dest))
    chunks = chunk_documents(
        documents, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    services()["vector_store"].add(chunks)
    return IngestResponse(documents=len(documents), chunks=len(chunks))


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest) -> AskResponse:
    """Answer a question with citations to the retrieved chunks."""
    result = ask(request.question, services()["agent"])
    citations = [
        f"[{i + 1}] {citation.source}" for i, citation in enumerate(result["citations"])
    ]
    return AskResponse(
        answer=result["answer"], citations=citations, chunks_used=result["chunks_used"]
    )


def run() -> None:
    """Entry point for the ``rag-assistant`` console script."""
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
