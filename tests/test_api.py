"""Tests for the FastAPI service (agent and store are faked)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import rag_assistant.api.main as api_main
from rag_assistant.types import Chunk, Citation


class FakeVectorStore:
    def __init__(self):
        self.added: list[Chunk] = []

    def add(self, chunks: list[Chunk]) -> None:
        self.added.extend(chunks)


@pytest.fixture
def client_and_store(monkeypatch):
    store = FakeVectorStore()

    def fake_ask(question: str, agent) -> dict:
        return {
            "answer": "Canned answer.",
            "citations": [
                Citation(chunk_id="a.md#0", source="a.md, p.1", excerpt="...")
            ],
            "chunks_used": 1,
        }

    monkeypatch.setattr(api_main, "ask", fake_ask)
    monkeypatch.setattr(
        api_main, "services", lambda: {"vector_store": store, "agent": object()}
    )
    return TestClient(api_main.app), store


def test_health(client_and_store):
    client, _ = client_and_store
    assert client.get("/health").json() == {"status": "ok"}


def test_ask_returns_answer_and_citations(client_and_store):
    client, _ = client_and_store
    resp = client.post("/ask", json={"question": "What?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Canned answer."
    assert body["citations"] == ["[1] a.md, p.1"]
    assert body["chunks_used"] == 1


def test_ingest_uploads_documents(client_and_store):
    client, store = client_and_store
    resp = client.post(
        "/ingest",
        files={"files": ("doc.md", "# Title\n\nhello world\n", "text/markdown")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["documents"] == 1
    assert body["chunks"] >= 1
    assert len(store.added) == body["chunks"]
