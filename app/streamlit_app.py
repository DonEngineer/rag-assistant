"""Minimal Streamlit demo UI for the assistant.

Talks to the FastAPI backend: upload documents, ask questions, inspect
citations. The API must be running first::

    uvicorn rag_assistant.api.main:app --reload
    streamlit run app/streamlit_app.py

Set RAG_API_URL to point at a non-default backend.
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

DEFAULT_API_URL = os.environ.get("RAG_API_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT = 180.0

st.set_page_config(page_title="RAG Assistant", layout="wide")
st.title("RAG Assistant")
st.caption("Agentic document Q&A — LangGraph + Chroma + Ollama (all local)")


def _api_ok(api_url: str) -> bool:
    try:
        response = httpx.get(f"{api_url}/health", timeout=5.0)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


with st.sidebar:
    st.header("Backend")
    api_url = st.text_input("API URL", value=DEFAULT_API_URL)
    if _api_ok(api_url):
        st.success("API reachable")
    else:
        st.error("API not reachable — start it with uvicorn first")
    st.divider()
    st.header("Ingest documents")
    uploads = st.file_uploader(
        "PDF or Markdown files",
        type=["pdf", "md", "markdown"],
        accept_multiple_files=True,
    )
    if st.button("Ingest", disabled=not uploads):
        with st.spinner("Ingesting documents..."):
            try:
                response = httpx.post(
                    f"{api_url}/ingest",
                    files=[("files", (u.name, u.getvalue(), u.type)) for u in uploads],
                    timeout=REQUEST_TIMEOUT,
                )
            except httpx.HTTPError as exc:
                st.error(f"Ingest request failed: {exc}")
            else:
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"Ingested {data['documents']} documents → {data['chunks']} chunks")
                else:
                    st.error(f"Ingest failed ({response.status_code})")

st.header("Ask a question")
question = st.text_input("Question", placeholder="What is chunk overlap?")
if st.button("Ask", disabled=not question.strip()):
    with st.spinner("Thinking..."):
        try:
            response = httpx.post(
                f"{api_url}/ask", json={"question": question}, timeout=REQUEST_TIMEOUT
            )
        except httpx.HTTPError as exc:
            st.error(f"Ask request failed: {exc}")
        else:
            if response.status_code != 200:
                st.error(f"Ask failed ({response.status_code})")
            else:
                data = response.json()
                st.markdown(data["answer"])
                with st.expander(f"Citations — {data['chunks_used']} chunks used"):
                    for citation in data["citations"]:
                        st.markdown(f"- {citation}")
