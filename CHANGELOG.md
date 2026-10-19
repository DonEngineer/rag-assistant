# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.0] — 2026-10-19

First release: a complete agentic document Q&A assistant.

### Added

- M1: project scaffold, README, `.ai/` handoff docs, CI (ruff + pytest).
- M2: ingestion pipeline — PDF (per-page) and Markdown (per-section) loaders,
  deterministic chunking (800/120), CLI.
- M3: Ollama embeddings (`nomic-embed-text`) and persistent Chroma vector store
  behind a swappable `VectorStore` interface.
- M4: LangGraph retrieval agent (retrieve → grade → rewrite → generate) with
  cited answers, plus the FastAPI service (`/health`, `/ingest`, `/ask`).
- M5: Streamlit demo UI wired to the API (upload, ask, citations).
- M6: lightweight eval harness — faithfulness and answer relevance scored by a
  local LLM judge, with CLI.
- M7: demo walkthrough, changelog, final docs polish.
