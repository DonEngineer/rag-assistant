# RAG Assistant — AI handoff context

> Read this file first when continuing work on this project in a new session
> or with a different model (Claude, Gemini, ChatGPT).

## What this is

Portfolio flagship: an agentic document Q&A (RAG) assistant. Goal: a public,
recruiter-readable repo demonstrating AI Engineer skills — ingestion, embeddings,
vector search, agentic orchestration (LangGraph), API design, and evals.

## Stack (approved 2026-10-16)

- Orchestration: **LangGraph** (agentic loop: retrieve → grade → rewrite → generate)
- Vector DB: **Chroma** (local, persistent); `vectorstore/base.py` interface keeps
  pgvector swappable later
- Embeddings + LLM: **Ollama** local (`nomic-embed-text`, `llama3.1`); OpenAI optional
  via `OPENAI_API_KEY`, never required
- API: FastAPI · UI: Streamlit · Evals: custom lightweight harness
- Python 3.11+, `pip install -e ".[dev]"`, ruff + pytest in CI

## Hard rules

1. **100% of repo content in ENGLISH** — code, comments, docstrings, README,
   commit messages, `.ai/` files. Never let Spanish slip in.
2. Keep dependencies light; justify every new one in the commit message.
3. No paid APIs in the default path. Flag any paid-API need before using it.
4. Every answer the system gives must include citations to retrieved chunks.

## Milestones

M1 scaffold ✅ → M2 ingestion ✅ → M3 embeddings+Chroma ✅ → M4 agent+API ✅ →
M5 UI ✅ → M6 evals ✅ → M7 polish. See README roadmap for details.

## Conventions

- `src/rag_assistant/<area>/` — one concern per package; `base.py` holds interfaces.
- Config via env vars with sane local defaults (`config.py`); `.env.example` documents them.
- Commits: imperative mood, English, e.g. `Add PDF loader with page-aware chunking`.
- Tests live in `tests/`; CI runs ruff + pytest on every push.
