# RAG Assistant

[![CI](https://github.com/DonEngineer/rag-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/DonEngineer/rag-assistant/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

An **agentic document Q&A assistant**: ingest PDFs and Markdown, chunk and embed them into a local vector store, then answer questions with an LLM — with citations to the exact chunks used. A small, dependency-light eval harness measures **faithfulness** and **answer relevance**, so quality is verifiable, not vibes.

Built as a portfolio project on the road to AI Engineer roles: every component is small, readable, and easy to demo.

## Why agentic RAG?

A naive RAG pipeline (embed → top-k → answer) breaks when retrieval misses. This project uses a **LangGraph agent** that closes the loop:

```
            ┌──────────────┐
            │   Question   │
            └──────┬───────┘
                   ▼
            ┌──────────────┐
            │   Retrieve   │──► Chroma (top-k chunks)
            └──────┬───────┘
                   ▼
            ┌──────────────┐     no
            │Grade documents│────────► Rewrite query ──┐
            └──────┬───────┘                            │
                   │ yes                                │
                   ▼                                    │
            ┌──────────────┐                            │
            │Generate answer│◄──────────────────────────┘
            │ + citations  │
            └──────────────┘
```

If the retrieved chunks don't support an answer, the agent rewrites the query and retries instead of hallucinating.

## Features

- 📄 Ingestion of PDF and Markdown documents with configurable chunking
- 🧠 Local embeddings + LLM via [Ollama](https://ollama.com) — **$0, fully offline**
- 🗄️ [Chroma](https://www.trychroma.com/) vector store (persistent, local); storage interface ready for pgvector
- 🤖 LangGraph retrieval agent with document grading and query rewriting
- 💬 Cited answers — every factual claim points back to source chunks
- 🌐 FastAPI API layer + minimal Streamlit demo UI
- 📊 Lightweight eval harness: faithfulness & answer relevance (LLM-as-judge)

## Quickstart

**Prerequisites:** Python 3.11+, [Ollama](https://ollama.com) running locally.

```bash
# 1. Pull the models (one-time)
ollama pull llama3.1
ollama pull nomic-embed-text

# 2. Install
pip install -e ".[dev]"

# 3. Ingest the sample docs  (M2)
python -m rag_assistant.ingestion.cli ingest ./data/sample_docs

# 4. Ask via the API  (M4)
uvicorn rag_assistant.api.main:app --reload
# POST /ask {"question": "What is hybrid chunking?"}

# 5. Or use the demo UI  (M5)
streamlit run app/streamlit_app.py
```

Copy `.env.example` to `.env` to override models, ports, or the Chroma persist directory. Set `OPENAI_API_KEY` to optionally compare against OpenAI models — it is never required.

## Project structure

```
src/rag_assistant/
├── ingestion/     PDF/Markdown loading + chunking          (M2)
├── embeddings/    Ollama embedding client                  (M3)
├── vectorstore/   Storage interface + Chroma backend       (M3)
├── retrieval/     LangGraph agent: retrieve→grade→rewrite  (M4)
├── generation/    Prompts + citation formatting             (M4)
├── api/           FastAPI service                          (M4)
└── eval/          Faithfulness/relevance harness           (M6)
app/               Streamlit demo UI                        (M5)
evals/data/        Eval question/answer sets                (M6)
data/sample_docs/ Sample documents for demo and evals
.ai/               Handoff docs for AI-assisted development
```

## Roadmap

| Milestone | Status |
|---|---|
| M1 Project scaffold, README, handoff docs | ✅ Done |
| M2 Ingestion pipeline (PDF/Markdown → chunks) | ✅ Done |
| M3 Embeddings + Chroma vector store | ✅ Done |
| M4 Agentic retrieval + cited answers + FastAPI | ✅ Done |
| M5 Streamlit demo UI | ✅ Done |
| M6 Eval harness (faithfulness, answer relevance) | ✅ Done |
| M7 Polish: demo, docs, release | ⬜ Next |

Stretch goals: pgvector backend, hybrid search (BM25 + dense), reranking, one-command Docker Compose demo.

## Evals

`src/rag_assistant/eval/` runs a question set from `evals/data/` against the pipeline and reports:

- **Faithfulness** — is every claim in the answer grounded in the retrieved chunks?
- **Answer relevance** — does the answer actually address the question?

Scored by the same local LLM, so the whole loop stays free and offline.

Run it with:

```bash
python -m rag_assistant.eval.cli evals/data/sample_eval.jsonl
```

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Orchestration | LangGraph | Agentic retrieval loop, not just a chain |
| Vector DB | Chroma | Zero-setup, local, demo-friendly |
| Embeddings / LLM | Ollama | Free, local, reproducible |
| API | FastAPI | Standard, typed, auto-docs |
| UI | Streamlit | Fastest credible demo |
| Evals | Custom harness | Light deps, transparent metrics |

## License

MIT — see [LICENSE](LICENSE).
