# Architecture

## Data flow

```
PDF/Markdown ──► loaders ──► chunking ──► Ollama embeddings ──► Chroma
                                                                  │
Question ──► LangGraph agent ──► retrieve(k) ──► grade ──► rewrite? ──► generate ──► answer + citations
                                                                  │
FastAPI (/ingest, /ask, /health) ◄── agent
Streamlit UI ◄── FastAPI
Eval harness ──► question set ──► pipeline ──► faithfulness / relevance scores
```

## Components

### ingestion (M2)
- `loaders.py`: `load_pdf(path)`, `load_markdown(path)` → `list[Document]`
  (text + metadata: source file, page number / heading).
- `chunking.py`: `chunk_documents(docs, size=800, overlap=120)` — recursive
  character splitting; source/page metadata preserved per chunk; deterministic
  chunk ids (`source#ordinal`) so re-ingestion is idempotent.

### embeddings (M3)
- `ollama.py`: `OllamaEmbeddings` client (default model `nomic-embed-text`),
  batch embedding with retries against `OLLAMA_BASE_URL`.

### vectorstore (M3)
- `base.py`: `VectorStore` ABC — `add(chunks)`, `query(text, k)`, `count()`.
- `chroma.py`: persistent Chroma backend (`CHROMA_PERSIST_DIR`).
- pgvector later implements the same ABC (stretch goal).

### retrieval (M4) — LangGraph agent
Nodes: `retrieve` → `grade_documents` → (`rewrite_query` → `retrieve` | `generate`).
- Grading: LLM-as-judge binary relevance per chunk; irrelevant chunks dropped.
- Rewrite: reformulate the question from the miss; retry up to `MAX_REWRITES`.
- State: `question, chunks, rewrites, answer, citations`.

### generation (M4)
- `prompts.py`: grounded-QA prompt — "answer ONLY from the context; cite [n]".
- Citation format: `[1] source, p.3` mapped from chunk metadata.

### api (M4)
- `POST /ingest` (upload documents), `POST /ask` →
  `{answer, citations, chunks_used}`, `GET /health`.

### eval (M6)
- `harness.py`: runs `evals/data/*.jsonl` (`{question, expected_points}`),
  LLM-judge scores **faithfulness** (claim grounding) and **answer relevance**
  (1–5), prints a Markdown report.

## Decision log

- 2026-10-16: LangGraph over plain LangChain — the agentic loop differentiates
  the project and matches NCP-AAI (agentic AI) certification topics.
- 2026-10-16: Chroma over pgvector for v1 — zero-setup demo; the `VectorStore`
  interface keeps pgvector as a stretch goal.
- 2026-10-16: Ollama over OpenAI as default — $0 and offline; OpenAI optional.
