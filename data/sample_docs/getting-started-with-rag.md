# Getting Started with RAG

Sample document for demos and evals. It is intentionally small so the full
ingest → retrieve → answer loop runs in seconds on a laptop.

## What is RAG?

Retrieval-Augmented Generation (RAG) grounds a large language model in external
documents. Instead of relying only on training data, the system retrieves
relevant chunks at query time and feeds them to the model as context.

## Chunking

Documents are split into chunks before embedding. Two parameters matter:

- **Chunk size**: how many characters or tokens each chunk holds. A common
  starting point is 800 characters.
- **Chunk overlap**: how many characters repeat across adjacent chunks. Overlap
  prevents key information from being split across a boundary. Typical values
  are 10–20% of the chunk size.

## Vector stores

**Chroma** is a good default for local demos: it persists to disk, needs no
server, and is easy to reset. **pgvector** is the better choice when you
already run Postgres in production or need to scale beyond a single machine,
since it keeps vectors next to the rest of your relational data.

## Citations

Every answer should cite the chunks it used, e.g. `[1] getting-started-with-rag.md`.
Citations let a human verify each claim instead of trusting the model blindly.
