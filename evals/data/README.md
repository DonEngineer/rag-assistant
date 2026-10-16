# Eval data

Question sets for the eval harness (M6). One JSON object per line:

```json
{"question": "What is chunk overlap?", "expected_points": ["overlap repeats tokens across chunks", "typical values 10-20% of chunk size"]}
```

- `question`: the question to ask the pipeline.
- `expected_points`: key facts a good answer should contain; the relevance judge
  checks coverage of these points.

Keep sets small (10–15 questions) and tied to `data/sample_docs/` so they run
offline and stay deterministic.
