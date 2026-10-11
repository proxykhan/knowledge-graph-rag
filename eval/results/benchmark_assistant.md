# Benchmark (assistant set): hybrid graph+vector RAG vs. vector-only RAG

15 questions, model `gemini-3.5-flash-lite`, run version `router-v3/extract-v3`. Accuracy = all required items mentioned (refusal for out-of-scope). Item recall = share of required items mentioned (answerable questions only).

| Question type | n | Vector RAG accuracy | Hybrid accuracy | Vector item recall | Hybrid item recall |
|---|---|---|---|---|---|
| single hop | 2 | 1/2 (50%) | 1/2 (50%) | 0.50 | 0.50 |
| two hop | 5 | 1/5 (20%) | 3/5 (60%) | 0.20 | 0.60 |
| three hop | 2 | 0/2 (0%) | 1/2 (50%) | 0.00 | 0.50 |
| out of scope | 6 | 5/6 (83%) | 6/6 (100%) | - | - |
| **all** | 15 | 7/15 (47%) | 11/15 (73%) | 0.22 | 0.56 |

| Cost and latency per query | Vector RAG | Hybrid |
|---|---|---|
| latency p50 | 5.9 s | 11.8 s |
| latency p95 | 8.5 s | 13.0 s |
| LLM tokens per query | 3,466 | 3,814 |
| cost per query (paid-tier equivalent) | $0.00246 | $0.00284 |
| answers needing regeneration | 0 | 0 |
