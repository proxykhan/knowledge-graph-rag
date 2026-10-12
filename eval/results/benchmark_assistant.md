# Benchmark (assistant set): hybrid graph+vector RAG vs. vector-only RAG

15 questions, model `qwen3:8b`, run version `router-v4/extract-v4/answer-qwen3:8b`. Accuracy = all required items mentioned (refusal for out-of-scope). Item recall = share of required items mentioned (answerable questions only).

| Question type | n | Vector RAG accuracy | Hybrid accuracy | Vector item recall | Hybrid item recall |
|---|---|---|---|---|---|
| single hop | 2 | 1/2 (50%) | 1/2 (50%) | 0.50 | 0.67 |
| two hop | 5 | 0/5 (0%) | 5/5 (100%) | 0.00 | 1.00 |
| three hop | 2 | 0/2 (0%) | 1/2 (50%) | 0.00 | 0.50 |
| out of scope | 6 | 2/6 (33%) | 2/6 (33%) | - | - |
| **all** | 15 | 3/15 (20%) | 9/15 (60%) | 0.11 | 0.81 |

| Cost and latency per query | Vector RAG | Hybrid |
|---|---|---|
| latency p50 | 6.0 s | 9.8 s |
| latency p95 | 12.9 s | 20.8 s |
| LLM tokens per query | 2,812 | 3,407 |
| cost per query (paid-tier equivalent) | $0.00000 | $0.00000 |
| answers needing regeneration | 0 | 0 |
