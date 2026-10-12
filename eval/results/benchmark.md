# Benchmark (main set): hybrid graph+vector RAG vs. vector-only RAG

50 questions, model `qwen3:8b`, run version `router-v4/extract-v4/answer-qwen3:8b`. Accuracy = all required items mentioned (refusal for out-of-scope). Item recall = share of required items mentioned (answerable questions only).

| Question type | n | Vector RAG accuracy | Hybrid accuracy | Vector item recall | Hybrid item recall |
|---|---|---|---|---|---|
| single hop | 12 | 9/12 (75%) | 10/12 (83%) | 0.75 | 0.83 |
| two hop | 12 | 1/12 (8%) | 7/12 (58%) | 0.11 | 0.64 |
| three hop | 8 | 0/8 (0%) | 6/8 (75%) | 0.00 | 0.79 |
| aggregation | 8 | 7/8 (88%) | 5/8 (62%) | 0.88 | 0.69 |
| out of scope | 6 | 5/6 (83%) | 4/5 (80%) | - | - |
| **all** | 46 | 22/46 (48%) | 32/45 (71%) | 0.43 | 0.74 |

| Cost and latency per query | Vector RAG | Hybrid |
|---|---|---|
| latency p50 | 5.6 s | 8.8 s |
| latency p95 | 16.0 s | 16.9 s |
| LLM tokens per query | 3,032 | 3,671 |
| cost per query (paid-tier equivalent) | $0.00000 | $0.00000 |
| answers needing regeneration | 3 | 1 |
