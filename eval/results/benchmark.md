# Benchmark (main set): hybrid graph+vector RAG vs. vector-only RAG

50 questions, model `gemini-3.5-flash-lite`, run version `router-v3/extract-v3`. Accuracy = all required items mentioned (refusal for out-of-scope). Item recall = share of required items mentioned (answerable questions only).

| Question type | n | Vector RAG accuracy | Hybrid accuracy | Vector item recall | Hybrid item recall |
|---|---|---|---|---|---|
| single hop | 12 | 9/12 (75%) | 9/12 (75%) | 0.75 | 0.75 |
| two hop | 12 | 1/12 (8%) | 10/12 (83%) | 0.17 | 0.89 |
| three hop | 8 | 0/8 (0%) | 8/8 (100%) | 0.00 | 1.00 |
| aggregation | 8 | 4/8 (50%) | 6/8 (75%) | 0.50 | 0.75 |
| out of scope | 10 | 10/10 (100%) | 10/10 (100%) | - | - |
| **all** | 50 | 24/50 (48%) | 43/50 (86%) | 0.38 | 0.84 |

| Cost and latency per query | Vector RAG | Hybrid |
|---|---|---|
| latency p50 | 5.9 s | 12.0 s |
| latency p95 | 8.4 s | 13.3 s |
| LLM tokens per query | 3,416 | 3,954 |
| cost per query (paid-tier equivalent) | $0.00218 | $0.00264 |
| answers needing regeneration | 0 | 0 |
