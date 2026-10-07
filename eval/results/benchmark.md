# Benchmark: hybrid graph+vector RAG vs. vector-only RAG

50 questions, model `gemini-3.5-flash-lite`, run version `router-v2/extract-v3`. Accuracy = all required items mentioned (refusal for out-of-scope). Item recall = share of required items mentioned (answerable questions only).

| Question type | n | Vector RAG accuracy | Hybrid accuracy | Vector item recall | Hybrid item recall |
|---|---|---|---|---|---|
| single hop | 12 | 9/12 (75%) | 9/12 (75%) | 0.75 | 0.75 |
| two hop | 12 | 1/12 (8%) | 11/12 (92%) | 0.17 | 0.92 |
| three hop | 8 | 0/8 (0%) | 4/8 (50%) | 0.00 | 0.54 |
| aggregation | 8 | 4/8 (50%) | 4/8 (50%) | 0.50 | 0.50 |
| out of scope | 10 | 10/10 (100%) | 10/10 (100%) | - | - |
| **all** | 50 | 24/50 (48%) | 38/50 (76%) | 0.38 | 0.71 |

| Cost and latency per query | Vector RAG | Hybrid |
|---|---|---|
| latency p50 | 6.0 s | 11.9 s |
| latency p95 | 12.0 s | 18.1 s |
| LLM tokens per query | 3,439 | 3,564 |
| cost per query (paid-tier equivalent) | $0.00224 | $0.00246 |
| answers needing regeneration | 0 | 1 |
