# Benchmark (blind set): hybrid graph+vector RAG vs. vector-only RAG

15 questions, model `gemini-3.5-flash-lite`, run version `router-v3/extract-v3`. Accuracy = all required items mentioned (refusal for out-of-scope). Item recall = share of required items mentioned (answerable questions only).

| Question type | n | Vector RAG accuracy | Hybrid accuracy | Vector item recall | Hybrid item recall |
|---|---|---|---|---|---|
| single hop | 1 | 1/1 (100%) | 1/1 (100%) | 1.00 | 1.00 |
| two facts | 9 | 1/9 (11%) | 0/9 (0%) | 0.11 | 0.06 |
| three facts | 5 | 0/5 (0%) | 0/5 (0%) | 0.00 | 0.00 |
| **all** | 15 | 2/15 (13%) | 1/15 (7%) | 0.13 | 0.10 |

| Cost and latency per query | Vector RAG | Hybrid |
|---|---|---|
| latency p50 | 7.3 s | 12.4 s |
| latency p95 | 9.3 s | 15.6 s |
| LLM tokens per query | 3,611 | 4,900 |
| cost per query (paid-tier equivalent) | $0.00257 | $0.00290 |
| answers needing regeneration | 0 | 0 |
