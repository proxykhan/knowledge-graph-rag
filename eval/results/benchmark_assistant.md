# Benchmark (assistant set): hybrid graph+vector RAG vs. vector-only RAG

15 questions, model `qwen3:8b`, run version `router-v4/extract-v4/answer-qwen3:8b`. Accuracy = all required items mentioned (refusal for out-of-scope). Item recall = share of required items mentioned (answerable questions only).

| Question type | n | Vector RAG accuracy | Hybrid accuracy | Vector item recall | Hybrid item recall |
|---|---|---|---|---|---|

| Cost and latency per query | Vector RAG | Hybrid |
|---|---|---|
