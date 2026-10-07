# Knowledge Graph RAG for Enterprise Data

Hybrid knowledge-graph + vector RAG over SEC 10-K filings from 20 semiconductor companies.

**Stack:** Python · Neo4j · pgvector · LangChain (text splitting) · Gemini API (free tier; Claude API switchable) · sentence-transformers · FastAPI

Answers multi-hop questions ("which equipment suppliers of NVIDIA's competitors…") that plain vector RAG misses.

## Results

> Filled in at Phase 5: accuracy by hop count vs. a vector-only baseline, latency p95, cost per query, one-time ingestion cost.

| Question type | Vector RAG | Graph + Vector RAG |
|---|---|---|
| Single hop | – | – |
| Two hop | – | – |
| Three hop | – | – |
| Aggregation | – | – |
| Out of scope (correct refusal) | – | – |

## Architecture

```
INGESTION
 10-K HTML ─► parse sections ─► chunk (chunk_id) ─┬─► LLM extraction     ─► entity resolution ─► Neo4j
                                                  └─► sentence-transformers ─────────────────► pgvector
QUERY
 question ─► router ─┬─ GRAPH  ─► resolve entities ─► Cypher template ─┐
                     ├─ VECTOR ─► HNSW search ──────────────────────────┼─► merge ─► LLM ─► citation check
                     └─ unsure ─► both ─────────────────────────────────┘
```

## Build progress

- [x] Phase 0: project setup, ontology, EDGAR download, section parsing, chunking
- [x] Phase 1: LLM extraction → entity resolution → Neo4j (MERGE, chunk_id on every edge). 5 filings, 404 chunks → 760 grounded relationships (62 rejected) → 432 entities, 457 distinct facts
- [x] Phase 2: pgvector index (HNSW), recall@k on a labeled set. Recall@5 0.83, recall@10 0.90 on 30 labeled questions (see below)
- [x] Phase 3: router + parameterized Cypher templates. 24/24 routing decisions correct on a small self-written set (see caveat below)
- [ ] Phase 4: merged context + validated citations, FastAPI endpoint
- [ ] Phase 5: benchmark vs. vector-only baseline

## Retrieval quality (Phase 2)

30 questions written in different words from the filings, each labeled with phrases quoted verbatim from one company's 10-K (`eval/retrieval_questions.jsonl`); every chunk containing a phrase counts as relevant. Embeddings: `BAAI/bge-small-en-v1.5`, local.

| Chunks embedded as | recall@1 | recall@5 | recall@10 | MRR@10 |
|---|---|---|---|---|
| raw chunk text | 0.30 | 0.67 | 0.77 | 0.47 |
| **company + section header + text** | **0.43** | **0.83** | **0.90** | **0.61** |

The header matters because 10-K text says "we": a question about NVIDIA otherwise has nothing to match. Of the 5 questions still missed at k=5, two are supplier questions ("whose processes does AMD rely on?"), the case the graph route is built for.

HNSW (`m=16`, `ef_construction=64`, `ef_search=40`) returned the exact top 10 for every question at every `ef_search` from 10 to 100. At 404 chunks the index is effectively exact, so `ef_search` is not yet a meaningful tuning knob here.

## Routing (Phase 3)

One light LLM call per question (Gemini 3.5 Flash-Lite, minimal thinking) returns a strict schema: route (`GRAPH`/`VECTOR`), confidence, a query type from the template library, entity mentions, and relation/direction enums. Confidence below 0.7 runs both paths. The graph path links mentions to node ids (alias → word overlap → embeddings) and runs one of six parameterized Cypher templates: `RELATIONS_OF`, `SHARED_NEIGHBORS`, `TWO_HOP`, `PATH_BETWEEN`, `COUNT`, `ENTITY_PROFILE`. It falls back to vectors when entities do not link or the graph returns nothing. Every decision is logged to `data/logs/routing.jsonl`.

The model never writes Cypher: entity ids and relation names reach Neo4j only as query parameters, relation and direction are enums validated before use, and a test passes a Cypher-injection string through every template to check it never appears in the query text.

| Routing eval (24 questions) | |
|---|---|
| route accuracy | 24/24 |
| route + query type accuracy | 24/24 |
| graph questions that returned rows | 15/15 |
| out-of-scope question | confidence 0.10 → both paths (refusal is Phase 4's job) |

**Caveat:** I wrote both the router's few-shot examples and these questions, and several are close in wording, so this overstates real-world accuracy. The Phase 5 benchmark uses separately written questions.

## Setup

1. Start Docker Desktop, then the databases:
   ```bash
   docker compose up -d
   ```
2. Python environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   pip install -e .
   ```
3. Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY` and `SEC_USER_AGENT`.
4. Download filings and chunk them:
   ```bash
   python -m kgrag.ingest.edgar NVDA AMD INTC AMAT MU
   python -m kgrag.ingest.chunker
   ```
5. Extract, resolve, load the graph, and index vectors:
   ```bash
   python -m kgrag.extract.extractor
   python -m kgrag.graph.resolve
   python -m kgrag.graph.load --reset
   python -m kgrag.vector.index
   python -m kgrag.eval.retrieval
   ```
6. Run tests: `pytest`

## Design decisions

- **Ontology is fixed in code** (`src/kgrag/ontology.py`): 7 entity types, 11 relationship types, with allowed source→target type pairs. Extraction output outside it is rejected.
- **`SUPPLIES` only, no `CUSTOMER_OF`**: one direction per fact, so every supply-chain query has one shape.
- **Financial tables are dropped** before chunking: they carry numbers, not relationships, and would roughly double extraction cost.
- **Sections Items 1, 1A, 7 only**: Business, Risk Factors and MD&A hold nearly all named customers, suppliers, competitors and acquisitions.
- **Pluggable LLM provider** (`src/kgrag/llm.py`): Gemini free tier by default, Claude with one setting (`LLM_PROVIDER=claude`). The free tier's limits are unpublished and per project, so requests are paced, per-minute 429s slow the pacer, and a per-day 429 stops the run cleanly; the per-chunk cache makes the next run resume where it stopped.
- **Grounding check on every extracted fact**: each relationship must carry a quote that appears verbatim in its chunk, or it is rejected. The chunk id is attached by code, never by the model.
- **Company + section header on every embedded chunk**: +16 points recall@5 (measured above).
- **Graph and vector store share one key, `chunk_id`**: every Neo4j edge lists the chunk ids that justify it, and every pgvector row lists the entity ids its chunk mentions.
- **One router call does routing and parameter extraction**, halving free-tier requests per question compared with separate calls.
- **Prompt version is part of the cache key**, so changing the prompt can never silently mix old and new extractions.

## What did not work

- **Newest Gemini models were unusable on the free tier.** `gemini-3.8-flash` and `gemini-3.7-flash` returned 503 "high demand" (and one 504 after 60s) even for a one-word prompt; `gemini-3.5-flash` answered in ~1.5s, so the pipeline uses it.
- **Prompt v1 copied the SEC's raw company title** ("APPLIED MATERIALS INC /DE") as the entity name and wrote long descriptive names ("export controls enacted by the United States government"). v2 passes clean legal names and asks for short standard names for non-company entities.
- **Gemini 3.5 Flash's free tier allows only 20 requests/day.** Found from the 429 error body after 12 chunks. Switched to Gemini 3.5 Flash-Lite and 4 chunks per request (~101 requests for 404 chunks). On an 8-chunk comparison Flash-Lite matched Flash on company-to-company relationships but skipped the filer's own market/technology/regulation links in 3 of 8 chunks, so those relationship types are under-covered.
- **Embedding similarity merged different products.** "Ryzen AI Max+ 388" / "392", "RDNA 3.5" / "RDNA 4", and via a union-find chain "AMD EPYC" into "Ryzen Embedded 9000". Fixed by requiring identical numbers for an embedding merge and stricter Product/Technology thresholds.
- **Two-letter acronyms are ambiguous.** "EC" merged into "export controls" (it can also mean European Commission). Acronym matching now needs 3+ letters.
- **"U.S. export controls" and "export controls" remain separate nodes**, on purpose: the bare phrase sometimes refers to other countries' controls.
- **10-Ks rarely name equipment customers.** Applied Materials' filing does not say it supplies TSMC or Intel, so supply chains through equipment makers are missing from the graph.
- **The first retrieval labels were incomplete.** 3 of 30 questions were answered by chunks the label missed (the ZT Systems acquisition is stated in both Risk Factors and MD&A). Inspecting the top results of every miss found them; labels now allow several verbatim phrases. Recall@5 moved from 0.73 to 0.83 by fixing labels, not the system.
- **Dead row versions degraded HNSW results.** Re-embedding updates every row; before `VACUUM`, `ef_search=10` returned only 82% of the exact top 10. The indexer now vacuums after every load.
- **Entity linking exposed duplicate nodes.** "Samsung" and "Samsung Electronics Co., Ltd." were separate entities, splitting Samsung's facts (also ASML, IMS, OpenAI, THATIC, KLA/KLA-Tencor). Added a rule: a one-word company name joins the single longer company name starting with that word, and does nothing when it is ambiguous ("Applied" → Applied Materials or Applied Ventures). "Micron" also linked to a Product node the extractor had mislabelled; the linker now prefers companies.
- **Network timeouts crashed the router eval.** The Gemini SDK raises `httpx` transport errors, not API errors, so the retry logic missed them. They are now retried like 503s.
- **Generic 10-K section parsing failed on Intel.** Intel's 10-K has no "Item 1A" headings in the body, only a page-number cross-reference index, so the parser found 0 sections. Fixed with per-filer start/end line markers (`SECTION_MARKERS` in `companies.py`). Those markers are tied to one filing year and need re-checking for each new 10-K.
