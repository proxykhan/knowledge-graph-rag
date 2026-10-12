# Running and deploying

## 1. Everything on your machine with Docker

Needs Docker Desktop and a `.env` file (copy `.env.example`; set `GEMINI_API_KEY` and `SEC_USER_AGENT`).

```bash
docker compose --profile app up -d --build
```

This starts Neo4j, Postgres (pgvector) and the app. The databases start empty; build the data once:

```bash
docker compose run --rm app python -m kgrag.ingest.edgar
docker compose run --rm app python -m kgrag.ingest.chunker
docker compose run --rm app python -m kgrag.extract.extractor --workers 6
docker compose run --rm app python -m kgrag.graph.resolve
docker compose run --rm app python -m kgrag.graph.load --reset
docker compose run --rm app python -m kgrag.graph.communities
docker compose run --rm app python -m kgrag.vector.index
```

Then open http://localhost:8000. Extraction is the slow step (about 500 Gemini requests for 25 filings; on the free tier it may stop at the daily limit and resume the next day when re-run).

## 2. A public demo online

The app is one Docker container plus two databases. Any container host works; one free-tier combination:

| Piece | Service | Notes |
|---|---|---|
| Graph | Neo4j AuraDB Free | gives a `neo4j+s://...` URI, user and password |
| Vectors | Supabase or Neon (Postgres with the `vector` extension) | use the connection string with `sslmode=require` |
| App | Render, Railway or Fly.io (Docker deploy from the GitHub repo) | needs about 1 GB of memory for the embedding model |

Steps (the accounts are yours to create; keep every password and key out of the repository):

1. Create the two databases and note their connection details.
2. Load them from your machine, pointing the pipeline at the cloud databases instead of local Docker:
   ```bash
   NEO4J_URI=neo4j+s://<your-aura-host> NEO4J_USER=neo4j NEO4J_PASSWORD=<aura-password> python -m kgrag.graph.load --reset
   NEO4J_URI=neo4j+s://<your-aura-host> NEO4J_USER=neo4j NEO4J_PASSWORD=<aura-password> python -m kgrag.graph.communities
   POSTGRES_DSN="postgresql://<user>:<password>@<host>:5432/<db>?sslmode=require" python -m kgrag.vector.index
   ```
3. On the container host, deploy this repository's `Dockerfile` and set these environment variables in its dashboard (never in code):
   `GEMINI_API_KEY`, `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `POSTGRES_DSN`, and the rate limits below.
4. The app also needs `data/processed/` (entities, chunks, communities); include it in the image or attach it as a disk.

**Protect your quota.** A public page spends your Gemini requests. Set per-visitor limits:

```
ASK_LIMIT_PER_MINUTE=5
ASK_LIMIT_PER_DAY=50
```

Visitors over the limit get a polite "please wait" message (HTTP 429). Repeated questions are served from the answer cache and cost nothing.
