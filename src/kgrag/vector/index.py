"""Phase 2: embed every chunk into pgvector, keyed by the same chunk_id as the graph.

Usage:
    python -m kgrag.vector.index                # contextual header (company + section) prepended
    python -m kgrag.vector.index --no-header    # raw chunk text only, for comparison

Each row stores the metadata the build guide asks for (document id, section path, date)
plus the entity ids the chunk mentions, so a retrieved passage can step into its graph
neighbourhood, and a graph path can step back to its text.
"""

import argparse
import json

import psycopg
from pgvector.psycopg import register_vector

from kgrag.config import PROCESSED_DIR, settings
from kgrag.ingest.companies import COMPANY_NAMES
from kgrag.vector import embed

# HNSW build parameters (pgvector defaults). m: links per node; ef_construction: build-time
# search width. Query-time width (ef_search) is set per query in search.py.
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 64


def passage_text(chunk: dict, header: bool) -> str:
    """Chunks say "we" and "our". The header names the company so a question about
    NVIDIA can match a chunk whose text never says "NVIDIA"."""
    if not header:
        return chunk["text"]
    company = COMPANY_NAMES.get(chunk["ticker"], chunk["company"])
    return f"{company} {chunk['section_path']}\n{chunk['text']}"


def connect() -> psycopg.Connection:
    conn = psycopg.connect(settings.postgres_dsn, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-header", action="store_true", help="embed raw chunk text only")
    args = parser.parse_args()
    header = not args.no_header

    with (PROCESSED_DIR / "chunks.jsonl").open(encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    entity_map_path = PROCESSED_DIR / "chunk_entities.json"
    chunk_entities = json.loads(entity_map_path.read_text(encoding="utf-8")) if entity_map_path.exists() else {}

    print(f"Embedding {len(chunks)} chunks with {settings.embedding_model} (header={header})...")
    vectors = embed.embed_passages([passage_text(c, header) for c in chunks])

    with connect() as conn:
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id     text PRIMARY KEY,
                doc_id       text NOT NULL,
                ticker       text NOT NULL,
                section_path text NOT NULL,
                report_date  date NOT NULL,
                entity_ids   text[] NOT NULL DEFAULT '{{}}',
                text         text NOT NULL,
                embedding    vector({embed.dim()}) NOT NULL,
                embedded_with_header boolean NOT NULL
            )""")
        with conn.cursor() as cur:
            cur.executemany("""
                INSERT INTO chunks (chunk_id, doc_id, ticker, section_path, report_date, entity_ids,
                                    text, embedding, embedded_with_header)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    entity_ids = EXCLUDED.entity_ids, text = EXCLUDED.text,
                    embedding = EXCLUDED.embedding, embedded_with_header = EXCLUDED.embedded_with_header""",
                [(c["chunk_id"], c["doc_id"], c["ticker"], c["section_path"], c["report_date"],
                  chunk_entities.get(c["chunk_id"], []), c["text"], v, header)
                 for c, v in zip(chunks, vectors)])
        conn.execute(f"""
            CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION})""")
        conn.execute("CREATE INDEX IF NOT EXISTS chunks_ticker ON chunks (ticker)")
        # Re-embedding updates every row; until vacuumed, the dead row versions stay in the
        # HNSW graph and crowd out live results at low ef_search (measured: ANN overlap@10
        # 0.82 before vacuum vs 1.00 after, at ef_search=10).
        conn.execute("VACUUM ANALYZE chunks")
        n = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    print(f"pgvector: {n} chunks indexed (HNSW m={HNSW_M}, ef_construction={HNSW_EF_CONSTRUCTION})")


if __name__ == "__main__":
    main()
