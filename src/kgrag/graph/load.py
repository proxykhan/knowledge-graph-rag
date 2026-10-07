"""Load resolved entities and validated relationships into Neo4j.

Usage:
    python -m kgrag.graph.load            # MERGE everything (safe to re-run)
    python -m kgrag.graph.load --reset    # wipe this project's graph first (after re-resolving entities)

Graph shape:
    (:Entity:Company {id, name, type, aliases, mentions})  -- one label per ontology type
    (:Chunk {id, doc_id, ticker, section, report_date})-[:MENTIONS]->(:Entity)
    (:Entity)-[:SUPPLIES {chunk_ids, evidence, confidence}]->(:Entity)   -- one edge per fact

The same fact found in several chunks becomes one edge whose chunk_ids lists all of them,
so every edge can be cited back to the sentences that justify it. Also writes
data/processed/chunk_entities.json (chunk id -> entity ids) for the pgvector index.
"""

import argparse
import json
from collections import defaultdict

from neo4j import GraphDatabase

from kgrag.config import PROCESSED_DIR, settings
from kgrag.graph.extractions import load_extractions
from kgrag.ontology import EntityType, RelationType

BATCH = 500


def build_rows() -> tuple[list[dict], list[dict], list[dict], list[dict], dict[str, list[str]]]:
    """Returns entities, chunk rows, mention rows, fact rows, and chunk id -> entity ids."""
    entities = json.loads((PROCESSED_DIR / "entities.json").read_text(encoding="utf-8"))
    alias_to_id = {(e["type"], a): e["id"] for e in entities for a in e["aliases"]}

    chunk_meta = {}
    with (PROCESSED_DIR / "chunks.jsonl").open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunk_meta[c["chunk_id"]] = {"id": c["chunk_id"], "doc_id": c["doc_id"], "ticker": c["ticker"],
                                         "section": c["section_path"], "report_date": c["report_date"]}

    mentions: dict[str, set[str]] = defaultdict(set)
    facts: dict[tuple[str, str, str], dict] = {}
    unresolved = 0
    for record in load_extractions():
        cid = record["chunk_id"]
        for e in record["entities"]:
            if (eid := alias_to_id.get((e["type"], e["name"]))):
                mentions[cid].add(eid)
        for r in record["relationships"]:
            s = alias_to_id.get((r["source_type"], r["source"]))
            t = alias_to_id.get((r["target_type"], r["target"]))
            if not s or not t:
                unresolved += 1
                continue
            if s == t:  # two aliases of one entity, e.g. a subsidiary resolved into its parent
                continue
            fact = facts.setdefault((s, r["relation"], t), {"s": s, "t": t, "rel": r["relation"],
                                                            "chunk_ids": [], "evidence": [], "confidence": 0.0})
            if cid not in fact["chunk_ids"]:
                fact["chunk_ids"].append(cid)
                fact["evidence"].append(r["evidence"])
            fact["confidence"] = max(fact["confidence"], r["confidence"])
    if unresolved:
        print(f"[warn] {unresolved} relationships reference names missing from entities.json; "
              "re-run kgrag.graph.resolve after extraction")

    chunks = [chunk_meta[cid] for cid in sorted(mentions) if cid in chunk_meta]
    mention_rows = [{"chunk": cid, "entity": eid} for cid, eids in mentions.items() for eid in sorted(eids)]
    return entities, chunks, mention_rows, list(facts.values()), {c: sorted(e) for c, e in mentions.items()}


def _batched(rows: list[dict]):
    for i in range(0, len(rows), BATCH):
        yield rows[i:i + BATCH]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="delete all Entity and Chunk nodes first")
    args = parser.parse_args()

    entities, chunks, mention_rows, facts, chunk_entities = build_rows()
    (PROCESSED_DIR / "chunk_entities.json").write_text(json.dumps(chunk_entities, indent=1), encoding="utf-8")

    driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
    with driver, driver.session() as session:
        if args.reset:
            session.run("MATCH (n) WHERE n:Entity OR n:Chunk DETACH DELETE n")
        session.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
        session.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
        session.run("CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)")

        # Labels and relationship types cannot be Cypher parameters. They come from the
        # ontology enums only, never from model output, so interpolating them is safe.
        for etype in EntityType:
            rows = [e for e in entities if e["type"] == etype.value]
            for batch in _batched(rows):
                session.run(f"""
                    UNWIND $rows AS row
                    MERGE (e:Entity {{id: row.id}})
                    SET e:`{etype.value}`, e.name = row.name, e.type = row.type,
                        e.aliases = row.aliases, e.mentions = row.mentions""", rows=batch)

        for batch in _batched(chunks):
            session.run("""
                UNWIND $rows AS row
                MERGE (c:Chunk {id: row.id})
                SET c.doc_id = row.doc_id, c.ticker = row.ticker, c.section = row.section,
                    c.report_date = row.report_date""", rows=batch)
        for batch in _batched(mention_rows):
            session.run("""
                UNWIND $rows AS row
                MATCH (c:Chunk {id: row.chunk}), (e:Entity {id: row.entity})
                MERGE (c)-[:MENTIONS]->(e)""", rows=batch)

        for rel in RelationType:
            rows = [f for f in facts if f["rel"] == rel.value]
            for batch in _batched(rows):
                session.run(f"""
                    UNWIND $rows AS row
                    MATCH (s:Entity {{id: row.s}}), (t:Entity {{id: row.t}})
                    MERGE (s)-[r:`{rel.value}`]->(t)
                    SET r.chunk_ids = row.chunk_ids, r.evidence = row.evidence,
                        r.confidence = row.confidence""", rows=batch)

        counts = session.run("""
            MATCH (e:Entity) WITH count(e) AS entities
            MATCH (c:Chunk) WITH entities, count(c) AS chunks
            OPTIONAL MATCH (:Entity)-[r]->(:Entity)
            RETURN entities, chunks, count(r) AS facts""").single()
    print(f"Neo4j: {counts['entities']} entities, {counts['chunks']} chunks, {counts['facts']} facts "
          f"(from {len(facts)} distinct facts in the cache)")


if __name__ == "__main__":
    main()
