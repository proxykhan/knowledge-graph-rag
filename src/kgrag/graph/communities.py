"""Find groups of closely connected companies (community detection) and label them.

Usage:
    python -m kgrag.graph.communities            # after kgrag.graph.load

Company-to-company links that are still current (supplies, competes with, partners with,
acquired, owned by) form a graph; Louvain modularity (networkx, fixed seed) splits it into
groups that are densely linked inside and sparsely linked to each other. One light LLM call
per group writes a short label and summary from the group's own facts; the facts, not the
summary, are what answers cite. Writes data/processed/communities.json and sets
`community` on each company node.
"""

import json
from collections import Counter

import networkx as nx
from neo4j import GraphDatabase
from pydantic import BaseModel, Field

from kgrag import llm
from kgrag.answer.context import statement
from kgrag.config import PROCESSED_DIR, settings

COMPANY_LINKS = ("SUPPLIES", "COMPETES_WITH", "PARTNERS_WITH", "ACQUIRED", "SUBSIDIARY_OF")
MIN_SIZE = 3
MAX_GROUPS = 12
FACTS_PER_GROUP = 25
OUT = PROCESSED_DIR / "communities.json"


class GroupLabel(BaseModel):
    label: str = Field(description="3-6 word name for the group, e.g. 'AI accelerator competitors'")
    summary: str = Field(description="Two sentences: what ties these companies together, using only the facts given")


SYSTEM = """You label groups of companies found in SEC 10-K filings.
Given the members of one group and the facts linking them, write a short label and a two-sentence
summary of what ties them together. Use only the facts given; do not add outside knowledge."""


def fetch_edges(driver) -> list[dict]:
    query = """
        MATCH (a:Company)-[r]->(b:Company)
        WHERE type(r) IN $rels AND coalesce(r.status, 'current') <> 'past'
        RETURN a.id AS a, a.name AS a_name, b.id AS b, b.name AS b_name, type(r) AS relation,
               r.chunk_ids AS chunk_ids, r.evidence AS evidence, coalesce(r.years, []) AS years,
               coalesce(r.status, 'current') AS status, r.confidence AS confidence"""
    with driver.session() as s:
        return [rec.data() for rec in s.run(query, rels=list(COMPANY_LINKS))]


def detect(edges: list[dict]) -> list[set[str]]:
    g = nx.Graph()
    for e in edges:
        # Competition and supply both mean "in the same business"; weight them equally.
        g.add_edge(e["a"], e["b"], weight=g.get_edge_data(e["a"], e["b"], {}).get("weight", 0) + 1)
    groups = nx.community.louvain_communities(g, weight="weight", seed=42)
    return sorted((c for c in groups if len(c) >= MIN_SIZE), key=len, reverse=True)[:MAX_GROUPS]


def main() -> None:
    driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password),
                                  notifications_min_severity="OFF")
    with driver:
        edges = fetch_edges(driver)
        groups = detect(edges)
        names = {e["a"]: e["a_name"] for e in edges} | {e["b"]: e["b_name"] for e in edges}
        result = []
        for i, members in enumerate(groups):
            inside = [e for e in edges if e["a"] in members and e["b"] in members]
            inside.sort(key=lambda e: -(e["confidence"] or 0))
            facts = [{"source": names[e["a"]], "relation": e["relation"], "target": names[e["b"]],
                      "chunk_ids": e["chunk_ids"], "evidence": e["evidence"], "years": e["years"],
                      "status": e["status"]} for e in inside[:FACTS_PER_GROUP]]
            degree = Counter(x for e in inside for x in (e["a"], e["b"]))
            ordered = sorted(members, key=lambda m: -degree[m])
            prompt = ("Members: " + ", ".join(names[m] for m in ordered) + "\n\nFacts:\n"
                      + "\n".join("- " + statement(f) for f in facts))
            try:
                label, _ = llm.generate_structured(SYSTEM, prompt, GroupLabel, light=True)
                label_text, summary = label.label, label.summary
            except (llm.LLMError, llm.QuotaExhausted) as e:
                label_text, summary = f"Group {i + 1}", f"(label unavailable: {e})"
            result.append({"id": i, "label": label_text, "summary": summary,
                           "members": [{"id": m, "name": names[m]} for m in ordered], "facts": facts})
            print(f"[{i}] {label_text} ({len(members)} companies): {', '.join(names[m] for m in ordered[:6])}…")

        with driver.session() as s:
            s.run("MATCH (c:Company) REMOVE c.community")
            s.run("""UNWIND $rows AS row MATCH (c:Entity {id: row.id}) SET c.community = row.group""",
                  rows=[{"id": m["id"], "group": g["id"]} for g in result for m in g["members"]])
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(result)} groups from {len(edges)} current company-to-company links -> {OUT}")


if __name__ == "__main__":
    main()
