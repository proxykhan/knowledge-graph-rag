"""Turn retrieval results into one labeled, de-duplicated context with citable sources.

Graph rows become plain-English statements (raw triples make awkward answers), each with
its evidence quote. Every statement and passage gets a short source id ("S3") that maps to
the chunk ids it came from, so the answer's citations can be checked against what was
actually retrieved.
"""

from dataclasses import dataclass, field

from kgrag.retrieval.retriever import RetrievalResult

MAX_GRAPH_FACTS = 25
MAX_PASSAGES = 5

_SENTENCES = {
    "SUPPLIES": "{s} supplies {t}.",
    "COMPETES_WITH": "{s} and {t} compete.",
    "PARTNERS_WITH": "{s} partners with {t}.",
    "SUBSIDIARY_OF": "{s} is a subsidiary of {t}.",
    "ACQUIRED": "{s} acquired {t}.",
    "EXECUTIVE_OF": "{s} is an executive of {t}.",
    "PRODUCES": "{s} produces {t}.",
    "USES_TECHNOLOGY": "{s} uses {t}.",
    "SERVES_MARKET": "{s} serves the {t} market.",
    "OPERATES_IN": "{s} operates in {t}.",
    "SUBJECT_TO": "{s} is subject to {t}.",
}


@dataclass
class Source:
    id: str                       # "S1"
    kind: str                     # "graph" | "passage"
    text: str                     # statement or passage text shown to the model
    chunk_ids: list[str]          # where it came from; what a citation resolves to
    evidence: list[str] = field(default_factory=list)  # verbatim quotes behind a graph fact


@dataclass
class Context:
    sources: list[Source]
    summaries: list[str]          # aggregate lines (counts), each pointing at source ids

    def by_id(self) -> dict[str, Source]:
        return {s.id: s for s in self.sources}

    def render(self) -> str:
        graph = [s for s in self.sources if s.kind == "graph"]
        passages = [s for s in self.sources if s.kind == "passage"]
        parts = []
        if graph or self.summaries:
            lines = [f"[{s.id}] {s.text} Evidence: \"{s.evidence[0]}\"" for s in graph]
            parts.append("## Graph facts (extracted from the filings)\n" + "\n".join(self.summaries + lines))
        if passages:
            parts.append("## Retrieved passages (verbatim filing text)\n" + "\n\n".join(
                f"[{s.id}] ({s.chunk_ids[0]})\n{s.text}" for s in passages))
        return "\n\n".join(parts) if parts else "(no context retrieved)"


def statement(edge: dict) -> str:
    template = _SENTENCES.get(edge["relation"], "{s} " + edge["relation"].lower().replace("_", " ") + " {t}.")
    return template.format(s=edge["source"], t=edge["target"])


def build_context(result: RetrievalResult) -> Context:
    sources: list[Source] = []
    edge_ids: dict[tuple[str, str, str], str] = {}

    def add_edge(edge: dict) -> str:
        key = (edge["source"], edge["relation"], edge["target"])
        if key not in edge_ids and len(edge_ids) < MAX_GRAPH_FACTS:
            sid = f"S{len(sources) + 1}"
            edge_ids[key] = sid
            sources.append(Source(sid, "graph", statement(edge), list(edge["chunk_ids"]), list(edge["evidence"])))
        return edge_ids.get(key, "")

    summaries = []
    for row in result.graph_rows:
        ids = [sid for sid in (add_edge(e) for e in row["edges"]) if sid]
        if "count" in row:  # COUNT template: an aggregate over the edges listed with it
            summaries.append(f"[graph aggregate] {row['entity']} has {row['count']} such relationships "
                             f"in the graph (see {', '.join(ids[:10])}).")

    seen_chunks = set()
    for p in result.passages[:MAX_PASSAGES]:
        if p.chunk_id in seen_chunks:
            continue
        seen_chunks.add(p.chunk_id)
        sources.append(Source(f"S{len(sources) + 1}", "passage", p.text, [p.chunk_id]))
    return Context(sources, summaries)
