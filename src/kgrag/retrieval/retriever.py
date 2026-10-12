"""Phase 3: route a question, run the graph template and/or vector search, log the decision.

Usage:
    python -m kgrag.retrieval.retriever "Which suppliers do NVIDIA and AMD share?"

Flow:
  router -> GRAPH (confident)  : link entities -> parameterized Cypher template
         -> VECTOR (confident) : HNSW search
         -> low confidence     : both
  The graph path falls back to vectors when entities do not link, the template needs
  entities the question lacks, or the graph returns nothing. Every decision, fallback
  and result count is appended to data/logs/routing.jsonl for the Phase 5 benchmark.
"""

import json
import re
import sys
import time
from itertools import combinations
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import lru_cache

from neo4j import GraphDatabase

from kgrag import llm
from kgrag.config import DATA_DIR, PROCESSED_DIR, settings
from kgrag.retrieval import router, templates
from kgrag.retrieval.entity_linker import Link, default_linker
from kgrag.retrieval.templates import QueryType
from kgrag.vector import embed
from kgrag.vector.index import connect
from kgrag.vector.search import Passage, search

LOG_PATH = DATA_DIR / "logs" / "routing.jsonl"
VECTOR_K = 5
EF_SEARCH = 40


@dataclass
class RetrievalResult:
    question: str
    route_used: str                       # GRAPH | VECTOR | BOTH
    decision: dict | None
    links: list[dict] = field(default_factory=list)
    unlinked: list[str] = field(default_factory=list)
    graph_rows: list[dict] = field(default_factory=list)
    passages: list[Passage] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # fallbacks and why
    direction_reversed: bool = False   # facts were found only with a hop reversed; the answer step is told
    usage: llm.Usage = field(default_factory=llm.Usage)  # router tokens


@lru_cache(maxsize=1)
def _driver():
    # Notifications off: optional properties (status, years) trigger "unknown property" warnings on old graphs.
    return GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password),
                                notifications_min_severity="OFF")


def build_graph_query(decision: router.RouterDecision, links: list[Link]) -> tuple[str, dict] | None:
    """Map a decision onto a template. None when the template's required entities are missing."""
    ids = [l.entity_id for l in links]
    hops = decision.hop_params() or [(None, templates.Direction.ANY)]
    rel, direction = hops[0]
    qt = decision.query_type
    if qt == QueryType.RELATIONS_OF and ids:
        return templates.relations_of(ids[0], rel, direction)
    if qt == QueryType.SHARED_NEIGHBORS and len(ids) >= 2:
        n = min(len(ids), 3)
        return templates.shared_neighbors(ids[:n], hops[:n] if len(hops) >= n else hops[:1])
    if qt == QueryType.CHAIN and ids:
        return templates.chain(ids[0], hops)
    if qt == QueryType.PATH_BETWEEN and len(ids) >= 2:
        return templates.path_between(ids[0], ids[1])
    if qt == QueryType.COUNT:
        return templates.count(ids[0] if ids else None, rel, direction)
    if qt == QueryType.ENTITY_PROFILE and ids:
        return templates.entity_profile(ids[0])
    return None


FACTS_PER_GROUP_IN_CONTEXT = 3
GROUPS_IN_CONTEXT = 8   # 8 x 3 facts stays under the context's 25-fact cap, so no group loses its facts


def overview_rows() -> list[dict]:
    """Corpus-level questions: one row per company group (largest first), with its label and
    summary (context only, written by an LLM) and a few of its strongest facts (citable)."""
    path = PROCESSED_DIR / "communities.json"
    if not path.exists():
        return []
    groups = json.loads(path.read_text(encoding="utf-8"))[:GROUPS_IN_CONTEXT]
    return [{"group": g["label"], "summary": g["summary"], "members": [m["name"] for m in g["members"]],
             "edges": g["facts"][:FACTS_PER_GROUP_IN_CONTEXT]} for g in groups]


_FLIP = {templates.Direction.OUT: templates.Direction.IN, templates.Direction.IN: templates.Direction.OUT}


def reversed_variants(decision: router.RouterDecision) -> list[tuple[router.RouterDecision, str]]:
    """Copies of the decision with hop directions reversed, fewest reversals first.

    Routers most often misread direction in possessives ("Micron's customers" read as
    "Micron's suppliers"). When the graph returns nothing, retrying the reversed reading is
    cheap. Undirected hops (direction "any") are never touched.
    """
    flippable = [i for i, h in enumerate(decision.hops[:templates.MAX_HOPS]) if h.direction in _FLIP]
    variants = []
    for size in range(1, len(flippable) + 1):
        for combo in combinations(flippable, size):
            hops = [h.model_copy(update={"direction": _FLIP[h.direction]}) if i in combo else h
                    for i, h in enumerate(decision.hops)]
            variants.append((decision.model_copy(update={"hops": hops}),
                             "reversed hop " + ", ".join(str(i + 1) for i in combo)))
    return variants


def _vector(question: str) -> list[Passage]:
    with connect() as conn:
        return search(conn, embed.embed_query(question), k=VECTOR_K, ef_search=EF_SEARCH)


TIME_QUESTION = re.compile(
    r"\b(chang(e|ed|es|ing)|no longer|stopped|dropped|added|former(ly)?|over time|earlier report|"
    r"latest report|previous report|between (its|their|the) .{0,40}reports?)\b", re.IGNORECASE)


def normalize_time_question(question: str, decision: router.RouterDecision
                            ) -> tuple[router.RouterDecision, str | None]:
    """Change-over-time questions need one lookup, not a chain.

    Graph facts already carry the fiscal years whose reports mention them and whether they
    ended, so "how did Intel's subsidiaries change" is SUBSIDIARY_OF/in plus those labels.
    Small routers tend to turn it into a chain that repeats one relationship
    (SUBSIDIARY_OF/in -> SUBSIDIARY_OF/out), which matches nothing. Only that shape is
    rewritten; genuine chains mix relationships and are left alone.
    """
    if (decision.route == router.Route.GRAPH and decision.query_type == QueryType.CHAIN
            and len(decision.hops) > 1 and len({h.relation for h in decision.hops}) == 1
            and TIME_QUESTION.search(question)):
        fixed = decision.model_copy(update={"query_type": QueryType.RELATIONS_OF, "hops": decision.hops[:1]})
        return fixed, "change-over-time question: repeated-relation chain read as one lookup"
    return decision, None


def retrieve(question: str) -> RetrievalResult:
    start = time.perf_counter()
    usage = llm.Usage()
    try:
        decision, usage = router.route(question)
    except llm.LLMError as e:
        # QuotaExhausted is deliberately not caught: silently degrading to vector-only would
        # make a benchmark run (or a user) believe the hybrid system answered.
        decision = None
        result = RetrievalResult(question, "VECTOR", None, notes=[f"router failed ({e}); vector only"])

    time_note = None
    if decision is not None:
        decision, time_note = normalize_time_question(question, decision)

    if decision is not None:
        confident = decision.confidence >= router.CONFIDENCE_THRESHOLD
        want_graph = decision.route == router.Route.GRAPH or not confident
        want_vector = decision.route == router.Route.VECTOR or not confident
        result = RetrievalResult(question, "BOTH" if want_graph and want_vector else
                                 "GRAPH" if want_graph else "VECTOR", decision.model_dump(mode="json"))
        if time_note:
            result.notes.append(time_note)
        if not confident:
            result.notes.append(f"confidence {decision.confidence:.2f} < {router.CONFIDENCE_THRESHOLD}: running both")

        if want_graph:
            linker = default_linker()
            links = []
            for mention in decision.entities:
                link = linker.link(mention)
                if link:
                    links.append(link)
                else:
                    result.unlinked.append(mention)
            result.links = [asdict(l) for l in links]
            query = build_graph_query(decision, links)
            if decision.query_type == QueryType.OVERVIEW:
                result.graph_rows = overview_rows()
                if not result.graph_rows:
                    result.notes.append("no company groups computed yet (run kgrag.graph.communities)")
            elif query is None:
                result.notes.append(f"{decision.query_type} needs entities that did not link; vector fallback")
            else:
                result.graph_rows = templates.run(_driver(), *query)
                if not result.graph_rows:
                    for variant, how in reversed_variants(decision):
                        retry = build_graph_query(variant, links)
                        if retry and (rows := templates.run(_driver(), *retry)):
                            result.graph_rows, result.direction_reversed = rows, True
                            result.notes.append(f"no rows as read; found facts with {how}")
                            break
                if not result.graph_rows:
                    result.notes.append("graph returned no rows; vector fallback")
            if not result.graph_rows:
                want_vector = True
                result.route_used = "VECTOR" if result.route_used == "GRAPH" else result.route_used

    if decision is None or want_vector:
        result.passages = _vector(question)

    result.usage = usage
    _log(result, usage, (time.perf_counter() - start) * 1000)
    return result


def retrieve_vector_only(question: str) -> RetrievalResult:
    """The baseline: plain vector RAG, no router and no graph."""
    return RetrievalResult(question, "VECTOR", None, passages=_vector(question))


def _log(result: RetrievalResult, usage: llm.Usage, latency_ms: float) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "router_version": router.ROUTER_VERSION,
        "model": llm.model_name(),
        "question": result.question,
        "decision": result.decision,
        "route_used": result.route_used,
        "links": [{"mention": l["mention"], "entity_id": l["entity_id"], "method": l["method"]} for l in result.links],
        "unlinked": result.unlinked,
        "n_graph_rows": len(result.graph_rows),
        "n_passages": len(result.passages),
        "notes": result.notes,
        "router_tokens": {"in": usage.input_tokens, "out": usage.output_tokens},
        "latency_ms": round(latency_ms),
    }
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main() -> None:
    question = " ".join(sys.argv[1:]) or "Which suppliers do NVIDIA and AMD share?"
    r = retrieve(question)
    d = r.decision or {}
    print(f"Q: {question}")
    print(f"route: {r.route_used}  (router said {d.get('route')} {d.get('query_type')} conf={d.get('confidence')})")
    if r.links:
        print("linked: " + ", ".join(f"{l['mention']} -> {l['name']} ({l['method']})" for l in r.links))
    if r.unlinked:
        print("unlinked: " + ", ".join(r.unlinked))
    for note in r.notes:
        print(f"note: {note}")
    for row in r.graph_rows[:8]:
        extra = {k: v for k, v in row.items() if k != "edges"}
        edges = " ; ".join(f"{e['source']} -{e['relation']}-> {e['target']} [{e['chunk_ids'][0]}]"
                           for e in row["edges"][:3])
        print(f"  graph: {extra or ''} {edges}")
    for p in r.passages:
        print(f"  passage {p.score:.3f} {p.chunk_id}: {p.text[:110].replace(chr(10), ' ')}...")


if __name__ == "__main__":
    main()
