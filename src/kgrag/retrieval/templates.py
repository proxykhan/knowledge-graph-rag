"""Parameterized Cypher template library, keyed by query type.

The router (an LLM) never writes Cypher. It picks a QueryType and fills parameters, which are
validated against enums before use. Values (entity ids, relation names) reach Neo4j only as
query parameters; the only text spliced into a query is an arrow from the fixed _ARROWS table
below, chosen by a Direction enum. This is a security control (no Cypher injection through
the question) and makes every answer reproducible from the logged parameters.

Every template returns rows of the form {"edges": [edge, ...], ...} where an edge is
{source, relation, target, chunk_ids, evidence, confidence}, so Phase 4 can turn any graph
result into cited statements the same way.
"""

from enum import StrEnum

from neo4j import Driver

from kgrag.ontology import RelationType


class QueryType(StrEnum):
    RELATIONS_OF = "RELATIONS_OF"            # who supplies X / what does X produce
    SHARED_NEIGHBORS = "SHARED_NEIGHBORS"    # suppliers shared by X and Y
    CHAIN = "CHAIN"                          # 1-3 hops: suppliers of X's competitors
    PATH_BETWEEN = "PATH_BETWEEN"            # how are X and Y connected
    COUNT = "COUNT"                          # how many suppliers does X have / who has the most
    ENTITY_PROFILE = "ENTITY_PROFILE"        # everything the graph knows about X
    NONE = "NONE"                            # not a graph question


class Direction(StrEnum):
    OUT = "out"   # named entity -> other   ("what does NVIDIA produce": PRODUCES out)
    IN = "in"     # other -> named entity   ("who supplies NVIDIA": SUPPLIES in)
    ANY = "any"


_ARROWS = {Direction.OUT: ("-[{r}]->"), Direction.IN: ("<-[{r}]-"), Direction.ANY: ("-[{r}]-")}

# Shared projection: a path's relationships as citable edges.
_EDGES = """[r IN relationships(p) | {source: startNode(r).name, relation: type(r), target: endNode(r).name,
             chunk_ids: r.chunk_ids, evidence: r.evidence, confidence: r.confidence}]"""
_REL_FILTER = "($rel IS NULL OR type({r}) = $rel)"
LIMIT = 50


def _arrow(direction: Direction, var: str) -> str:
    return _ARROWS[direction].format(r=var)


def _rel_param(rel: RelationType | None) -> str | None:
    return rel.value if rel else None


def relations_of(entity_id: str, rel: RelationType | None, direction: Direction) -> tuple[str, dict]:
    query = f"""
        MATCH p = (e:Entity {{id: $id}}){_arrow(direction, 'r')}(n:Entity)
        WHERE {_REL_FILTER.format(r='r')}
        RETURN {_EDGES} AS edges
        ORDER BY r.confidence DESC, size(r.chunk_ids) DESC LIMIT {LIMIT}"""
    return query, {"id": entity_id, "rel": _rel_param(rel)}


def shared_neighbors(a: str, b: str, rel: RelationType | None, direction: Direction) -> tuple[str, dict]:
    query = f"""
        MATCH p1 = (a:Entity {{id: $a}}){_arrow(direction, 'r1')}(n:Entity),
              p2 = (b:Entity {{id: $b}}){_arrow(direction, 'r2')}(n)
        WHERE {_REL_FILTER.format(r='r1')} AND {_REL_FILTER.format(r='r2')} AND n <> a AND n <> b
        WITH n, p1, p2
        RETURN n.name AS shared,
               {_EDGES.replace('relationships(p)', 'relationships(p1)')} +
               {_EDGES.replace('relationships(p)', 'relationships(p2)')} AS edges
        LIMIT {LIMIT}"""
    return query, {"a": a, "b": b, "rel": _rel_param(rel)}


MAX_HOPS = 3


def chain(entity_id: str, hops: list[tuple[RelationType | None, Direction]]) -> tuple[str, dict]:
    """Follow 1-3 hops from one entity, e.g. Micron -SUPPLIES-> x -COMPETES_WITH- y <-SUPPLIES- z.

    The pattern is assembled from fixed _ARROWS fragments, one per hop; relation names
    go in as parameters $rel0..$rel2.
    """
    if not 1 <= len(hops) <= MAX_HOPS:
        raise ValueError(f"chain needs 1-{MAX_HOPS} hops, got {len(hops)}")
    pattern = "(e:Entity {id: $id})"
    filters, params = [], {"id": entity_id}
    for i, (rel, direction) in enumerate(hops):
        pattern += f"{_arrow(direction, f'r{i}')}(n{i}:Entity)"
        filters.append(f"($rel{i} IS NULL OR type(r{i}) = $rel{i})")
        params[f"rel{i}"] = _rel_param(rel)
    last = f"n{len(hops) - 1}"
    score = " * ".join(f"r{i}.confidence" for i in range(len(hops)))
    query = f"""
        MATCH p = {pattern}
        WHERE {' AND '.join(filters)} AND {last} <> e
        RETURN {last}.name AS reached, {_EDGES} AS edges
        ORDER BY {score} DESC LIMIT {LIMIT}"""
    return query, params


def path_between(a: str, b: str) -> tuple[str, dict]:
    query = f"""
        MATCH (a:Entity {{id: $a}}), (b:Entity {{id: $b}})
        MATCH p = allShortestPaths((a)-[*..3]-(b))
        // Paths through hubs ("both operate in the United States") are true but uninformative.
        RETURN {_EDGES} AS edges
        ORDER BY size([x IN nodes(p) WHERE x:Location OR x:Market]) ASC LIMIT 10"""
    return query, {"a": a, "b": b}


def count(entity_id: str | None, rel: RelationType | None, direction: Direction) -> tuple[str, dict]:
    """With an entity: how many neighbours it has via rel. Without: entities ranked by that count."""
    anchor = "(e:Entity {id: $id})" if entity_id else "(e:Entity)"
    query = f"""
        MATCH p = {anchor}{_arrow(direction, 'r')}(n:Entity)
        WHERE {_REL_FILTER.format(r='r')}
        WITH e, count(DISTINCT n) AS n_related, collect({_EDGES}[0])[..20] AS edges
        RETURN e.name AS entity, n_related AS count, edges
        ORDER BY count DESC LIMIT 10"""
    return query, {"id": entity_id, "rel": _rel_param(rel)}


def entity_profile(entity_id: str) -> tuple[str, dict]:
    query = f"""
        MATCH p = (e:Entity {{id: $id}})-[r]-(n:Entity)
        RETURN {_EDGES} AS edges
        ORDER BY r.confidence DESC, size(r.chunk_ids) DESC LIMIT {LIMIT}"""
    return query, {"id": entity_id}


def run(driver: Driver, query: str, params: dict) -> list[dict]:
    with driver.session() as session:
        return [record.data() for record in session.run(query, params)]
