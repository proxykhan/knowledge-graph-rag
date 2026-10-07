import pytest

from kgrag.ontology import RelationType as R
from kgrag.retrieval import templates as T
from kgrag.retrieval.entity_linker import EntityLinker, Link
from kgrag.retrieval.retriever import build_graph_query
from kgrag.retrieval.router import RouterDecision

HOSTILE = "x'}) MATCH (n) DETACH DELETE n //"


@pytest.mark.parametrize("query", [
    T.relations_of(HOSTILE, R.SUPPLIES, T.Direction.IN),
    T.shared_neighbors(HOSTILE, HOSTILE, R.SUPPLIES, T.Direction.IN),
    T.chain(HOSTILE, [(R.COMPETES_WITH, T.Direction.ANY), (R.SUPPLIES, T.Direction.IN)]),
    T.path_between(HOSTILE, HOSTILE),
    T.count(HOSTILE, None, T.Direction.ANY),
    T.entity_profile(HOSTILE),
])
def test_values_never_reach_query_text(query):
    cypher, params = query
    assert HOSTILE not in cypher            # only ever a parameter
    assert HOSTILE in params.values()
    assert "DELETE" not in cypher and "SET" not in cypher and "CREATE" not in cypher


def _decision(**kw):
    base = dict(route="GRAPH", confidence=0.9, query_type="RELATIONS_OF", entities=["NVIDIA"],
                hops=[{"relation": "SUPPLIES", "direction": "in"}])
    return RouterDecision.model_validate(base | kw)


def _link(eid):
    return Link(eid, eid, eid, "Company", "alias", 1.0)


def test_template_needs_its_entities():
    assert build_graph_query(_decision(query_type="SHARED_NEIGHBORS"), [_link("company:nvidia")]) is None
    cypher, params = build_graph_query(_decision(query_type="SHARED_NEIGHBORS"),
                                       [_link("company:nvidia"), _link("company:amd")])
    assert params == {"a": "company:nvidia", "b": "company:amd", "rel": "SUPPLIES"}


def test_any_relation_becomes_null_parameter():
    _, params = build_graph_query(_decision(hops=[{"relation": "ANY", "direction": "any"}]),
                                  [_link("company:nvidia")])
    assert params["rel"] is None


def test_router_rejects_relations_outside_the_ontology():
    with pytest.raises(ValueError):
        _decision(hops=[{"relation": "OWNS_SECRETLY", "direction": "in"}])


def test_linker_prefers_company_over_mislabelled_product():
    linker = EntityLinker([
        {"id": "product:micron", "type": "Product", "name": "Micron", "aliases": ["Micron"], "mentions": 1},
        {"id": "company:micron-technology", "type": "Company", "name": "Micron Technology, Inc.",
         "aliases": ["Micron Technology, Inc."], "mentions": 9},
    ])
    assert linker.link("Micron").entity_id == "company:micron-technology"
    assert linker.link("micron technology").method == "alias"


def test_three_hop_chain_builds_one_pattern_with_parameters():
    cypher, params = build_graph_query(_decision(query_type="CHAIN", entities=["Micron"], hops=[
        {"relation": "SUPPLIES", "direction": "out"},
        {"relation": "COMPETES_WITH", "direction": "any"},
        {"relation": "SUPPLIES", "direction": "in"}]), [_link("company:micron-technology")])
    assert "-[r0]->(n0:Entity)-[r1]-(n1:Entity)<-[r2]-(n2:Entity)" in cypher
    assert params == {"id": "company:micron-technology", "rel0": "SUPPLIES", "rel1": "COMPETES_WITH", "rel2": "SUPPLIES"}


def test_chain_rejects_too_many_hops():
    with pytest.raises(ValueError):
        T.chain("x", [(None, T.Direction.ANY)] * 4)
