"""Cheap query router: one light LLM call returns the route, the graph query type and its
parameters (entity mentions, and a list of hops: relation type + direction) as a strict schema.

Relations and directions are enums, so the model can only choose values the template
library accepts; it never writes Cypher.
"""

from enum import StrEnum

from pydantic import BaseModel, Field

from kgrag import llm
from kgrag.ontology import RelationType
from kgrag.retrieval.templates import MAX_HOPS, Direction, QueryType

ROUTER_VERSION = "v2"
CONFIDENCE_THRESHOLD = 0.7  # below this, run both paths and merge


class Route(StrEnum):
    GRAPH = "GRAPH"
    VECTOR = "VECTOR"


# Relation parameter: any ontology relation, or ANY when the question names none.
RelationParam = StrEnum("RelationParam", {r.name: r.value for r in RelationType} | {"ANY": "ANY"})


class Hop(BaseModel):
    relation: RelationParam
    direction: Direction = Field(description="out: previous entity -> next; in: next -> previous; any")

    def relation_type(self) -> RelationType | None:
        return None if self.relation.value == "ANY" else RelationType(self.relation.value)


class RouterDecision(BaseModel):
    route: Route
    confidence: float = Field(description="0.0 to 1.0: how sure the route is right")
    query_type: QueryType = Field(description="Graph template to use; NONE when route is VECTOR")
    entities: list[str] = Field(description="Entity names exactly as written in the question, in order")
    hops: list[Hop] = Field(description=f"Relationship hops starting from the first entity, 1 to {MAX_HOPS}. "
                                        "One hop for every query type except CHAIN; empty for VECTOR")

    def hop_params(self) -> list[tuple[RelationType | None, Direction]]:
        return [(h.relation_type(), h.direction) for h in self.hops[:MAX_HOPS]]


SYSTEM_PROMPT = f"""You route questions about semiconductor companies' SEC 10-K filings to a retrieval path.

GRAPH: a knowledge graph of entities and relationships extracted from the filings.
  Use it for connections between entities, multi-hop chains, comparisons across entities,
  and counts over relationships.
VECTOR: semantic search over filing passages.
  Use it for definitions, explanations, numbers, dates, policies, risks, and single facts
  stated in one passage.

Relationship types: {", ".join(r.value for r in RelationType)}.
SUPPLIES points from supplier to customer. COMPETES_WITH and PARTNERS_WITH have no direction (use any).
A hop's direction is relative to the entity it starts from: out = that entity -> next, in = next -> that entity.

Graph query types:
- RELATIONS_OF: one entity, one hop. "Who supplies NVIDIA?"
- SHARED_NEIGHBORS: two entities, what they have in common through one hop. "Which suppliers do NVIDIA and AMD share?"
- CHAIN: one entity, a chain of 2 or 3 hops, read outward from the entity. "Who supplies NVIDIA's competitors?"
- PATH_BETWEEN: two entities, how they are connected. "How is Micron connected to AMD?"
- COUNT: how many / who has the most, one hop. "Which company lists the most suppliers?"
- ENTITY_PROFILE: everything about one entity, no specific relationship.

Examples (question -> route, query_type, entities, hops as relation/direction):
- "Who manufactures chips for AMD?" -> GRAPH, RELATIONS_OF, [AMD], [SUPPLIES/in]
- "What products does Micron make?" -> GRAPH, RELATIONS_OF, [Micron], [PRODUCES/out]
- "Which companies did Intel acquire?" -> GRAPH, RELATIONS_OF, [Intel], [ACQUIRED/out]
- "Which suppliers do NVIDIA and AMD have in common?" -> GRAPH, SHARED_NEIGHBORS, [NVIDIA, AMD], [SUPPLIES/in]
- "Which companies supply NVIDIA's competitors?" -> GRAPH, CHAIN, [NVIDIA], [COMPETES_WITH/any, SUPPLIES/in]
- "What regulations apply to TSMC's customers?" -> GRAPH, CHAIN, [TSMC], [SUPPLIES/out, SUBJECT_TO/out]
- "Who supplies the competitors of SK hynix's customers?" -> GRAPH, CHAIN, [SK hynix], [SUPPLIES/out, COMPETES_WITH/any, SUPPLIES/in]
- "How is Samsung related to Intel?" -> GRAPH, PATH_BETWEEN, [Samsung, Intel], [ANY/any]
- "Which company names the most competitors?" -> GRAPH, COUNT, [], [COMPETES_WITH/any]
- "How many suppliers does AMD list?" -> GRAPH, COUNT, [AMD], [SUPPLIES/in]
- "What risks does NVIDIA describe about export controls?" -> VECTOR, NONE, [NVIDIA], []
- "How much did Micron's revenue grow?" -> VECTOR, NONE, [Micron], []
- "What is HBM?" -> VECTOR, NONE, [], []
- "When did AMD complete the ZT Systems acquisition?" -> VECTOR, NONE, [AMD, ZT Systems], []

Questions about a single date, amount or explanation go to VECTOR even if they name entities.
Questions outside these companies' filings: VECTOR with low confidence."""


def route(question: str) -> tuple[RouterDecision, llm.Usage]:
    """Raises llm.LLMError or llm.QuotaExhausted; the retriever handles both."""
    decision, usage = llm.generate_structured(SYSTEM_PROMPT, f"Question: {question}", RouterDecision, light=True)
    return decision, usage
