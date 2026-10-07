"""Cheap query router: one light LLM call returns the route, the graph query type and its
parameters (entity mentions, relation types, directions) as a strict schema.

The relation and direction fields are enums, so the model can only choose values the
template library accepts; it never writes Cypher.
"""

from enum import StrEnum

from pydantic import BaseModel, Field

from kgrag import llm
from kgrag.ontology import RelationType
from kgrag.retrieval.templates import Direction, QueryType

ROUTER_VERSION = "v1"
CONFIDENCE_THRESHOLD = 0.7  # below this, run both paths and merge


class Route(StrEnum):
    GRAPH = "GRAPH"
    VECTOR = "VECTOR"


# Relation parameter: any ontology relation, or ANY when the question names none.
RelationParam = StrEnum("RelationParam", {r.name: r.value for r in RelationType} | {"ANY": "ANY"})


class RouterDecision(BaseModel):
    route: Route
    confidence: float = Field(description="0.0 to 1.0: how sure the route is right")
    query_type: QueryType = Field(description="Graph template to use; NONE when route is VECTOR")
    entities: list[str] = Field(description="Entity names exactly as written in the question, in order")
    relation: RelationParam = Field(description="Relationship for the first hop, or ANY")
    direction: Direction = Field(description="out: named entity -> other; in: other -> named entity")
    second_relation: RelationParam = Field(description="TWO_HOP only: relationship for the second hop, else ANY")
    second_direction: Direction = Field(description="TWO_HOP only: direction of the second hop, else any")

    def relation_type(self) -> RelationType | None:
        return None if self.relation.value == "ANY" else RelationType(self.relation.value)

    def second_relation_type(self) -> RelationType | None:
        return None if self.second_relation.value == "ANY" else RelationType(self.second_relation.value)


SYSTEM_PROMPT = f"""You route questions about semiconductor companies' SEC 10-K filings to a retrieval path.

GRAPH: a knowledge graph of entities and relationships extracted from the filings.
  Use it for connections between entities, multi-hop chains, comparisons across entities,
  and counts over relationships.
VECTOR: semantic search over filing passages.
  Use it for definitions, explanations, numbers, dates, policies, risks, and single facts
  stated in one passage.

Relationship types: {", ".join(r.value for r in RelationType)}.
SUPPLIES points from supplier to customer. COMPETES_WITH and PARTNERS_WITH have no direction (use any).
Direction is relative to the first named entity: out = entity -> other, in = other -> entity.

Graph query types:
- RELATIONS_OF: one entity, one relationship. "Who supplies NVIDIA?"
- SHARED_NEIGHBORS: two entities, what they have in common. "Which suppliers do NVIDIA and AMD share?"
- TWO_HOP: one entity, a chain of two relationships. "Who supplies NVIDIA's competitors?"
- PATH_BETWEEN: two entities, how they are connected. "How is Micron connected to AMD?"
- COUNT: how many / who has the most. "Which company lists the most suppliers?"
- ENTITY_PROFILE: everything about one entity, no specific relationship. "What do the filings say about Mobileye's ties?"

Examples (question -> route, query_type, entities, relation, direction, second_relation, second_direction):
- "Who manufactures chips for AMD?" -> GRAPH, RELATIONS_OF, [AMD], SUPPLIES, in, ANY, any
- "What products does Micron make?" -> GRAPH, RELATIONS_OF, [Micron], PRODUCES, out, ANY, any
- "Which companies did Intel acquire?" -> GRAPH, RELATIONS_OF, [Intel], ACQUIRED, out, ANY, any
- "Which suppliers do NVIDIA and AMD have in common?" -> GRAPH, SHARED_NEIGHBORS, [NVIDIA, AMD], SUPPLIES, in, ANY, any
- "Which companies supply NVIDIA's competitors?" -> GRAPH, TWO_HOP, [NVIDIA], COMPETES_WITH, any, SUPPLIES, in
- "What regulations apply to TSMC's customers?" -> GRAPH, TWO_HOP, [TSMC], SUPPLIES, out, SUBJECT_TO, out
- "How is Samsung related to Intel?" -> GRAPH, PATH_BETWEEN, [Samsung, Intel], ANY, any, ANY, any
- "Which company names the most competitors?" -> GRAPH, COUNT, [], COMPETES_WITH, any, ANY, any
- "How many suppliers does AMD list?" -> GRAPH, COUNT, [AMD], SUPPLIES, in, ANY, any
- "What risks does NVIDIA describe about export controls?" -> VECTOR, NONE, [NVIDIA], ANY, any, ANY, any
- "How much did Micron's revenue grow?" -> VECTOR, NONE, [Micron], ANY, any, ANY, any
- "What is HBM?" -> VECTOR, NONE, [], ANY, any, ANY, any
- "When did AMD complete the ZT Systems acquisition?" -> VECTOR, NONE, [AMD, ZT Systems], ANY, any, ANY, any

Questions about a single date, amount or explanation go to VECTOR even if they name entities.
Questions outside these companies' filings: VECTOR with low confidence."""


def route(question: str) -> tuple[RouterDecision, llm.Usage]:
    """Raises llm.LLMError or llm.QuotaExhausted; the retriever handles both."""
    decision, usage = llm.generate_structured(SYSTEM_PROMPT, f"Question: {question}", RouterDecision, light=True)
    return decision, usage
