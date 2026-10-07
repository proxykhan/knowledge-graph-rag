"""The strict output schema the LLM must return.

Several chunks (excerpts) go into one request to stay within free-tier daily request
limits. The model returns results per excerpt number; our code maps that number to the
chunk id and verifies it via the evidence quote, so chunk ids never come from the model.
"""

from pydantic import BaseModel, Field

from kgrag.ontology import EntityType, RelationType


class ExtractedEntity(BaseModel):
    name: str = Field(description="Canonical full name, e.g. 'Taiwan Semiconductor Manufacturing Company Limited'")
    type: EntityType


class ExtractedRelationship(BaseModel):
    source: str = Field(description="Name of the source entity, exactly as listed in entities")
    source_type: EntityType
    relation: RelationType
    target: str = Field(description="Name of the target entity, exactly as listed in entities")
    target_type: EntityType
    evidence: str = Field(description="Verbatim quote from the excerpt (one sentence or less) that states this fact")
    confidence: float = Field(description="0.0 to 1.0: how explicitly the excerpt states this fact")


class ChunkExtraction(BaseModel):
    entities: list[ExtractedEntity]
    relationships: list[ExtractedRelationship]


class ExcerptExtraction(ChunkExtraction):
    excerpt_id: int = Field(description="The number of the excerpt these facts come from")


class BatchExtraction(BaseModel):
    excerpts: list[ExcerptExtraction] = Field(description="One entry per excerpt, in order")
