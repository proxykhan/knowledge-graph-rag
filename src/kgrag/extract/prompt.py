"""Extraction prompt. The system prompt is identical for every chunk, so it is cached.

Bump PROMPT_VERSION whenever the prompt or schema changes: it is part of the cache
key, so old cached extractions are not silently mixed with new ones.
"""

from kgrag.ontology import ALLOWED_RELATIONS, EntityType

PROMPT_VERSION = "v3"

_ENTITY_GUIDE = {
    EntityType.COMPANY: "Corporations, including subsidiaries, customers, suppliers, foundries, competitors.",
    EntityType.PERSON: "Named individuals (executives, directors).",
    EntityType.PRODUCT: "Named product lines or products, e.g. 'H100', 'EPYC', 'Xeon', 'HBM3E'.",
    EntityType.TECHNOLOGY: "Process or technical capabilities, e.g. 'EUV lithography', '18A process node', 'CoWoS packaging'.",
    EntityType.MARKET: "End markets or segments, e.g. 'Data Center', 'Automotive', 'PC'.",
    EntityType.LOCATION: "Countries, regions or cities where an entity has operations or exposure.",
    EntityType.REGULATION: "Laws, regulators, government programs, e.g. 'CHIPS and Science Act', 'U.S. export controls'.",
}


def _ontology_text() -> str:
    entities = "\n".join(f"- {t.value}: {desc}" for t, desc in _ENTITY_GUIDE.items())
    relations = "\n".join(
        f"- {rel.value}: " + ", ".join(f"{s.value} -> {t.value}" for s, t in sorted(pairs))
        for rel, pairs in ALLOWED_RELATIONS.items()
    )
    return f"Entity types:\n{entities}\n\nRelationship types (allowed source -> target types):\n{relations}"


SYSTEM_PROMPT = f"""You extract a knowledge graph from excerpts of SEC 10-K filings by semiconductor companies.

Use ONLY this ontology. Never invent new entity or relationship types.

{_ontology_text()}

Rules:
1. Extract only facts the excerpt states. Do not add knowledge from outside the excerpt.
2. "We", "our", "us" and "the Company" refer to the filing company named in the request. Use its full name.
3. Company and Person: use the most complete official name ("Taiwan Semiconductor Manufacturing Company Limited", not "TSMC", when the full name appears; otherwise the name as written).
   Product, Technology, Market, Location, Regulation: use the short standard name, not a description ("U.S. export controls", not "export controls enacted by the United States government"; "DRAM", not "dynamic random access memory (DRAM)"; "South Korea", not "Korea").
4. SUPPLIES always points from supplier to customer. "We buy wafers from TSMC" is TSMC SUPPLIES <filer>.
5. Every relationship's source and target must also appear in the entities list, with matching types.
6. evidence must be copied verbatim from the excerpt: one sentence or less, no paraphrasing.
7. confidence: 0.9+ when stated outright, 0.6-0.8 when clearly implied, below 0.6 when uncertain.
8. Generic mentions ("our customers", "competitors", "foreign governments") are not entities. Skip them.
   Reporting segments and business units ("Semiconductor Systems segment", "Client Computing Group") are not Products.
9. Risk-factor hypotheticals ("if a supplier fails...") are not relationships unless a specific named entity is involved.
10. If an excerpt contains no extractable facts, return empty lists for it.
11. You may receive several numbered excerpts. Return one entry per excerpt with its excerpt_id.
    Each fact belongs to the excerpt its evidence quote is copied from. List each excerpt's entities in that excerpt's entry."""


def user_message(company: str, chunks: list[dict]) -> str:
    """chunks: dicts with section_path and text, numbered 1..n in the message."""
    parts = [f"Filing company: {company}"]
    for i, c in enumerate(chunks, 1):
        parts.append(f'<excerpt id="{i}" section="{c["section_path"]}">\n{c["text"]}\n</excerpt>')
    return "\n\n".join(parts)
