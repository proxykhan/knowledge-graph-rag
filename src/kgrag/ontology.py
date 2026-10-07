"""The fixed ontology for the graph.

Phase 1 rule from the build guide: constrain the ontology BEFORE writing extraction
code (5-10 entity types, 8-15 relationship types). Extraction output that does not
fit this ontology is rejected, so the graph stays queryable.

Domain: semiconductor industry 10-K filings.
"""

from enum import StrEnum


class EntityType(StrEnum):
    COMPANY = "Company"          # any corporation, incl. subsidiaries, customers, suppliers
    PERSON = "Person"            # executives, directors
    PRODUCT = "Product"          # named product lines: "H100", "EPYC", "Snapdragon"
    TECHNOLOGY = "Technology"    # process/tech: "EUV lithography", "HBM", "3nm process"
    MARKET = "Market"            # end markets / segments: "Data Center", "Automotive"
    LOCATION = "Location"        # countries, regions, cities with operations
    REGULATION = "Regulation"    # laws, regulators, programs: "CHIPS Act", "U.S. export controls"


class RelationType(StrEnum):
    SUBSIDIARY_OF = "SUBSIDIARY_OF"
    ACQUIRED = "ACQUIRED"
    COMPETES_WITH = "COMPETES_WITH"
    SUPPLIES = "SUPPLIES"                # supplier -> customer (we never store CUSTOMER_OF;
                                         # one direction only, so queries have one shape)
    PARTNERS_WITH = "PARTNERS_WITH"
    EXECUTIVE_OF = "EXECUTIVE_OF"
    PRODUCES = "PRODUCES"
    USES_TECHNOLOGY = "USES_TECHNOLOGY"
    SERVES_MARKET = "SERVES_MARKET"
    OPERATES_IN = "OPERATES_IN"
    SUBJECT_TO = "SUBJECT_TO"


E, R = EntityType, RelationType

# Allowed (source type, target type) pairs per relationship. Anything else is rejected.
ALLOWED_RELATIONS: dict[RelationType, set[tuple[EntityType, EntityType]]] = {
    R.SUBSIDIARY_OF: {(E.COMPANY, E.COMPANY)},
    R.ACQUIRED: {(E.COMPANY, E.COMPANY), (E.COMPANY, E.PRODUCT), (E.COMPANY, E.TECHNOLOGY)},
    R.COMPETES_WITH: {(E.COMPANY, E.COMPANY), (E.PRODUCT, E.PRODUCT)},
    R.SUPPLIES: {(E.COMPANY, E.COMPANY)},
    R.PARTNERS_WITH: {(E.COMPANY, E.COMPANY)},
    R.EXECUTIVE_OF: {(E.PERSON, E.COMPANY)},
    R.PRODUCES: {(E.COMPANY, E.PRODUCT)},
    R.USES_TECHNOLOGY: {(E.PRODUCT, E.TECHNOLOGY), (E.COMPANY, E.TECHNOLOGY)},
    R.SERVES_MARKET: {(E.COMPANY, E.MARKET), (E.PRODUCT, E.MARKET)},
    R.OPERATES_IN: {(E.COMPANY, E.LOCATION)},
    R.SUBJECT_TO: {(E.COMPANY, E.REGULATION), (E.PRODUCT, E.REGULATION)},
}

# Relationships with no direction: store once, query both ways.
SYMMETRIC_RELATIONS = {R.COMPETES_WITH, R.PARTNERS_WITH}


def is_allowed(rel: RelationType, source: EntityType, target: EntityType) -> bool:
    return (source, target) in ALLOWED_RELATIONS[rel]
