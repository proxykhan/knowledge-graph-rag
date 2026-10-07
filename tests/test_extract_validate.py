from kgrag.extract.extractor import validate
from kgrag.extract.schema import ChunkExtraction

CHUNK = "We rely on Taiwan Semiconductor Manufacturing Company Limited to produce our GPUs. AMD competes with us."


def _extraction(relationships):
    return ChunkExtraction.model_validate({
        "entities": [
            {"name": "NVIDIA Corporation", "type": "Company"},
            {"name": "Taiwan Semiconductor Manufacturing Company Limited", "type": "Company"},
            {"name": "Advanced Micro Devices, Inc.", "type": "Company"},
        ],
        "relationships": relationships,
    })


def _rel(source, relation, target, evidence, confidence=0.9):
    return {"source": source, "source_type": "Company", "relation": relation, "target": target,
            "target_type": "Company", "evidence": evidence, "confidence": confidence}


def test_valid_relationship_kept_with_chunk_id():
    result = validate(_extraction([_rel(
        "Taiwan Semiconductor Manufacturing Company Limited", "SUPPLIES", "NVIDIA Corporation",
        "We rely on Taiwan Semiconductor Manufacturing Company Limited to produce our GPUs.")]), "c1", CHUNK)
    assert len(result.relationships) == 1 and not result.rejected
    assert result.relationships[0]["chunk_id"] == "c1"


def test_evidence_must_appear_in_chunk():
    result = validate(_extraction([_rel(
        "Taiwan Semiconductor Manufacturing Company Limited", "SUPPLIES", "NVIDIA Corporation",
        "TSMC is our sole foundry.")]), "c1", CHUNK)
    assert not result.relationships
    assert result.rejected[0]["reason"] == "evidence quote not found verbatim in chunk"


def test_evidence_match_ignores_curly_quotes_and_whitespace():
    chunk = "Our  partner’s fabs\nsupply us."
    result = validate(ChunkExtraction.model_validate({
        "entities": [{"name": "A", "type": "Company"}, {"name": "B", "type": "Company"}],
        "relationships": [_rel("A", "SUPPLIES", "B", "our partner's fabs supply us")],
    }), "c1", chunk)
    assert len(result.relationships) == 1


def test_unknown_entity_and_symmetric_ordering():
    result = validate(_extraction([
        _rel("Intel Corporation", "COMPETES_WITH", "NVIDIA Corporation", "AMD competes with us."),
        _rel("NVIDIA Corporation", "COMPETES_WITH", "Advanced Micro Devices, Inc.", "AMD competes with us."),
    ]), "c1", CHUNK)
    assert result.rejected[0]["reason"] == "source or target missing from entities list"
    # COMPETES_WITH is symmetric: stored in alphabetical order so it exists once, not twice.
    assert result.relationships[0]["source"] == "Advanced Micro Devices, Inc."
