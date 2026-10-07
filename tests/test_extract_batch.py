from kgrag.extract.extractor import make_batches, split_batch
from kgrag.extract.schema import BatchExtraction

CHUNKS = [
    {"chunk_id": "D:1:0000", "doc_id": "D", "text": "Taiwan Semiconductor Manufacturing Company Limited makes our wafers."},
    {"chunk_id": "D:1:0001", "doc_id": "D", "text": "We compete with Advanced Micro Devices, Inc. in data center."},
]
ENTS = [{"name": "NVIDIA Corporation", "type": "Company"},
        {"name": "Taiwan Semiconductor Manufacturing Company Limited", "type": "Company"}]
TSMC_REL = {"source": "Taiwan Semiconductor Manufacturing Company Limited", "source_type": "Company",
            "relation": "SUPPLIES", "target": "NVIDIA Corporation", "target_type": "Company",
            "evidence": "Taiwan Semiconductor Manufacturing Company Limited makes our wafers.", "confidence": 0.95}


def test_fact_attributed_to_wrong_excerpt_is_moved_with_its_entities():
    batch = BatchExtraction.model_validate({"excerpts": [
        {"excerpt_id": 1, "entities": [], "relationships": []},
        {"excerpt_id": 2, "entities": ENTS, "relationships": [TSMC_REL]},  # quote is really from excerpt 1
    ]})
    first, second = split_batch(batch, CHUNKS)
    assert [r.relation for r in first.relationships] == ["SUPPLIES"]
    assert {e.name for e in first.entities} == {e["name"] for e in ENTS}
    assert second.relationships == []


def test_skipped_excerpt_is_none_so_it_is_retried():
    batch = BatchExtraction.model_validate({"excerpts": [
        {"excerpt_id": 1, "entities": ENTS, "relationships": [TSMC_REL]},
    ]})
    first, second = split_batch(batch, CHUNKS)
    assert len(first.relationships) == 1
    assert second is None


def test_batches_never_mix_filings():
    chunks = [{"doc_id": "A"}] * 5 + [{"doc_id": "B"}] * 2
    assert [len(b) for b in make_batches(chunks)] == [4, 1, 2]
