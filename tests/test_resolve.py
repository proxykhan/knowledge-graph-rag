from kgrag.graph.resolve import normalize, resolve


def _resolve(names_by_type, model=None):
    counts = {(t, n): 1 for t, names in names_by_type.items() for n in names}
    return {frozenset(e["aliases"]): e for e in resolve(counts, model=model)}


def test_legal_suffixes_and_punctuation_collapse():
    assert normalize("Micron Technology, Inc.") == normalize("MICRON TECHNOLOGY INC") == "micron technology"
    clusters = _resolve({"Company": ["Acme Corp", "Acme Corporation", "ACME", "Acme, Inc."]})
    assert len(clusters) == 1


def test_acronym_merges_company_but_not_person():
    clusters = _resolve({
        "Company": ["Taiwan Semiconductor Manufacturing Company Limited", "TSMC", "Intel Corporation"],
        "Person": ["Jensen Huang", "JH"],
    })
    assert frozenset({"Taiwan Semiconductor Manufacturing Company Limited", "TSMC"}) in clusters
    assert frozenset({"Jensen Huang"}) in clusters and frozenset({"JH"}) in clusters


def test_one_word_company_joins_unique_longer_name_only():
    clusters = _resolve({"Company": ["Samsung", "Samsung Electronics Co., Ltd.",
                                     "Applied", "Applied Materials, Inc.", "Applied Ventures, LLC"]})
    assert frozenset({"Samsung", "Samsung Electronics Co., Ltd."}) in clusters
    assert frozenset({"Applied"}) in clusters  # two candidates: ambiguous, no merge


def test_synonyms_give_stable_id():
    clusters = _resolve({"Location": ["Korea", "South Korea", "North Korea"]})
    merged = clusters[frozenset({"Korea", "South Korea"})]
    assert merged["id"] == "location:south-korea"
    assert frozenset({"North Korea"}) in clusters


class FakeModel:
    """Embeds every name identically, so only the shared-word guard can keep names apart."""

    def encode(self, texts, **_):
        import numpy as np
        return np.ones((len(texts), 4)) / 2.0


def test_embedding_merge_requires_shared_significant_word():
    clusters = _resolve({"Company": ["Analog Devices", "Advanced Micro Devices", "Samsung Electronics",
                                     "Samsung Electronics America"]}, model=FakeModel())
    assert frozenset({"Analog Devices"}) in clusters          # only shares "devices", a stopword
    assert frozenset({"Samsung Electronics", "Samsung Electronics America"}) in clusters


def test_embedding_merge_requires_same_numbers():
    clusters = _resolve({"Product": ["Ryzen AI Max+ 388", "Ryzen AI Max+ 392", "Ryzen Z1 Series"]},
                        model=FakeModel())
    assert len(clusters) == 3


def test_official_filer_name_is_canonical():
    clusters = _resolve({"Company": ["Nvidia Corporation", "Nvidia Corporation", "NVIDIA Corporation"]})
    (entity,) = clusters.values()
    assert entity["name"] == "NVIDIA Corporation"


def test_fact_status_comes_from_latest_report_and_hedged_quotes_go_last():
    from kgrag.graph.load import summarize_fact

    fact = {"s": "x", "t": "y", "rel": "SUBSIDIARY_OF", "mentions": {
        "a": {"chunk_id": "a", "evidence": "Altera, our wholly owned subsidiary", "status": "current",
              "confidence": 0.9, "report_date": "2024-12-28"},
        "b": {"chunk_id": "b", "evidence": "If needed, we would sell Altera", "status": "planned",
              "confidence": 0.9, "report_date": "2025-12-27"},
        "c": {"chunk_id": "c", "evidence": "We sold 51% of Altera", "status": "past",
              "confidence": 0.8, "report_date": "2025-12-27"},
    }}
    edge = summarize_fact(fact)
    assert edge["status"] == "past" and edge["years"] == [2024, 2025]
    assert edge["evidence"][0] == "We sold 51% of Altera"   # latest report, not hedged
    assert edge["chunk_ids"] == ["c", "b", "a"]
