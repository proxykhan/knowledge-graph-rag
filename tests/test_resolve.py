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
