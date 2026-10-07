"""Link entity mentions in a question ("Nvidia", "TSMC", "Samsung") to graph node ids.

Tried in order, first hit wins:
  1. exact match on a normalized alias ("nvidia" == normalize("NVIDIA Corporation"))
  2. the mention's words are a subset of exactly one entity's normalized name ("samsung")
     ties go to the most-mentioned entity
  3. embedding similarity >= MIN_SIMILARITY against all aliases
A mention that links to nothing is reported, so the retriever can fall back to vectors.
"""

import json
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from kgrag.config import PROCESSED_DIR
from kgrag.graph.resolve import SYNONYMS, normalize
from kgrag.vector import embed

MIN_SIMILARITY = 0.85


@dataclass
class Link:
    mention: str
    entity_id: str
    name: str
    type: str
    method: str     # "alias" | "words" | "embedding"
    score: float


class EntityLinker:
    def __init__(self, entities: list[dict]):
        self.entities = {e["id"]: e for e in entities}
        self.by_key: dict[str, list[str]] = {}
        self.alias_rows: list[tuple[str, str]] = []  # (alias, entity id)
        for e in entities:
            for alias in set(e["aliases"]) | {e["name"]}:
                key = normalize(alias)
                key = SYNONYMS.get((e["type"], key), key)
                self.by_key.setdefault(key, [])
                if e["id"] not in self.by_key[key]:
                    self.by_key[key].append(e["id"])
                self.alias_rows.append((alias, e["id"]))
        self._alias_vecs = None

    def _best(self, ids: list[str]) -> dict:
        # Prefer companies (the corpus is about companies), then the most-mentioned entity.
        return max((self.entities[i] for i in ids), key=lambda e: (e["type"] == "Company", e["mentions"]))

    def link(self, mention: str) -> Link | None:
        key = normalize(mention)
        exact = self.by_key.get(key, [])
        words = set(key.split())
        by_words = [i for k, ids in self.by_key.items() if words and words <= set(k.split()) for i in ids]

        def is_company(i: str) -> bool:
            return self.entities[i]["type"] == "Company"

        # A company beats a non-company exact match: "Micron" is Micron Technology, Inc., not a
        # product node the extractor once mislabelled "Micron".
        for ids, method in ((exact, "alias"), (by_words, "words")):
            if any(is_company(i) for i in ids):
                e = self._best([i for i in ids if is_company(i)])
                return Link(mention, e["id"], e["name"], e["type"], method, 1.0)
        for ids, method in ((exact, "alias"), (by_words, "words")):
            if ids:
                e = self._best(ids)
                return Link(mention, e["id"], e["name"], e["type"], method, 1.0)

        if self._alias_vecs is None:
            self._alias_vecs = embed.embed_passages([a for a, _ in self.alias_rows])
        sims = self._alias_vecs @ embed.embed_passages([mention])[0]
        best = int(np.argmax(sims))
        if sims[best] >= MIN_SIMILARITY:
            e = self.entities[self.alias_rows[best][1]]
            return Link(mention, e["id"], e["name"], e["type"], "embedding", float(sims[best]))
        return None


@lru_cache(maxsize=1)
def default_linker() -> EntityLinker:
    return EntityLinker(json.loads((PROCESSED_DIR / "entities.json").read_text(encoding="utf-8")))
