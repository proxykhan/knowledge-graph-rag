"""Entity resolution: collapse surface forms ("Acme Corp", "Acme Corporation", "ACME") into one node.

Usage:
    python -m kgrag.graph.resolve            # writes data/processed/entities.json
    python -m kgrag.graph.resolve --report   # also prints near-threshold pairs, for tuning

Two names of the same entity type merge when any of these holds:
  1. same normalized key (lowercase, punctuation and legal suffixes like "Inc." removed)
  2. one is the acronym of the other ("TSMC" / "Taiwan Semiconductor Manufacturing Company")
  3. a manual synonym (SYNONYMS below)
  4. embedding cosine >= threshold AND they share a significant word AND their numbers match.
     The word check stops look-alike names ("Analog Devices" / "Advanced Micro Devices") from
     merging; the number check stops product generations ("Ryzen AI Max 388" / "392",
     "RDNA 3.5" / "RDNA 4"), which embeddings barely tell apart.
"""

import argparse
import json
import re
from collections import Counter, defaultdict

import numpy as np

from kgrag.config import PROCESSED_DIR, settings
from kgrag.graph.extractions import load_extractions
from kgrag.ingest.companies import COMPANY_NAMES

OFFICIAL_NAMES = set(COMPANY_NAMES.values())

# Per-type cosine thresholds, tuned with --report on this corpus.
THRESHOLDS = {"Company": 0.92, "Person": 0.95, "Location": 0.93, "Product": 0.95, "Technology": 0.95}
DEFAULT_THRESHOLD = 0.90

LEGAL_SUFFIXES = {"inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited",
                  "llc", "plc", "nv", "sa", "ag", "se", "the"}
# Words too generic to count as "sharing a significant word".
STOPWORDS = LEGAL_SUFFIXES | {"of", "and", "&", "for", "in", "on", "united", "north", "south", "east",
                              "west", "technology", "technologies", "semiconductor", "semiconductors",
                              "systems", "group", "international", "global", "electronics", "devices"}

# Acronyms are meaningful for these types; for people, places and products they mostly collide.
ACRONYM_TYPES = {"Company", "Technology", "Regulation", "Market"}

# (type, normalized key) -> normalized key it is the same as.
SYNONYMS = {
    ("Location", "korea"): "south korea",
    ("Location", "republic of korea"): "south korea",
    ("Location", "us"): "united states",
    ("Location", "usa"): "united states",
    ("Location", "united states of america"): "united states",
    ("Location", "prc"): "china",
    ("Location", "peoples republic of china"): "china",
    ("Location", "mainland china"): "china",
    ("Company", "gf"): "globalfoundries",  # nickname, not an acronym of a one-word name
    ("Company", "global foundries"): "globalfoundries",
    ("Company", "zt group int l"): "zt systems",  # legal name of ZT Systems
}


def tokens(name: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", name.lower().replace("&", " and "))


def normalize(name: str) -> str:
    toks = tokens(name)
    core = [t for t in toks if t not in LEGAL_SUFFIXES] or toks
    return " ".join(core)


def acronyms(name: str) -> set[str]:
    """Acronyms with 0, 1, 2... trailing legal suffixes dropped:
    "Taiwan Semiconductor Manufacturing Company Limited" -> tsmcl, tsmc, tsm."""
    toks = [t for t in tokens(name) if t not in {"of", "and", "the"}]
    out = set()
    while toks:
        out.add("".join(t[0] for t in toks))
        if toks[-1] not in LEGAL_SUFFIXES:
            break
        toks = toks[:-1]
    return {a for a in out if len(a) >= 2}


def numbers(key: str) -> set[str]:
    return {t for t in key.split() if any(ch.isdigit() for ch in t)}


def significant_words(key: str) -> set[str]:
    return {t for t in key.split() if t not in STOPWORDS and len(t) > 1}


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def resolve(surface_counts: dict[tuple[str, str], int], model=None, report: bool = False) -> list[dict]:
    """surface_counts: (type, surface name) -> number of mentions. Returns resolved entities."""
    by_type: dict[str, list[str]] = defaultdict(list)
    for etype, name in surface_counts:
        by_type[etype].append(name)

    resolved = []
    for etype, names in sorted(by_type.items()):
        # Steps 1 + 3: exact normalized keys (after synonyms) form the initial groups.
        keys = []
        for n in names:
            k = normalize(n)
            keys.append(SYNONYMS.get((etype, k), k))
        uniq_keys = sorted(set(keys))
        idx = {k: i for i, k in enumerate(uniq_keys)}
        uf = UnionFind(len(uniq_keys))

        # Step 2: acronyms. A short single-token key merges with a longer key whose acronym it is.
        # Built from surface names: the normalized key has lost words the acronym may use
        # ("Taiwan Semiconductor Manufacturing Company" -> TSMC, not TSM).
        acro_index = defaultdict(set)
        for n, k in zip(names, keys):
            for a in acronyms(n):
                acro_index[a].add(k)
        for k in uniq_keys if etype in ACRONYM_TYPES else []:
            # 3+ letters: two-letter acronyms are ambiguous ("EC": export controls or European Commission)
            if " " not in k and 3 <= len(k) <= 6:
                for long_key in acro_index.get(k, []):
                    if long_key != k and " " in long_key:
                        uf.union(idx[k], idx[long_key])

        # Step 2b: a one-word company name ("samsung") joins the single longer company name
        # starting with that word ("samsung electronics"). If two longer names share the
        # word ("applied materials", "applied ventures") it is ambiguous and nothing merges.
        if etype == "Company":
            for k in uniq_keys:
                if " " not in k and len(k) >= 3:
                    longer = [o for o in uniq_keys if o != k and o.split()[0] == k]
                    if len(longer) == 1:
                        uf.union(idx[k], idx[longer[0]])

        # Step 4: embedding similarity + shared significant word.
        if model is not None and len(uniq_keys) > 1:
            emb = model.encode(uniq_keys, normalize_embeddings=True, show_progress_bar=False)
            sims = emb @ emb.T
            threshold = THRESHOLDS.get(etype, DEFAULT_THRESHOLD)
            words = [significant_words(k) for k in uniq_keys]
            nums = [numbers(k) for k in uniq_keys]
            for i in range(len(uniq_keys)):
                for j in range(i + 1, len(uniq_keys)):
                    s = float(sims[i, j])
                    if s < threshold - 0.08:
                        continue
                    shared = bool(words[i] & words[j])
                    same_numbers = nums[i] == nums[j]
                    merged = s >= threshold and shared and same_numbers
                    if merged:
                        uf.union(i, j)
                    if report:
                        mark = ("MERGE" if merged else "below" if s < threshold
                                else "no-shared-word" if not shared else "numbers-differ")
                        print(f"  [{etype}] {s:.3f} {mark:>14}  {uniq_keys[i]!r} ~ {uniq_keys[j]!r}")

        # Build clusters of surface names.
        clusters: dict[int, list[str]] = defaultdict(list)
        for name, k in zip(names, keys):
            clusters[uf.find(idx[k])].append(name)
        for members in clusters.values():
            # Canonical name, in order of preference: an official filer name, a synonym target,
            # a multi-word name rather than an acronym ("TSMC"), the most mentioned form, the longest.
            targets = set(SYNONYMS.values())
            canonical = max(members, key=lambda n: (n in OFFICIAL_NAMES, normalize(n) in targets,
                                                    " " in normalize(n),  # full name over acronym
                                                    surface_counts[(etype, n)], len(n)))
            key = SYNONYMS.get((etype, normalize(canonical)), normalize(canonical))
            resolved.append({
                "id": f"{etype.lower()}:{slug(key) or slug(canonical)}",
                "type": etype,
                "name": canonical,
                "aliases": sorted(set(members)),
                "mentions": sum(surface_counts[(etype, n)] for n in members),
            })

    # Two clusters can slug to the same id only if their canonical names normalize identically,
    # which step 1 already merged; assert so a silent collision can never corrupt the graph.
    ids = Counter(e["id"] for e in resolved)
    assert all(c == 1 for c in ids.values()), [i for i, c in ids.items() if c > 1]
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true", help="print near-threshold pairs")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    surface_counts: Counter = Counter()
    for record in load_extractions():
        for e in record["entities"]:
            surface_counts[(e["type"], e["name"])] += 1

    model = SentenceTransformer(settings.embedding_model)
    entities = resolve(surface_counts, model=model, report=args.report)

    out = PROCESSED_DIR / "entities.json"
    out.write_text(json.dumps(sorted(entities, key=lambda e: (e["type"], e["name"])),
                              indent=2, ensure_ascii=False), encoding="utf-8")
    merged = [e for e in entities if len(e["aliases"]) > 1]
    print(f"{len(surface_counts)} surface names -> {len(entities)} entities "
          f"({len(merged)} with aliases). Wrote {out}")
    for e in sorted(merged, key=lambda e: -len(e["aliases"]))[:15]:
        print(f"  {e['type']:<10} {e['name']}  <-  {', '.join(a for a in e['aliases'] if a != e['name'])}")


if __name__ == "__main__":
    main()
