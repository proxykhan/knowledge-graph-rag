"""Generate a blind benchmark from filing text alone, without consulting the graph.

Usage:
    GEMINI_MODEL=gemini-3.5-flash python -m kgrag.eval.make_blind

Why: the first 50 benchmark questions were written with the graph in view, so they could only
ask about chains the extractor had found. Here nothing touches the graph or the extractions:

  1. Passages are linked by plain text: two (or three) passages from DIFFERENT companies' filings
     that mention the same capitalised name. Chosen with a fixed random seed.
  2. A different model from the one the system uses writes one question per passage set, plus
     the answer items, each with accepted spellings.
  3. Fixed acceptance rule: every answer item's first spelling must appear verbatim in the
     passages. Rejections are logged with their reason; nothing is picked by hand.

Output: eval/blind_questions.jsonl (accepted) and eval/results/blind_generation_log.jsonl (all).
"""

import json
import random
import re
from collections import defaultdict

from pydantic import BaseModel, Field

from kgrag import llm
from kgrag.config import PROCESSED_DIR, PROJECT_ROOT
from kgrag.extract.extractor import _normalize

SEED = 20261011
N_SINGLE, N_TWO, N_THREE = 6, 12, 8
PER_REQUEST = 3
OUT = PROJECT_ROOT / "eval" / "blind_questions.jsonl"
LOG = PROJECT_ROOT / "eval" / "results" / "blind_generation_log.jsonl"

# Capitalised names: runs of capitalised words or all-caps tokens ("SK Hynix", "TSMC", "Mobileye").
NAME = re.compile(r"\b(?:[A-Z][a-z]+[A-Za-z]*|[A-Z]{2,}[a-z]?)(?:\s+(?:[A-Z][a-z]+[A-Za-z]*|[A-Z]{2,}))*\b")
# Words that are capitalised for grammar, not because they name something.
STOP = {"We", "Our", "The", "In", "If", "As", "For", "Item", "Risk", "Factors", "Annual", "Report", "Form",
        "Business", "Data", "Center", "Fiscal", "Note", "Notes", "This", "These", "Any", "Such", "Some", "Many",
        "United", "States", "U", "S", "U.S", "SEC", "GAAP", "AI", "PC", "PCs", "CPU", "GPU", "GPUs", "CPUs",
        "January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
        "November", "December", "Company", "Board", "Directors", "Act", "Part", "II", "III", "IV", "Table",
        "Contents", "Management", "Discussion", "Analysis", "Results", "Operations", "Inc", "Corporation"}
FILER_WORDS = {"NVIDIA", "AMD", "Intel", "Micron", "Applied", "Materials", "Advanced", "Micro", "Devices"}


def names_in(text: str, ticker_words: set[str]) -> set[str]:
    out = set()
    for m in NAME.finditer(text):
        words = m.group(0).split()
        if all(w in STOP for w in words) or len(m.group(0)) < 3:
            continue
        if set(words) <= ticker_words | STOP:
            continue  # the filer naming itself
        out.add(m.group(0))
    return out


class Item(BaseModel):
    spellings: list[str] = Field(description="First: copied verbatim from a passage. Then 0-2 common short forms, e.g. TSMC")


class BlindQuestion(BaseModel):
    set_id: int = Field(description="The number of the passage set this question is for")
    question: str = Field(description="A natural question a business analyst might ask")
    answer_items: list[Item] = Field(description="1-4 short answer items that a correct answer must mention")
    uses_all_passages: bool = Field(description="true if answering needs facts from every passage in the set")


class BlindBatch(BaseModel):
    questions: list[BlindQuestion]


SYSTEM = """You write evaluation questions for a question-answering system over SEC 10-K filings.

For each numbered passage set, write ONE question.
- Multi-passage sets: the question must need facts from EVERY passage in the set (connect them through
  the shared name), and must not be answerable from any single passage. Phrase it naturally, without
  saying "according to passage 2".
- Single-passage sets: ask about one specific fact (a name, number, date or relationship).
- answer_items: 1 to 4 short items a correct answer must mention. The first spelling of each item must be
  copied exactly from the passages; you may add up to two common short forms (e.g. "TSMC").
- Do not ask about the passages themselves, page numbers, or anything not stated in them.
- Set uses_all_passages honestly."""


def build_sets(chunks: list[dict], rng: random.Random) -> list[tuple[str, list[dict]]]:
    by_ticker_words = {c["ticker"]: FILER_WORDS for c in chunks}
    chunk_names = {c["chunk_id"]: names_in(c["text"], by_ticker_words[c["ticker"]]) for c in chunks}
    by_name = defaultdict(list)
    for c in chunks:
        for n in chunk_names[c["chunk_id"]]:
            by_name[n].append(c)

    pairs = []
    for name, cs in by_name.items():
        tickers = {c["ticker"] for c in cs}
        if len(tickers) >= 2:
            for a in cs:
                for b in cs:
                    if a["ticker"] < b["ticker"]:
                        pairs.append((name, a, b))
    rng.shuffle(pairs)

    sets, used = [], set()
    for name, a, b in pairs:  # two-passage sets, each chunk used once
        if len([s for s in sets if s[0] == "two_hop"]) >= N_TWO:
            break
        if a["chunk_id"] in used or b["chunk_id"] in used:
            continue
        used |= {a["chunk_id"], b["chunk_id"]}
        sets.append(("two_hop", [a, b]))

    for name, a, b in pairs:  # three-passage chains: a-b share one name, b-c share another
        if len([s for s in sets if s[0] == "three_hop"]) >= N_THREE:
            break
        if a["chunk_id"] in used or b["chunk_id"] in used:
            continue
        options = [c for n2 in chunk_names[b["chunk_id"]] - {name} for c in by_name[n2]
                   if c["ticker"] not in (a["ticker"], b["ticker"]) and c["chunk_id"] not in used]
        if options:
            c = rng.choice(options)
            used |= {a["chunk_id"], b["chunk_id"], c["chunk_id"]}
            sets.append(("three_hop", [a, b, c]))

    singles = [c for c in chunks if c["chunk_id"] not in used and chunk_names[c["chunk_id"]]]
    for c in rng.sample(singles, N_SINGLE):
        sets.append(("single_hop", [c]))
    return sets


def main() -> None:
    chunks = [json.loads(line) for line in (PROCESSED_DIR / "chunks.jsonl").open(encoding="utf-8")]
    rng = random.Random(SEED)
    sets = build_sets(chunks, rng)
    print(f"{len(sets)} passage sets: " + ", ".join(f"{k}={sum(s[0] == k for s in sets)}"
                                                  for k in ("single_hop", "two_hop", "three_hop")))
    LOG.parent.mkdir(parents=True, exist_ok=True)
    accepted, log = [], []
    for start in range(0, len(sets), PER_REQUEST):
        batch = sets[start:start + PER_REQUEST]
        parts = []
        for i, (kind, passages) in enumerate(batch, 1):
            body = "\n\n".join(f"[Passage {j}] ({p['company']}, {p['section_path']})\n{p['text']}"
                               for j, p in enumerate(passages, 1))
            parts.append(f"=== Passage set {i} ({len(passages)} passage{'s' if len(passages) > 1 else ''}) ===\n{body}")
        try:
            result, _ = llm.generate_structured(SYSTEM, "\n\n".join(parts), BlindBatch)
        except llm.QuotaExhausted as e:
            print(f"STOPPED: {e}")
            break
        except llm.LLMError as e:
            log.append({"batch_start": start, "status": "generation_failed", "reason": str(e)})
            continue
        for q in result.questions:
            if not 1 <= q.set_id <= len(batch):
                continue
            kind, passages = batch[q.set_id - 1]
            text = _normalize(" ".join(p["text"] for p in passages))
            missing = [it.spellings[0] for it in q.answer_items if not it.spellings or _normalize(it.spellings[0]) not in text]
            entry = {"category": kind, "question": q.question, "required": [it.spellings for it in q.answer_items],
                     "gold_source": " + ".join(p["chunk_id"] for p in passages),
                     "uses_all_passages": q.uses_all_passages, "generator": llm.model_name()}
            if not q.answer_items:
                entry |= {"status": "rejected", "reason": "no answer items"}
            elif missing:
                entry |= {"status": "rejected", "reason": f"answer items not verbatim in passages: {missing}"}
            elif kind != "single_hop" and not q.uses_all_passages:
                entry |= {"status": "rejected", "reason": "generator says it does not need every passage"}
            else:
                entry |= {"status": "accepted"}
                accepted.append(entry)
            log.append(entry)
            print(f"[{entry['status']:8}] {kind:10} {q.question}")

    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        for i, e in enumerate(accepted, 1):
            # Random co-mention rarely yields chainable facts: the generator combines facts with
            # "and" rather than chaining them, so these are labelled by fact count, not hop count.
            category = {"two_hop": "two_facts", "three_hop": "three_facts"}.get(e["category"], e["category"])
            f.write(json.dumps({"id": f"b{i:02d}", "category": category, "question": e["question"],
                                "required": e["required"], "gold_source": e["gold_source"]},
                               ensure_ascii=False) + "\n")
    with LOG.open("w", encoding="utf-8", newline="\n") as f:
        for e in log:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"\nAccepted {len(accepted)} of {len(log)} generated questions -> {OUT}")


if __name__ == "__main__":
    main()
