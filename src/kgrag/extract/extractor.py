"""Phase 1 extraction: chunks -> LLM -> schema validation -> ontology validation -> cache.

Usage:
    python -m kgrag.extract.extractor --limit 5                    # pilot: first 5 chunks
    python -m kgrag.extract.extractor --contains "Taiwan Semiconductor"
    python -m kgrag.extract.extractor                              # all chunks in chunks.jsonl

BATCH_SIZE consecutive chunks of one filing go into each request, so the free tier's
requests-per-day limit covers BATCH_SIZE times more text. Results are still validated and
cached per chunk under data/cache/extraction/<model>/<prompt version>/<doc hash>/, so
re-running only calls the LLM for chunks it has not seen. On the Gemini free tier a run
stops cleanly when the daily quota is reached; run it again the next day to continue.
"""

import argparse
import json
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from kgrag import llm
from kgrag.config import CACHE_DIR, PROCESSED_DIR
from kgrag.extract.prompt import PROMPT_VERSION, SYSTEM_PROMPT, user_message
from kgrag.extract.schema import BatchExtraction, ChunkExtraction, ExtractedEntity
from kgrag.ingest.companies import COMPANY_NAMES
from kgrag.ontology import SYMMETRIC_RELATIONS, is_allowed

BATCH_SIZE = 4
MAX_ATTEMPTS = 3
CONCURRENCY = 2  # requests are paced by llm.py; this only overlaps network waits


@dataclass
class ChunkResult:
    chunk_id: str
    entities: list[dict] = field(default_factory=list)
    relationships: list[dict] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)   # failed ontology/evidence checks, with reason
    error: str | None = None                             # set when extraction failed; chunk is not cached
    attempts: int = 0


def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip().lower()


def validate(extraction: ChunkExtraction, chunk_id: str, chunk_text: str) -> ChunkResult:
    """Ontology + grounding checks the JSON schema cannot express."""
    result = ChunkResult(chunk_id=chunk_id)
    entities = {}
    for e in extraction.entities:
        name = e.name.strip()
        if name:
            entities[(name, e.type)] = {"name": name, "type": e.type.value}
    result.entities = list(entities.values())

    text_norm = _normalize(chunk_text)
    for r in extraction.relationships:
        rel = r.model_dump(mode="json") | {"source": r.source.strip(), "target": r.target.strip(),
                                            "chunk_id": chunk_id}
        if not is_allowed(r.relation, r.source_type, r.target_type):
            reason = f"{r.relation.value} not allowed for {r.source_type.value} -> {r.target_type.value}"
        elif (rel["source"], r.source_type) not in entities or (rel["target"], r.target_type) not in entities:
            reason = "source or target missing from entities list"
        elif _normalize(rel["source"]) == _normalize(rel["target"]):
            reason = "self-loop"
        elif _normalize(r.evidence) not in text_norm:
            reason = "evidence quote not found verbatim in chunk"
        elif not 0.0 <= r.confidence <= 1.0:
            reason = "confidence outside 0-1"
        else:
            if r.relation in SYMMETRIC_RELATIONS and rel["source"] > rel["target"]:
                rel["source"], rel["target"] = rel["target"], rel["source"]
                rel["source_type"], rel["target_type"] = rel["target_type"], rel["source_type"]
            result.relationships.append(rel)
            continue
        result.rejected.append(rel | {"reason": reason})
    return result


def split_batch(batch: BatchExtraction, chunks: list[dict]) -> list[ChunkExtraction | None]:
    """Map per-excerpt results onto chunks; None marks an excerpt the model skipped.

    The model says which excerpt a fact came from, but the evidence quote decides: a fact
    whose quote is only found in another excerpt of the batch is moved there, together
    with its two entity definitions.
    """
    per_chunk: list[ChunkExtraction | None] = [None] * len(chunks)
    texts = [_normalize(c["text"]) for c in chunks]
    moves: list[tuple[int, object, list[ExtractedEntity]]] = []

    for ex in batch.excerpts:
        i = ex.excerpt_id - 1
        if not 0 <= i < len(chunks):
            continue
        if per_chunk[i] is None:
            per_chunk[i] = ChunkExtraction(entities=[], relationships=[])
        per_chunk[i].entities.extend(ex.entities)
        for r in ex.relationships:
            evidence = _normalize(r.evidence)
            home = i if evidence in texts[i] else next(
                (j for j, t in enumerate(texts) if evidence in t), i)  # not found anywhere: validate rejects it
            if home == i:
                per_chunk[i].relationships.append(r)
            else:
                ents = [e for e in ex.entities
                        if (e.name, e.type) in {(r.source, r.source_type), (r.target, r.target_type)}]
                moves.append((home, r, ents))

    for home, r, ents in moves:
        if per_chunk[home] is None:
            per_chunk[home] = ChunkExtraction(entities=[], relationships=[])
        per_chunk[home].relationships.append(r)
        per_chunk[home].entities.extend(ents)
    return per_chunk


def _cache_path(chunk: dict):
    safe_id = chunk["chunk_id"].replace(":", "_")
    return (CACHE_DIR / "extraction" / llm.model_name() / PROMPT_VERSION
            / chunk["doc_hash"][:16] / f"{safe_id}.json")


def extract_batch(chunks: list[dict]) -> tuple[list[ChunkResult], llm.Usage]:
    """Raises llm.QuotaExhausted; every other failure is retried, then reported in .error."""
    usage = llm.Usage()
    last_error = None
    company = COMPANY_NAMES.get(chunks[0]["ticker"], chunks[0]["company"])
    message = user_message(company, chunks)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            parsed, call_usage = llm.generate_structured(SYSTEM_PROMPT, message, BatchExtraction)
        except llm.LLMError as e:
            last_error = str(e)
            continue
        usage.merge(call_usage)
        results = []
        for chunk, extraction in zip(chunks, split_batch(parsed, chunks)):
            if extraction is None:
                results.append(ChunkResult(chunk_id=chunk["chunk_id"], error="excerpt missing from response",
                                           attempts=attempt))
                continue
            result = validate(extraction, chunk["chunk_id"], chunk["text"])
            result.attempts = attempt
            results.append(result)
        return results, usage
    return [ChunkResult(chunk_id=c["chunk_id"], error=last_error, attempts=MAX_ATTEMPTS) for c in chunks], usage


def make_batches(chunks: list[dict]) -> list[list[dict]]:
    """Consecutive chunks of the same filing, BATCH_SIZE at a time."""
    batches: list[list[dict]] = []
    for c in chunks:
        if batches and len(batches[-1]) < BATCH_SIZE and batches[-1][0]["doc_id"] == c["doc_id"]:
            batches[-1].append(c)
        else:
            batches.append([c])
    return batches


def load_chunks(limit: int | None = None) -> list[dict]:
    with (PROCESSED_DIR / "chunks.jsonl").open(encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    return chunks[:limit] if limit else chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only the first N chunks (pilot run)")
    parser.add_argument("--contains", default=None, help="only chunks whose text contains this string")
    args = parser.parse_args()

    model = llm.model_name()
    chunks = load_chunks()
    if args.contains:
        chunks = [c for c in chunks if args.contains.lower() in c["text"].lower()]
    chunks = chunks[:args.limit] if args.limit else chunks
    todo = [c for c in chunks if not _cache_path(c).exists()]
    batches = make_batches(todo)
    print(f"{len(chunks)} chunks, {len(chunks) - len(todo)} cached, {len(todo)} to extract "
          f"in {len(batches)} requests with {model}")

    total = llm.Usage()
    done = failed = 0
    quota_hit = None
    with ThreadPoolExecutor(CONCURRENCY) as pool:
        futures = {pool.submit(extract_batch, b): b for b in batches}
        for fut in as_completed(futures):
            if fut.cancelled():  # dropped after the daily quota was hit
                continue
            batch = futures[fut]
            try:
                results, usage = fut.result()
            except llm.QuotaExhausted as e:
                if quota_hit is None:
                    quota_hit = str(e)
                    for f in futures:
                        f.cancel()  # drop queued batches; their chunks stay uncached for next run
                continue
            total.merge(usage)
            for chunk, result in zip(batch, results):
                if result.error:
                    failed += 1  # not cached, so the next run retries it
                    print(f"FAILED {result.chunk_id}: {result.error}")
                    continue
                done += 1
                path = _cache_path(chunk)
                path.parent.mkdir(parents=True, exist_ok=True)
                record = vars(result) | {"model": model, "prompt_version": PROMPT_VERSION,
                                         "batch": [c["chunk_id"] for c in batch]}
                path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"[{done}/{len(todo)}] {result.chunk_id}: {len(result.entities)} entities, "
                      f"{len(result.relationships)} rels, {len(result.rejected)} rejected")
            print(f"  request done: {usage.input_tokens} in / {usage.output_tokens} out tokens")

    print(f"\nExtracted {done}, failed {failed}, remaining {len(todo) - done - failed}.")
    print(f"Tokens: {vars(total)}")
    print(f"Paid-tier equivalent cost: ${total.cost(model):.4f}"
          + (f"  (${total.cost(model) / done:.5f}/chunk)" if done else ""))
    if done and (args.limit or args.contains):
        n_all = len(load_chunks())
        print(f"Projected for all {n_all} chunks: ~{-(-n_all // BATCH_SIZE)} requests, "
              f"~${total.cost(model) / done * n_all:.2f} at paid-tier prices")
    if quota_hit:
        print(f"\nSTOPPED: {quota_hit}. Run the same command again after the quota resets.")


if __name__ == "__main__":
    main()
