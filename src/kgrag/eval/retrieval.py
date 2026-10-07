"""Phase 2 gate: measure retrieval recall@k on a labeled question set before building the router.

Usage:
    python -m kgrag.eval.retrieval

Labels: each question in eval/retrieval_questions.jsonl names a company and phrases quoted
verbatim from its filing; every chunk of that filing containing any phrase counts as relevant
(several phrases when the filing states the same fact in more than one place).
Reports, for several HNSW ef_search values and for exact search:
  - recall@k: share of questions with at least one relevant chunk in the top k
  - MRR: mean reciprocal rank of the first relevant chunk (within top 10)
  - ANN overlap@10: share of the exact top 10 that the HNSW index also returns
"""

import json
import time

from kgrag.config import PROCESSED_DIR, PROJECT_ROOT
from kgrag.extract.extractor import _normalize
from kgrag.vector import embed
from kgrag.vector.index import connect
from kgrag.vector.search import search

QUESTIONS = PROJECT_ROOT / "eval" / "retrieval_questions.jsonl"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
KS = (1, 3, 5, 10)
EF_SEARCH_VALUES = (10, 20, 40, 100)


def load_labeled() -> list[dict]:
    with (PROCESSED_DIR / "chunks.jsonl").open(encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    norm = [(c["chunk_id"], c["ticker"], _normalize(c["text"])) for c in chunks]
    questions = [json.loads(line) for line in QUESTIONS.open(encoding="utf-8")]
    for q in questions:
        phrases = [_normalize(p) for p in q["phrases"]]
        q["relevant"] = {cid for cid, t, text in norm if t == q["ticker"] and any(p in text for p in phrases)}
        if not q["relevant"]:
            raise SystemExit(f"{q['id']}: no phrase found in any {q['ticker']} chunk; fix the label")
    return questions


def evaluate(conn, questions: list[dict], ef_search: int, exact: bool) -> tuple[dict, dict[str, list[str]]]:
    hits = {k: 0 for k in KS}
    rr_sum = 0.0
    latencies = []
    top10 = {}
    for q in questions:
        start = time.perf_counter()
        results = search(conn, q["vec"], k=max(KS), ef_search=ef_search, exact=exact)
        latencies.append((time.perf_counter() - start) * 1000)
        ids = [p.chunk_id for p in results]
        top10[q["id"]] = ids
        first = next((i for i, cid in enumerate(ids) if cid in q["relevant"]), None)
        for k in KS:
            hits[k] += first is not None and first < k
        rr_sum += 1 / (first + 1) if first is not None else 0
    n = len(questions)
    latencies.sort()
    metrics = {f"recall@{k}": hits[k] / n for k in KS}
    metrics["mrr@10"] = rr_sum / n
    metrics["p50_ms"] = latencies[n // 2]
    return metrics, top10


def main() -> None:
    questions = load_labeled()
    for q in questions:
        q["vec"] = embed.embed_query(q["question"])

    with connect() as conn:
        header = conn.execute("SELECT bool_and(embedded_with_header) FROM chunks").fetchone()[0]
        exact_metrics, exact_top = evaluate(conn, questions, ef_search=40, exact=True)
        rows = [("exact", exact_metrics, 1.0)]
        for ef in EF_SEARCH_VALUES:
            metrics, top = evaluate(conn, questions, ef_search=ef, exact=False)
            overlap = sum(len(set(top[i]) & set(exact_top[i])) for i in top) / (10 * len(top))
            rows.append((f"hnsw ef={ef}", metrics, overlap))

        misses = [q for q in questions if not set(exact_top[q["id"]][:5]) & q["relevant"]]

    variant = "header" if header else "no_header"
    print(f"\nRetrieval on {len(questions)} labeled questions (chunks embedded with "
          f"{'company/section header' if header else 'raw text only'})\n")
    cols = [f"recall@{k}" for k in KS] + ["mrr@10"]
    print("| search | " + " | ".join(cols) + " | ANN overlap@10 | p50 ms |")
    print("|---" * (len(cols) + 3) + "|")
    for name, m, overlap in rows:
        print(f"| {name} | " + " | ".join(f"{m[c]:.2f}" for c in cols) + f" | {overlap:.2f} | {m['p50_ms']:.1f} |")
    if misses:
        print(f"\nMissed at k=5 (exact): " + ", ".join(f"{q['id']} ({q['question'][:60]}...)" for q in misses))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"retrieval_{variant}.json"
    out.write_text(json.dumps({"variant": variant, "n_questions": len(questions),
                               "rows": [{"search": n, **m, "ann_overlap@10": o} for n, m, o in rows],
                               "missed_at_5": [q["id"] for q in misses]}, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
