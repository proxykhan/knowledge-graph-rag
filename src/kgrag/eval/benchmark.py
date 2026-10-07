"""Phase 5: benchmark hybrid graph+vector RAG against plain vector RAG, by question type.

Usage:
    python -m kgrag.eval.benchmark            # run (resumes; already-answered questions are skipped)
    python -m kgrag.eval.benchmark --report   # only rebuild the report from saved runs

Both systems use the same answer generator and the same citation checks; only retrieval
differs. Scoring is deterministic string matching, no LLM judge:
  - answerable question: correct when the answer is not a refusal and mentions every
    required item (any accepted spelling); item recall gives partial credit
  - out-of-scope question: correct when the system refuses
Runs append to eval/results/benchmark_runs.jsonl so a quota stop loses nothing.
"""

import argparse
import json
import re
import statistics
import unicodedata
from collections import defaultdict
from dataclasses import asdict

from kgrag import llm
from kgrag.config import PROJECT_ROOT
from kgrag.extract.prompt import PROMPT_VERSION
from kgrag.retrieval.router import ROUTER_VERSION

QUESTIONS = PROJECT_ROOT / "eval" / "benchmark_questions.jsonl"
RUNS = PROJECT_ROOT / "eval" / "results" / "benchmark_runs.jsonl"
REPORT = PROJECT_ROOT / "eval" / "results" / "benchmark.md"
SYSTEMS = ("vector", "hybrid")
CATEGORIES = ("single_hop", "two_hop", "three_hop", "aggregation", "out_of_scope")
RUN_VERSION = f"router-{ROUTER_VERSION}/extract-{PROMPT_VERSION}"


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("’", "'").lower()
    return re.sub(r"\s+", " ", text)


def mentions(answer: str, alias: str) -> bool:
    a, t = _norm(alias), _norm(answer)
    if len(a) <= 3 or a.isdigit():  # "3", "80", "UMC": whole word only, so "3" does not match "2023"
        return re.search(rf"(?<![\w.]){re.escape(a)}(?![\w])", t) is not None
    return a in t


def score(question: dict, answerable: bool, answer_text: str) -> dict:
    if question["category"] == "out_of_scope":
        return {"correct": not answerable, "item_recall": None}
    found = [any(mentions(answer_text, alias) for alias in item) for item in question["required"]]
    recall = sum(found) / len(found)
    return {"correct": answerable and all(found), "item_recall": recall if answerable else 0.0}


def load_runs() -> dict[tuple[str, str], dict]:
    if not RUNS.exists():
        return {}
    runs = {}
    for line in RUNS.open(encoding="utf-8"):
        r = json.loads(line)
        if r["run_version"] == RUN_VERSION:
            runs[(r["system"], r["id"])] = r
    return runs


def run(questions: list[dict]) -> None:
    from kgrag.answer.generate import answer  # heavy imports only when actually running

    done = load_runs()
    todo = [(s, q) for q in questions for s in SYSTEMS if (s, q["id"]) not in done]
    print(f"{len(questions)} questions x {len(SYSTEMS)} systems: {len(done)} done, {len(todo)} to run")
    RUNS.parent.mkdir(parents=True, exist_ok=True)
    for i, (system, q) in enumerate(todo, 1):
        try:
            a = answer(q["question"], mode=system)
        except llm.QuotaExhausted as e:
            print(f"\nSTOPPED at {i - 1}/{len(todo)}: {e}. Re-run later to continue.")
            return
        s = score(q, a.answerable, a.answer)
        record = {"run_version": RUN_VERSION, "system": system, "id": q["id"], "category": q["category"],
                  **s, "answerable": a.answerable, "answer": a.answer, "route_used": a.route_used,
                  "attempts": a.attempts, "dropped_claims": len(a.dropped_claims),
                  "latency_ms": a.latency_ms, "tokens_in": a.tokens_in, "tokens_out": a.tokens_out,
                  "n_citations": sum(len(c["citations"]) for c in a.claims)}
        with RUNS.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[{i}/{len(todo)}] {system:6} {q['id']} {'OK ' if s['correct'] else 'XX '} "
              f"route={a.route_used:6} {a.latency_ms} ms  {a.answer[:90]!r}")


def _p(values: list[float], pct: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, round(pct / 100 * (len(values) - 1)))]


def report(questions: list[dict]) -> str:
    runs = load_runs()
    qs = {q["id"]: q for q in questions}
    by = defaultdict(list)
    for (system, qid), r in runs.items():
        # Re-score from the saved answer so a corrected answer key applies to every system alike.
        r.update(score(qs[qid], r["answerable"], r["answer"]))
        by[(system, r["category"])].append(r)
        by[(system, "all")].append(r)
    model = llm.model_name()

    lines = [f"# Benchmark: hybrid graph+vector RAG vs. vector-only RAG", "",
             f"{len(questions)} questions, model `{model}`, run version `{RUN_VERSION}`. "
             "Accuracy = all required items mentioned (refusal for out-of-scope). "
             "Item recall = share of required items mentioned (answerable questions only).", "",
             "| Question type | n | Vector RAG accuracy | Hybrid accuracy | Vector item recall | Hybrid item recall |",
             "|---|---|---|---|---|---|"]
    for cat in CATEGORIES + ("all",):
        v, h = by[("vector", cat)], by[("hybrid", cat)]
        if not v and not h:
            continue

        def acc(rs):
            return f"{sum(r['correct'] for r in rs)}/{len(rs)} ({sum(r['correct'] for r in rs) / len(rs):.0%})" if rs else "-"

        def rec(rs):
            vals = [r["item_recall"] for r in rs if r["item_recall"] is not None]
            return f"{statistics.mean(vals):.2f}" if vals else "-"

        label = cat.replace("_", " ")
        lines.append(f"| {'**all**' if cat == 'all' else label} | {max(len(v), len(h))} | {acc(v)} | {acc(h)} | {rec(v)} | {rec(h)} |")

    lines += ["", "| Cost and latency per query | Vector RAG | Hybrid |", "|---|---|---|"]
    stats = {}
    for system in SYSTEMS:
        rs = by[(system, "all")]
        if not rs:
            continue
        lat = [r["latency_ms"] for r in rs]
        usage = llm.Usage(sum(r["tokens_in"] for r in rs), sum(r["tokens_out"] for r in rs))
        stats[system] = {"p50": _p(lat, 50), "p95": _p(lat, 95),
                         "tokens": (usage.input_tokens + usage.output_tokens) / len(rs),
                         "cost": usage.cost(model) / len(rs),
                         "regen": sum(r["attempts"] > 1 for r in rs)}
    if stats:
        def row(label, fmt):
            return f"| {label} | " + " | ".join(fmt(stats[s]) if s in stats else "-" for s in SYSTEMS) + " |"
        lines += [row("latency p50", lambda s: f"{s['p50'] / 1000:.1f} s"),
                  row("latency p95", lambda s: f"{s['p95'] / 1000:.1f} s"),
                  row("LLM tokens per query", lambda s: f"{s['tokens']:,.0f}"),
                  row("cost per query (paid-tier equivalent)", lambda s: f"${s['cost']:.5f}"),
                  row("answers needing regeneration", lambda s: str(s["regen"]))]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true", help="rebuild the report without running")
    args = parser.parse_args()
    questions = [json.loads(line) for line in QUESTIONS.open(encoding="utf-8")]
    if not args.report:
        run(questions)
    text = report(questions)
    REPORT.write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
