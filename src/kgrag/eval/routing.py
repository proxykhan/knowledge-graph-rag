"""Measure the router on labeled questions: route accuracy, query-type accuracy, hop accuracy
(relation + direction, where the question file gives expected hops), and how often a GRAPH
decision produced graph rows (vs. falling back to vectors).

Usage:
    python -m kgrag.eval.routing              # eval/routing_questions.jsonl (24 questions)
    python -m kgrag.eval.routing --set dev    # eval/routing_dev_questions.jsonl (direction-focused)

These are development sets: router changes are tuned against them, never against the
benchmark questions. Runs the full retriever, so every decision is also logged.
"""

import argparse
import json
from collections import Counter

from kgrag.config import PROJECT_ROOT
from kgrag.retrieval.retriever import retrieve
from kgrag.retrieval.router import ROUTER_VERSION

SETS = {"main": "routing_questions.jsonl", "dev": "routing_dev_questions.jsonl"}
UNDIRECTED = {"COMPETES_WITH", "PARTNERS_WITH"}


def hops_match(expected: list[str], got: list[dict]) -> bool:
    if len(expected) != len(got):
        return False
    for want, h in zip(expected, got):
        rel, direction = want.split("/")
        if h["relation"] != rel:
            return False
        if rel not in UNDIRECTED and h["direction"] != direction:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=SETS, default="main")
    args = parser.parse_args()
    questions = [json.loads(line) for line in (PROJECT_ROOT / "eval" / SETS[args.set]).open(encoding="utf-8")]

    rows = []
    for q in questions:
        r = retrieve(q["question"])
        d = r.decision or {}
        hop_ok = hops_match(q["hops"], d.get("hops", [])) if "hops" in q else None
        rows.append({
            "id": q["id"], "question": q["question"],
            "expected_route": q["route"], "route": d.get("route"),
            "expected_query_type": q["query_type"], "query_type": d.get("query_type"),
            "expected_hops": q.get("hops"), "hops": [f"{h['relation']}/{h['direction']}" for h in d.get("hops", [])],
            "hops_ok": hop_ok, "confidence": d.get("confidence"), "route_used": r.route_used,
            "graph_rows": len(r.graph_rows), "direction_reversed": r.direction_reversed,
            "notes": r.notes, "unlinked": r.unlinked,
        })
        ok = d.get("route") == q["route"] and d.get("query_type") == q["query_type"] and hop_ok is not False
        print(f"{'ok ' if ok else 'XX '}{q['id']} {d.get('route')}/{d.get('query_type')} "
              f"hops={rows[-1]['hops']}{'' if hop_ok is None else (' (hops ok)' if hop_ok else ' (HOPS WRONG)')} "
              f"rows={len(r.graph_rows)} {'; '.join(r.notes)}")

    n = len(rows)
    graph = [r for r in rows if r["expected_route"] == "GRAPH"]
    with_hops = [r for r in rows if r["hops_ok"] is not None]
    summary = {
        "set": args.set, "router_version": ROUTER_VERSION, "n": n,
        "route_accuracy": sum(r["route"] == r["expected_route"] for r in rows) / n,
        "route_and_query_type_accuracy": sum(r["route"] == r["expected_route"] and
                                             r["query_type"] == r["expected_query_type"] for r in rows) / n,
        "hops_correct": f"{sum(r['hops_ok'] for r in with_hops)}/{len(with_hops)}" if with_hops else None,
        "graph_questions_with_rows": f"{sum(r['graph_rows'] > 0 for r in graph)}/{len(graph)}",
        "rescued_by_direction_retry": sum(r["direction_reversed"] for r in rows),
        "route_used": dict(Counter(r["route_used"] for r in rows)),
    }
    print("\n" + json.dumps(summary, indent=2))
    out = PROJECT_ROOT / "eval" / "results" / f"routing_{args.set}_{ROUTER_VERSION}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
