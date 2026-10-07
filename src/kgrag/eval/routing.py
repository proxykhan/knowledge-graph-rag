"""Measure the router on labeled questions: route accuracy, query-type accuracy, and how often
a GRAPH decision produced graph rows (vs. falling back to vectors).

Usage:
    python -m kgrag.eval.routing

Runs the full retriever, so every decision is also written to data/logs/routing.jsonl.
"""

import json
from collections import Counter

from kgrag.config import PROJECT_ROOT
from kgrag.retrieval.retriever import retrieve

QUESTIONS = PROJECT_ROOT / "eval" / "routing_questions.jsonl"
RESULTS = PROJECT_ROOT / "eval" / "results" / "routing.json"


def main() -> None:
    questions = [json.loads(line) for line in QUESTIONS.open(encoding="utf-8")]
    rows = []
    for q in questions:
        r = retrieve(q["question"])
        d = r.decision or {}
        rows.append({
            "id": q["id"], "question": q["question"],
            "expected_route": q["route"], "route": d.get("route"),
            "expected_query_type": q["query_type"], "query_type": d.get("query_type"),
            "confidence": d.get("confidence"), "route_used": r.route_used,
            "graph_rows": len(r.graph_rows), "notes": r.notes, "unlinked": r.unlinked,
        })
        mark = "ok " if d.get("route") == q["route"] and d.get("query_type") == q["query_type"] else "XX "
        print(f"{mark}{q['id']} {d.get('route')}/{d.get('query_type')} conf={d.get('confidence')} "
              f"used={r.route_used} rows={len(r.graph_rows)} {'; '.join(r.notes)}")

    n = len(rows)
    route_ok = sum(r["route"] == r["expected_route"] for r in rows)
    type_ok = sum(r["route"] == r["expected_route"] and r["query_type"] == r["expected_query_type"] for r in rows)
    graph = [r for r in rows if r["expected_route"] == "GRAPH"]
    answered = sum(r["graph_rows"] > 0 for r in graph)
    summary = {
        "n": n,
        "route_accuracy": route_ok / n,
        "route_and_query_type_accuracy": type_ok / n,
        "graph_questions_with_rows": f"{answered}/{len(graph)}",
        "route_used": dict(Counter(r["route_used"] for r in rows)),
    }
    print("\n" + json.dumps(summary, indent=2))
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
