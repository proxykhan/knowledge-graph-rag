import json

from kgrag.config import PROJECT_ROOT
from kgrag.eval.benchmark import CATEGORIES, mentions, score

Q = {"category": "two_hop", "required": [["TSMC", "Taiwan Semiconductor"], ["Samsung"]]}


def test_any_alias_counts_and_all_items_are_required():
    assert score(Q, True, "Taiwan Semiconductor and Samsung supply both.")["correct"]
    partial = score(Q, True, "TSMC supplies both.")
    assert not partial["correct"] and partial["item_recall"] == 0.5


def test_refusal_scores_zero_on_answerable_and_full_on_out_of_scope():
    assert score(Q, False, "TSMC and Samsung")["item_recall"] == 0.0
    assert score({"category": "out_of_scope", "required": []}, False, "refused")["correct"]
    assert not score({"category": "out_of_scope", "required": []}, True, "Paris")["correct"]


def test_short_and_numeric_aliases_match_whole_words_only():
    assert mentions("NVIDIA names three memory suppliers (3).", "3")
    assert not mentions("In fiscal 2023 revenue rose.", "3")
    assert mentions("It uses UMC wafers.", "UMC") and not mentions("drumcircle", "UMC")
    assert mentions("first built on Intel's 1γ node", "1γ")


def test_benchmark_file_is_well_formed():
    rows = [json.loads(l) for l in (PROJECT_ROOT / "eval" / "benchmark_questions.jsonl").open(encoding="utf-8")]
    assert len(rows) >= 50 and len({r["id"] for r in rows}) == len(rows)
    assert {r["category"] for r in rows} == set(CATEGORIES)
    for r in rows:
        assert (r["category"] == "out_of_scope") == (r["required"] == [])
