import json

from kgrag.config import PROJECT_ROOT
from kgrag.vector.index import passage_text

CHUNK = {"ticker": "NVDA", "company": "NVIDIA CORP", "section_path": "10-K > Item 1 Business",
         "text": "We purchase memory from SK Hynix Inc."}


def test_header_names_the_company_the_chunk_calls_we():
    assert passage_text(CHUNK, header=True).startswith("NVIDIA Corporation 10-K > Item 1 Business\n")
    assert passage_text(CHUNK, header=False) == CHUNK["text"]


def test_question_file_is_well_formed():
    rows = [json.loads(line) for line in (PROJECT_ROOT / "eval" / "retrieval_questions.jsonl").open(encoding="utf-8")]
    assert len(rows) == 30
    assert len({r["id"] for r in rows}) == 30
    assert all(r["phrases"] and r["ticker"] and r["question"] for r in rows)
