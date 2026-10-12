from kgrag.answer.context import Context, Source, build_context, statement
from kgrag.answer.generate import AnswerDraft, check
from kgrag.retrieval.retriever import RetrievalResult
from kgrag.vector.search import Passage

EDGE = {"source": "TSMC", "relation": "SUPPLIES", "target": "NVIDIA", "chunk_ids": ["N:1:0011"],
        "evidence": ["We utilize foundries, such as TSMC."], "confidence": 0.95}

CTX = Context([
    Source("S1", "graph", "TSMC supplies NVIDIA.", ["N:1:0011"], ["We utilize foundries, such as TSMC."]),
    Source("S2", "passage", "We incurred approximately $800 million in inventory charges.", ["A:1A:0033"]),
], [])


def _draft(claims, answerable=True):
    return AnswerDraft.model_validate({"answerable": answerable, "claims": claims})


def test_graph_edges_become_sentences_and_are_deduplicated():
    result = RetrievalResult("q", "GRAPH", None, graph_rows=[{"edges": [EDGE]}, {"edges": [EDGE, EDGE]}],
                             passages=[Passage("A:1A:0033", "AMD", "s", "text", [], 0.8)])
    ctx = build_context(result)
    assert [s.text for s in ctx.sources] == ["TSMC supplies NVIDIA.", "text"]
    assert statement({**EDGE, "relation": "SERVES_MARKET", "target": "Data Center"}) == \
        "TSMC serves the Data Center market."
    assert "## Graph facts" in ctx.render() and "## Retrieved passages" in ctx.render()


def test_valid_claims_pass():
    claims, problems = check(_draft([{"text": "AMD took about $800 million of charges.", "citations": ["S2"]},
                                     {"text": "TSMC supplies NVIDIA.", "citations": ["[S1]"]}]), CTX)
    assert problems == [] and len(claims) == 2


def test_invented_source_is_rejected():
    claims, problems = check(_draft([{"text": "TSMC supplies NVIDIA.", "citations": ["S9"]}]), CTX)
    assert claims == [] and "not in the context" in problems[0]


def test_uncited_claim_is_rejected():
    _, problems = check(_draft([{"text": "TSMC supplies NVIDIA.", "citations": []}]), CTX)
    assert "no citation" in problems[0]


def test_number_not_in_cited_source_is_rejected():
    claims, problems = check(_draft([{"text": "AMD took $440 million of charges.", "citations": ["S2"]}]), CTX)
    assert claims == [] and "['440']" in problems[0]


def test_refusal_with_no_claims_is_valid():
    claims, problems = check(_draft([], answerable=False), CTX)
    assert claims == [] and problems == []


def test_api_ask_shape(monkeypatch):
    from fastapi.testclient import TestClient

    from kgrag import api
    from kgrag.answer.generate import Answer

    monkeypatch.setattr(api, "answer", lambda q, **kw: Answer(q, True, "TSMC supplies NVIDIA. [S1]", [
        {"text": "TSMC supplies NVIDIA.", "citations": [
            {"source_id": "S1", "kind": "graph", "chunk_ids": ["N:1:0011"], "quote": "We utilize foundries"}]}],
        "GRAPH", 1))
    client = TestClient(api.app)
    r = client.post("/ask", json={"question": "Who supplies NVIDIA?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answerable"] and body["claims"][0]["citations"][0]["chunk_ids"] == ["N:1:0011"]
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_web_interface_is_served():
    from fastapi.testclient import TestClient

    from kgrag import api

    r = TestClient(api.app).get("/")
    assert r.status_code == 200 and "Filing Explorer" in r.text


def test_answer_cache_returns_stored_answer(monkeypatch, tmp_path):
    from kgrag.answer import generate

    calls = []
    monkeypatch.setattr(generate, "ANSWER_CACHE", tmp_path)
    monkeypatch.setattr(generate, "_answer", lambda q, mode="hybrid": calls.append(q) or generate.Answer(
        q, True, "TSMC supplies NVIDIA. [S1]", [], "GRAPH", 1, latency_ms=9000))
    first = generate.answer("Who supplies NVIDIA?", use_cache=True)
    second = generate.answer("  who supplies   NVIDIA? ", use_cache=True)   # same question, other spacing/case
    assert calls == ["Who supplies NVIDIA?"] and not first.cached and second.cached
    assert second.answer == first.answer and second.latency_ms < 9000
    generate.answer("Who supplies NVIDIA?")                                # benchmarks bypass the cache
    assert len(calls) == 2


def test_rate_limiter_blocks_after_the_limit_and_zero_disables():
    from kgrag.api import RateLimiter

    limited = RateLimiter(per_minute=2, per_day=0)
    assert [limited.allow("1.2.3.4") for _ in range(3)] == [True, True, False]
    assert limited.allow("5.6.7.8")                      # other visitors unaffected
    assert all(RateLimiter(0, 0).allow("x") for _ in range(50))
