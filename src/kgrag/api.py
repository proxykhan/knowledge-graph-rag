"""FastAPI service: grounded answers with validated citations.

Run:
    uvicorn kgrag.api:app --port 8000
Then open http://localhost:8000/docs

Endpoints:
    POST /ask                 {"question": "..."} -> answer, claims with citations, route, validation info
    GET  /chunks/{chunk_id}   the source text a citation points to
    GET  /health              Neo4j and Postgres reachability
"""

from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from kgrag.answer.generate import answer
from kgrag.retrieval.retriever import _driver
from kgrag.vector.index import connect

app = FastAPI(title="Knowledge Graph RAG over SEC 10-K filings", version="0.4.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class CitationOut(BaseModel):
    source_id: str
    kind: str                 # "graph" (extracted fact) or "passage" (verbatim text)
    chunk_ids: list[str]      # resolvable with GET /chunks/{chunk_id}
    quote: str


class ClaimOut(BaseModel):
    text: str
    citations: list[CitationOut]


class AskResponse(BaseModel):
    question: str
    answerable: bool
    answer: str
    claims: list[ClaimOut]
    route_used: str           # GRAPH | VECTOR | BOTH
    attempts: int             # generation attempts; >1 means a draft failed citation checks
    dropped_claims: list[dict]
    rejections: list[list[str]]
    retrieval_notes: list[str]
    latency_ms: int


class ChunkOut(BaseModel):
    chunk_id: str
    doc_id: str
    ticker: str
    section_path: str
    report_date: str
    text: str


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    return AskResponse(**asdict(answer(req.question.strip())))


@app.get("/chunks/{chunk_id}", response_model=ChunkOut)
def get_chunk(chunk_id: str) -> ChunkOut:
    with connect() as conn:
        row = conn.execute(
            "SELECT chunk_id, doc_id, ticker, section_path, report_date::text, text FROM chunks WHERE chunk_id = %s",
            (chunk_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"unknown chunk_id {chunk_id!r}")
    return ChunkOut(**dict(zip(ChunkOut.model_fields, row)))


@app.get("/health")
def health() -> dict:
    status = {}
    try:
        with connect() as conn:
            status["postgres_chunks"] = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    except Exception as e:  # report, don't crash, so the endpoint stays useful when one store is down
        status["postgres_error"] = str(e)
    try:
        with _driver().session() as s:
            status["neo4j_entities"] = s.run("MATCH (e:Entity) RETURN count(e) AS n").single()["n"]
    except Exception as e:
        status["neo4j_error"] = str(e)
    status["ok"] = "postgres_error" not in status and "neo4j_error" not in status
    return status
