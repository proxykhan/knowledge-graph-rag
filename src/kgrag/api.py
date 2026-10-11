"""FastAPI service: grounded answers with validated citations, plus a web interface.

Run:
    uvicorn kgrag.api:app --port 8000
Then open http://localhost:8000 (web interface) or http://localhost:8000/docs (API)

Endpoints:
    GET  /                    the web interface (src/kgrag/web/index.html)
    POST /ask                 {"question": "..."} -> answer, claims with citations, sources, route
    GET  /chunks/{chunk_id}   the source text a citation points to
    GET  /graph/entity        ?name=NVIDIA -> that entity's connections in the graph
    GET  /entities            entity names for search suggestions
    GET  /health              Neo4j and Postgres reachability
"""

from dataclasses import asdict

import json

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from kgrag import llm
from kgrag.answer.generate import answer
from kgrag.config import PROCESSED_DIR, PROJECT_ROOT
from kgrag.retrieval import templates
from kgrag.retrieval.entity_linker import default_linker
from kgrag.retrieval.retriever import _driver
from kgrag.vector.index import connect

WEB_DIR = PROJECT_ROOT / "src" / "kgrag" / "web"

app = FastAPI(title="Knowledge Graph RAG over SEC 10-K filings", version="0.5.0")
app.mount("/assets", StaticFiles(directory=PROJECT_ROOT / "docs"), name="assets")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class CitationOut(BaseModel):
    source_id: str
    kind: str                 # "graph" (extracted fact) or "passage" (verbatim text)
    chunk_ids: list[str]      # resolvable with GET /chunks/{chunk_id}
    quote: str


class EdgeOut(BaseModel):
    source: str
    source_type: str | None = None
    relation: str
    target: str
    target_type: str | None = None


class SourceOut(BaseModel):
    id: str
    kind: str                 # "graph" | "passage"
    text: str
    chunk_ids: list[str]
    evidence: list[str]
    edge: EdgeOut | None
    cited: bool               # used by the answer, vs. only retrieved


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
    tokens_in: int
    tokens_out: int
    mode: str
    sources: list[SourceOut]


class ChunkOut(BaseModel):
    chunk_id: str
    doc_id: str
    ticker: str
    section_path: str
    report_date: str
    text: str


@app.get("/", include_in_schema=False)
def web_interface() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        return AskResponse(**asdict(answer(req.question.strip())))
    except llm.QuotaExhausted as e:
        raise HTTPException(status_code=503, detail=f"LLM quota exhausted, try again later: {e}")


@app.get("/chunks/{chunk_id}", response_model=ChunkOut)
def get_chunk(chunk_id: str) -> ChunkOut:
    with connect() as conn:
        row = conn.execute(
            "SELECT chunk_id, doc_id, ticker, section_path, report_date::text, text FROM chunks WHERE chunk_id = %s",
            (chunk_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"unknown chunk_id {chunk_id!r}")
    return ChunkOut(**dict(zip(ChunkOut.model_fields, row)))


class GraphEdgeOut(EdgeOut):
    chunk_ids: list[str]
    evidence: list[str]


class EntityGraphOut(BaseModel):
    entity: dict
    edges: list[GraphEdgeOut]


@app.get("/graph/entity", response_model=EntityGraphOut)
def entity_graph(name: str = Query(min_length=2, max_length=200)) -> EntityGraphOut:
    link = default_linker().link(name)
    if link is None:
        raise HTTPException(status_code=404, detail=f"No entity in the graph matches {name!r}")
    rows = templates.run(_driver(), *templates.entity_profile(link.entity_id, limit=300))
    edges = [e for row in rows for e in row["edges"]]
    return EntityGraphOut(entity={"id": link.entity_id, "name": link.name, "type": link.type}, edges=edges)


@app.get("/entities")
def entities() -> list[dict]:
    rows = json.loads((PROCESSED_DIR / "entities.json").read_text(encoding="utf-8"))
    rows.sort(key=lambda e: (e["type"] != "Company", -e["mentions"]))
    return [{"name": e["name"], "type": e["type"]} for e in rows]


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
