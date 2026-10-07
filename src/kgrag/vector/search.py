"""Vector search over the pgvector chunks table."""

from dataclasses import dataclass

import numpy as np
import psycopg


@dataclass
class Passage:
    chunk_id: str
    ticker: str
    section_path: str
    text: str
    entity_ids: list[str]
    score: float  # cosine similarity, higher is closer


def search(conn: psycopg.Connection, query_vec: np.ndarray, k: int = 5,
           ef_search: int = 40, exact: bool = False) -> list[Passage]:
    """Top-k chunks by cosine similarity.

    ef_search: HNSW query-time candidate list size; higher = better recall, slower.
    exact=True disables index scans, giving true nearest neighbours to measure the index against.
    """
    with conn.transaction():
        # SET LOCAL lasts only for this transaction, so concurrent callers are unaffected.
        conn.execute(f"SET LOCAL hnsw.ef_search = {int(ef_search)}")
        if exact:
            conn.execute("SET LOCAL enable_indexscan = off")
        else:
            # On a small table the planner may prefer a sequential scan, which would make
            # ef_search meaningless; force the HNSW index so its behaviour is what we measure.
            conn.execute("SET LOCAL enable_seqscan = off")
        rows = conn.execute("""
            SELECT chunk_id, ticker, section_path, text, entity_ids, 1 - (embedding <=> %s) AS score
            FROM chunks
            ORDER BY embedding <=> %s
            LIMIT %s""", (query_vec, query_vec, k)).fetchall()
    return [Passage(*row) for row in rows]
