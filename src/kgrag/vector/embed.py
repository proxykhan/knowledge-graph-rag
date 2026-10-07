"""Local embeddings with sentence-transformers (no API, no cost).

BGE v1.5 models are trained with an instruction prefix on short queries and none on
passages; using it measurably improves query -> passage retrieval.
"""

from functools import lru_cache

import numpy as np

from kgrag.config import settings

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embedding_model)


def dim() -> int:
    return model().get_sentence_embedding_dimension()


def embed_passages(texts: list[str]) -> np.ndarray:
    return model().encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)


def embed_query(query: str) -> np.ndarray:
    return model().encode(QUERY_INSTRUCTION + query, normalize_embeddings=True)
