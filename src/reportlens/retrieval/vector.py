"""Vector search: find the chunks whose embeddings are closest to the question's."""

import time
from collections.abc import Callable

import numpy as np
import psycopg

from reportlens.core.config import get_settings
from reportlens.core.logging import get_logger
from reportlens.ingest.embed import embed_query
from reportlens.retrieval.models import RetrievedChunk

log = get_logger(__name__)

QueryEmbedder = Callable[[str], np.ndarray]

# <=> is pgvector's cosine distance (0 = same direction, 2 = opposite), so 1 - distance is
# cosine similarity. Ordering by the bare distance expression lets Postgres use the HNSW index.
_SEARCH_SQL = """
    SELECT chunk_id, company, year, section, pages, text, 1 - (embedding <=> %(q)s) AS score
    FROM chunks
    ORDER BY embedding <=> %(q)s
    LIMIT %(k)s
"""


def vector_search(
    conn: psycopg.Connection,
    question: str,
    k: int | None = None,
    embed: QueryEmbedder = embed_query,
) -> list[RetrievedChunk]:
    """Return the k chunks most similar to the question, best first."""
    k = k or get_settings().retrieval_top_k
    start = time.perf_counter()
    rows = conn.execute(_SEARCH_SQL, {"q": embed(question), "k": k}).fetchall()
    results = [
        RetrievedChunk(
            chunk_id=chunk_id,
            company=company,
            year=year,
            section=section,
            pages=pages,
            text=text,
            score=float(score),
        )
        for chunk_id, company, year, section, pages, text, score in rows
    ]
    log.info(
        "retrieval.vector k=%d results=%d top_score=%.3f latency=%.3fs",
        k,
        len(results),
        results[0].score if results else 0.0,
        time.perf_counter() - start,
    )
    return results
