"""Vector search: find the chunks whose embeddings are closest to the question's."""

import time
from collections.abc import Callable

import numpy as np
import psycopg

from reportlens.core.config import get_settings
from reportlens.core.logging import get_logger
from reportlens.ingest.embed import embed_query
from reportlens.retrieval.filters import QueryFilters
from reportlens.retrieval.models import RetrievedChunk

log = get_logger(__name__)

QueryEmbedder = Callable[[str], np.ndarray]


def search_sql(filters: QueryFilters | None = None) -> str:
    """The search query, with a WHERE clause for any filters.

    <=> is pgvector's cosine distance (0 = same direction, 2 = opposite), so 1 - distance is
    cosine similarity. Ordering by the bare distance expression lets Postgres use the HNSW index.
    """
    conditions = []
    if filters and filters.companies:
        conditions.append("company = ANY(%(companies)s)")
    if filters and filters.years:
        conditions.append("year = ANY(%(years)s)")
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    return f"""
        SELECT chunk_id, company, year, section, pages, text, 1 - (embedding <=> %(q)s) AS score
        FROM chunks
        {where}
        ORDER BY embedding <=> %(q)s
        LIMIT %(k)s
    """


def _run(conn: psycopg.Connection, vector: np.ndarray, k: int, filters: QueryFilters | None):
    params = {"q": vector, "k": k}
    if filters:
        params |= {"companies": filters.companies, "years": filters.years}
    with conn.transaction():
        # The HNSW index finds nearest neighbours first and filters after, so a strict WHERE
        # could leave fewer than k rows. Iterative scan keeps searching until k rows pass.
        conn.execute("SET LOCAL hnsw.iterative_scan = strict_order")
        return conn.execute(search_sql(filters), params).fetchall()


def vector_search(
    conn: psycopg.Connection,
    question: str,
    k: int | None = None,
    embed: QueryEmbedder = embed_query,
    filters: QueryFilters | None = None,
) -> list[RetrievedChunk]:
    """Return the k chunks most similar to the question, best first.

    With filters, only matching reports are searched; if nothing matches, falls back to an
    unfiltered search so a bad filter is never worse than no filter.
    """
    k = k or get_settings().retrieval_top_k
    filters = None if filters is None or filters.is_empty() else filters
    start = time.perf_counter()
    vector = embed(question)
    rows = _run(conn, vector, k, filters)
    if filters and not rows:
        log.info(
            "retrieval.filter_fallback companies=%s years=%s", filters.companies, filters.years
        )
        rows = _run(conn, vector, k, None)

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
        "retrieval.vector k=%d filtered=%s results=%d top_score=%.3f latency=%.3fs",
        k,
        filters is not None,
        len(results),
        results[0].score if results else 0.0,
        time.perf_counter() - start,
    )
    return results
