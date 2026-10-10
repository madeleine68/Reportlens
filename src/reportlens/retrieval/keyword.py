"""Keyword search with Postgres full-text search: finds chunks sharing the question's words.

Catches exact terms vector search can blur ("CET1", "attributable to equity holders"). Ranking
uses ts_rank_cd (more matching terms, closer together, rank higher). This is BM25-like, not true
BM25: it has no corpus-wide term weighting.
"""

import time

import psycopg

from reportlens.core.config import get_settings
from reportlens.core.logging import get_logger
from reportlens.retrieval.filters import QueryFilters, filter_conditions, where_sql
from reportlens.retrieval.models import RetrievedChunk

log = get_logger(__name__)

# plainto_tsquery stems the question and drops stopwords, but joins terms with AND, so every
# word would have to appear. Swapping & for | makes any term match; ranking rewards more of them.
_QUERY_CTE = """
    WITH query AS (
        SELECT replace(plainto_tsquery('english', %(question)s)::text, '&', '|')::tsquery AS q
    )
"""


def _run(conn: psycopg.Connection, question: str, k: int, filters: QueryFilters | None):
    conditions, params = filter_conditions(filters)
    sql = f"""{_QUERY_CTE}
        SELECT chunk_id, company, year, section, pages, text, context,
               ts_rank_cd(tsv, query.q) AS score
        FROM chunks, query
        {where_sql(["tsv @@ query.q", *conditions])}
        ORDER BY score DESC, chunk_id
        LIMIT %(k)s
    """
    return conn.execute(sql, {"question": question, "k": k, **params}).fetchall()


def keyword_search(
    conn: psycopg.Connection,
    question: str,
    k: int | None = None,
    filters: QueryFilters | None = None,
) -> list[RetrievedChunk]:
    """Return the k chunks whose text best matches the question's terms, best first.

    Filters work as in vector search, including the unfiltered fallback when nothing matches.
    """
    k = k or get_settings().retrieval_top_k
    filters = None if filters is None or filters.is_empty() else filters
    start = time.perf_counter()
    rows = _run(conn, question, k, filters)
    if filters and not rows:
        log.info(
            "retrieval.filter_fallback companies=%s years=%s", filters.companies, filters.years
        )
        rows = _run(conn, question, k, None)

    results = [
        RetrievedChunk(
            chunk_id=chunk_id,
            company=company,
            year=year,
            section=section,
            pages=pages,
            text=text,
            context=context,
            score=float(score),
        )
        for chunk_id, company, year, section, pages, text, context, score in rows
    ]
    log.info(
        "retrieval.keyword k=%d filtered=%s results=%d latency=%.3fs",
        k,
        filters is not None,
        len(results),
        time.perf_counter() - start,
    )
    return results
