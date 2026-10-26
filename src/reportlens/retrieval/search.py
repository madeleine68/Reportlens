"""The retrieval entry point used by answer generation; settings pick which techniques run."""

from functools import lru_cache

import psycopg

from reportlens.core.config import get_settings
from reportlens.retrieval.filters import (
    ChunkKind,
    FilterCache,
    QueryFilters,
    extract_filters_cached,
    known_values,
)
from reportlens.retrieval.hybrid import hybrid_search
from reportlens.retrieval.keyword import keyword_search
from reportlens.retrieval.models import RetrievedChunk
from reportlens.retrieval.rerank import rerank
from reportlens.retrieval.vector import vector_search


@lru_cache
def filter_cache() -> FilterCache:
    """The process-wide filter cache (loaded from disk once)."""
    return FilterCache(get_settings().filter_cache_path)


def search(conn: psycopg.Connection, question: str) -> list[RetrievedChunk]:
    """Retrieve chunks for a question.

    RETRIEVAL_MODE picks the retriever (vector, keyword or hybrid); RETRIEVAL_USE_FILTERS adds
    LLM-extracted company/year filters; RETRIEVAL_RERANK fetches more candidates and lets a
    cross-encoder choose the final top k; RETRIEVAL_FIGURE_SLOTS reserves slots for charts.
    """
    settings = get_settings()
    filters = None
    if settings.retrieval_use_filters:
        filters = extract_filters_cached(question, known_values(conn), filter_cache())

    k = settings.retrieval_top_k
    slots = min(settings.retrieval_figure_slots, k)
    if not slots:
        return _retrieve(conn, question, k, filters)

    text = _retrieve(conn, question, k, filters, kind="text")
    charts = [
        chart
        for chart in _retrieve(conn, question, slots, filters, kind="figure")
        if _matches(chart, filters)
    ]
    chosen = text[: k - len(charts)] + charts
    if settings.retrieval_rerank:  # same question, same cross-encoder: the scores compare
        chosen.sort(key=lambda chunk: chunk.score, reverse=True)
    return chosen


def _retrieve(
    conn: psycopg.Connection,
    question: str,
    k: int,
    filters: QueryFilters | None,
    kind: ChunkKind | None = None,
) -> list[RetrievedChunk]:
    """The top k chunks (of one kind, if given) from the configured retriever and reranker."""
    settings = get_settings()
    retriever = {"vector": vector_search, "keyword": keyword_search, "hybrid": hybrid_search}[
        settings.retrieval_mode
    ]
    if not settings.retrieval_rerank:
        return retriever(conn, question, k=k, filters=filters, kind=kind)
    candidates = retriever(conn, question, k=settings.rerank_candidates, filters=filters, kind=kind)
    return rerank(question, candidates, top_n=k)


def _matches(chunk: RetrievedChunk, filters: QueryFilters | None) -> bool:
    """Whether a chunk is from a report the filters allow.

    The retrievers fall back to every report when no chunk matches the filters. For text that
    beats returning nothing, but a chart from another company would only take a text slot.
    """
    if filters is None:
        return True
    return (not filters.companies or chunk.company in filters.companies) and (
        not filters.years or chunk.year in filters.years
    )
