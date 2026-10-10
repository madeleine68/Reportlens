"""The retrieval entry point used by answer generation; settings pick which techniques run."""

from functools import lru_cache

import psycopg

from reportlens.core.config import get_settings
from reportlens.retrieval.filters import FilterCache, extract_filters_cached, known_values
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
    cross-encoder choose the final top k.
    """
    settings = get_settings()
    filters = None
    if settings.retrieval_use_filters:
        filters = extract_filters_cached(question, known_values(conn), filter_cache())

    retriever = {"vector": vector_search, "keyword": keyword_search, "hybrid": hybrid_search}[
        settings.retrieval_mode
    ]
    if not settings.retrieval_rerank:
        return retriever(conn, question, k=settings.retrieval_top_k, filters=filters)
    candidates = retriever(conn, question, k=settings.rerank_candidates, filters=filters)
    return rerank(question, candidates, top_n=settings.retrieval_top_k)
