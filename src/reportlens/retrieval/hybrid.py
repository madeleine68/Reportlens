"""Hybrid search: merge vector and keyword results with Reciprocal Rank Fusion (RRF).

RRF (Cormack et al., 2009) scores each chunk by summing 1 / (k + rank) over every ranked list
it appears in. It uses only ranks, never raw scores, so it can combine retrievers whose scores
are on different scales (cosine similarity is 0-1, ts_rank_cd is unbounded). A chunk ranked
well by both retrievers beats one ranked first by only one of them.
"""

import psycopg

from reportlens.core.config import get_settings
from reportlens.retrieval.filters import ChunkKind, QueryFilters
from reportlens.retrieval.keyword import keyword_search
from reportlens.retrieval.models import RetrievedChunk
from reportlens.retrieval.vector import vector_search


def rrf_merge(
    ranked_lists: list[list[RetrievedChunk]], k: int = 60, top_n: int | None = None
) -> list[RetrievedChunk]:
    """Fuse ranked lists into one, best first; each chunk's score becomes its RRF score.

    k dampens the advantage of the very top ranks: with k=60, rank 1 scores 1/61 and rank 10
    scores 1/70, so agreement between lists matters more than being first in one list.
    Ties keep the order in which chunks were first seen.
    """
    scores: dict[str, float] = {}
    chunks: dict[str, RetrievedChunk] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1 / (k + rank)
            chunks.setdefault(chunk.chunk_id, chunk)

    # sorted() is stable, so equal scores stay in first-seen order.
    best = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)[:top_n]
    return [chunks[chunk_id].model_copy(update={"score": scores[chunk_id]}) for chunk_id in best]


def hybrid_search(
    conn: psycopg.Connection,
    question: str,
    k: int | None = None,
    filters: QueryFilters | None = None,
    kind: ChunkKind | None = None,
) -> list[RetrievedChunk]:
    """Take the top candidates from vector and keyword search, fuse them, and keep the best k."""
    settings = get_settings()
    k = k or settings.retrieval_top_k
    depth = settings.hybrid_candidates  # look deeper than k so the fusion has more to choose from
    vector = vector_search(conn, question, k=depth, filters=filters, kind=kind)
    keyword = keyword_search(conn, question, k=depth, filters=filters, kind=kind)
    return rrf_merge([vector, keyword], k=settings.rrf_k, top_n=k)
