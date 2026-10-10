"""Reranking with a cross-encoder: rescore candidate chunks by reading question and chunk together.

Retrievers embed or index the question and the chunks separately, which is fast but coarse. A
cross-encoder reads each (question, chunk) pair jointly, so it judges relevance far better, but
it's too slow to run over every chunk. So retrieval gathers ~40 candidates and the cross-encoder
picks the best few.
"""

import time
from functools import lru_cache

from sentence_transformers import CrossEncoder

from reportlens.core.config import get_settings
from reportlens.core.logging import get_logger
from reportlens.retrieval.models import RetrievedChunk

log = get_logger(__name__)


@lru_cache
def _model() -> CrossEncoder:
    return CrossEncoder(get_settings().rerank_model, device="cpu")


def rerank(question: str, chunks: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
    """Return the top_n chunks by cross-encoder relevance, best first, with that as their score."""
    if not chunks:
        return []
    start = time.perf_counter()
    scores = _model().predict([(question, c.text) for c in chunks], show_progress_bar=False)
    ranked = sorted(zip(chunks, scores, strict=True), key=lambda pair: pair[1], reverse=True)
    results = [c.model_copy(update={"score": float(s)}) for c, s in ranked[:top_n]]
    log.info(
        "retrieval.rerank candidates=%d kept=%d latency=%.2fs",
        len(chunks),
        len(results),
        time.perf_counter() - start,
    )
    return results
