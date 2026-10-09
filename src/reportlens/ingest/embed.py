"""Text embeddings with sentence-transformers. All vectors are L2-normalized (cosine = dot product).

Passages and queries are embedded differently: bge models expect an instruction before queries
only. Use embed_passages for chunks and embed_query for questions; never mix them up.
"""

import time
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from reportlens.core.config import get_settings
from reportlens.core.logging import get_logger

log = get_logger(__name__)


@lru_cache
def _model() -> SentenceTransformer:
    return SentenceTransformer(get_settings().embedding_model, device="cpu")


def _encode(texts: list[str], batch_size: int) -> np.ndarray:
    start = time.perf_counter()
    vectors = _model().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,  # timing is logged below instead
    )
    log.info("embed texts=%d latency=%.1fs", len(texts), time.perf_counter() - start)
    return vectors


def embed_passages(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Embed document chunks, one row per text."""
    return _encode(texts, batch_size)


def embed_query(question: str) -> np.ndarray:
    """Embed a user question, adding the model's query instruction."""
    return _encode([get_settings().embedding_query_prefix + question], batch_size=1)[0]
