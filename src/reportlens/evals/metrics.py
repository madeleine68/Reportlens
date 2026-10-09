"""Retrieval and summary metrics: plain functions, no LLM and no I/O."""

import statistics

from reportlens.evals.questions import ExpectedSource
from reportlens.retrieval.models import RetrievedChunk


def chunk_matches(chunk: RetrievedChunk, source: ExpectedSource) -> bool:
    """A chunk covers a source if it is from the same report and spans the expected page."""
    return (
        chunk.company.lower() == source.company.lower()
        and chunk.year == source.year
        and source.page in chunk.pages
    )


def recall_at_k(
    retrieved: list[RetrievedChunk], sources: list[ExpectedSource], k: int
) -> float | None:
    """Share of expected sources covered by the top-k chunks. None if there are no sources."""
    if not sources:
        return None
    top = retrieved[:k]
    found = sum(any(chunk_matches(chunk, s) for chunk in top) for s in sources)
    return found / len(sources)


def reciprocal_rank(retrieved: list[RetrievedChunk], sources: list[ExpectedSource]) -> float | None:
    """1 / rank of the first chunk covering any expected source (0 if none). None if no sources."""
    if not sources:
        return None
    for rank, chunk in enumerate(retrieved, start=1):
        if any(chunk_matches(chunk, s) for s in sources):
            return 1 / rank
    return 0.0


def mean(values: list[float | None]) -> float | None:
    """Average of the values that apply (None means 'not applicable', not zero)."""
    present = [v for v in values if v is not None]
    return statistics.fmean(present) if present else None


def p50(values: list[float]) -> float | None:
    """Median, the usual 'typical latency' figure."""
    return statistics.median(values) if values else None


def cost_usd(input_tokens: int, output_tokens: int, price_in: float, price_out: float) -> float:
    """Dollar cost from token counts and per-million-token prices."""
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000
