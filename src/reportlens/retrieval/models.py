"""Shared result type for every retriever (vector now; keyword, hybrid and rerank in Phase 2)."""

from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    """A chunk returned by search, with enough metadata to cite it."""

    chunk_id: str
    company: str
    year: int
    section: str | None
    pages: list[int]
    text: str
    score: float  # higher is more relevant; scale depends on the retriever
