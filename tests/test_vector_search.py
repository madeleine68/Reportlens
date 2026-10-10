"""Vector search against a real Postgres (reportlens_test), with hand-made vectors."""

import numpy as np
import psycopg
import pytest

from reportlens.ingest.load import load_document
from reportlens.ingest.models import Chunk, ParsedDocument
from reportlens.retrieval.vector import vector_search

E0, E1 = np.eye(2, 384)
# chunk 0 points along E0, chunk 2 along E1, chunk 1 halfway between (45 degrees from each).
VECTORS = np.array([E0, (E0 + E1) / np.sqrt(2), E1])


def _load_three_chunks(conn: psycopg.Connection) -> None:
    doc = ParsedDocument(
        company="bce",
        year=2022,
        source_path="data/raw/bce/2022.pdf",
        sha256="x",
        converter_version="1",
        extract_version="1",
        num_pages=9,
        blocks=[],
    )
    chunks = [
        Chunk(
            chunk_id=f"bce-2022-{i:05d}",
            company="bce",
            year=2022,
            text=f"chunk {i}",
            pages=[i + 1],
            section=f"Section {i}",
            num_tokens=2,
        )
        for i in range(3)
    ]
    load_document(conn, doc, chunks, embed=lambda texts: VECTORS)


def test_returns_closest_chunks_first_with_cosine_scores(conn: psycopg.Connection) -> None:
    _load_three_chunks(conn)

    results = vector_search(conn, "anything", k=3, embed=lambda q: E0)

    assert [r.chunk_id for r in results] == ["bce-2022-00000", "bce-2022-00001", "bce-2022-00002"]
    assert [round(r.score, 3) for r in results] == [1.0, 0.707, 0.0]


def test_results_carry_citation_metadata(conn: psycopg.Connection) -> None:
    _load_three_chunks(conn)

    [top] = vector_search(conn, "anything", k=1, embed=lambda q: E1)

    assert top.model_dump(exclude={"score"}) == {
        "chunk_id": "bce-2022-00002",
        "company": "bce",
        "year": 2022,
        "section": "Section 2",
        "pages": [3],
        "text": "chunk 2",
        "context": "",  # naive chunks carry no label
    }


def test_k_defaults_to_settings(conn: psycopg.Connection, monkeypatch: pytest.MonkeyPatch) -> None:
    from reportlens.core.config import get_settings

    _load_three_chunks(conn)
    monkeypatch.setenv("RETRIEVAL_TOP_K", "2")
    get_settings.cache_clear()

    assert len(vector_search(conn, "anything", embed=lambda q: E0)) == 2


def test_empty_database_returns_nothing(conn: psycopg.Connection) -> None:
    assert vector_search(conn, "anything", k=5, embed=lambda q: E0) == []
