"""Integration tests against a real Postgres (a separate reportlens_test database)."""

import numpy as np
import psycopg
import pytest

from reportlens.core.config import get_settings
from reportlens.ingest.load import document_id, ingest_config, is_current, load_document
from reportlens.ingest.models import Chunk, ParsedDocument


def fake_embed(texts: list[str]) -> np.ndarray:
    """One-hot vectors, so each chunk is distinguishable without loading a model."""
    return np.eye(len(texts), 384)


def _doc(sha256: str = "abc") -> ParsedDocument:
    return ParsedDocument(
        company="rbc",
        year=2023,
        source_path="data/raw/rbc/2023.pdf",
        sha256=sha256,
        converter_version="1",
        extract_version="1",
        num_pages=3,
        blocks=[],
    )


def _chunks(n: int) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"rbc-2023-{i:05d}",
            company="rbc",
            year=2023,
            text=f"chunk {i}",
            pages=[i + 1, i + 2],
            section="Net income",
            num_tokens=2,
        )
        for i in range(n)
    ]


def test_load_stores_chunks_with_metadata_and_vectors(conn: psycopg.Connection) -> None:
    load_document(conn, _doc(), _chunks(2), embed=fake_embed)

    rows = conn.execute(
        "SELECT chunk_id, document_id, company, year, section, pages, text, embedding"
        " FROM chunks ORDER BY chunk_id"
    ).fetchall()
    assert [r[:7] for r in rows] == [
        ("rbc-2023-00000", "rbc-2023", "rbc", 2023, "Net income", [1, 2], "chunk 0"),
        ("rbc-2023-00001", "rbc-2023", "rbc", 2023, "Net income", [2, 3], "chunk 1"),
    ]
    assert np.array_equal(rows[1][7].to_numpy(), np.eye(2, 384)[1])


def test_is_current_only_after_loading_same_pdf_and_settings(
    conn: psycopg.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert not is_current(conn, _doc())
    load_document(conn, _doc(), _chunks(1), embed=fake_embed)
    assert is_current(conn, _doc())
    assert not is_current(conn, _doc(sha256="changed"))

    monkeypatch.setenv("CHUNK_TOKENS", "300")
    get_settings.cache_clear()
    assert not is_current(conn, _doc())


def test_reload_replaces_old_chunks(conn: psycopg.Connection) -> None:
    load_document(conn, _doc(), _chunks(3), embed=fake_embed)
    load_document(conn, _doc(sha256="new"), _chunks(1), embed=fake_embed)

    assert conn.execute("SELECT count(*) FROM chunks").fetchone() == (1,)
    assert conn.execute("SELECT sha256 FROM documents").fetchall() == [("new",)]


def test_failed_load_leaves_previous_version_intact(conn: psycopg.Connection) -> None:
    load_document(conn, _doc(), _chunks(2), embed=fake_embed)

    def too_few_vectors(texts: list[str]) -> np.ndarray:
        return np.eye(1, 384)  # mismatched length makes the insert fail mid-transaction

    with pytest.raises(ValueError):
        load_document(conn, _doc(sha256="new"), _chunks(2), embed=too_few_vectors)

    assert conn.execute("SELECT count(*) FROM chunks").fetchone() == (2,)
    assert conn.execute("SELECT sha256 FROM documents").fetchall() == [("abc",)]


def test_ids_and_config_fingerprint() -> None:
    assert document_id(_doc()) == "rbc-2023"
    assert ingest_config(_doc()) == (
        "converter=1;extract=1;chunk=450/50;model=BAAI/bge-small-en-v1.5"
    )
