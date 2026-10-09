"""Keyword (full-text) search against a real Postgres (reportlens_test)."""

import numpy as np
import psycopg
import pytest

from reportlens.ingest.load import load_document
from reportlens.ingest.models import Chunk, ParsedDocument
from reportlens.retrieval.filters import QueryFilters
from reportlens.retrieval.keyword import keyword_search

TEXTS = {
    ("rbc", 2023): [
        "Our CET1 ratio was 14.5%, up 190 bps from last year.",
        "Net income of $14,866 million was down 6% from last year.",
        "The board met eleven times and reviewed climate risk.",
    ],
    ("scotiabank", 2023): [
        "Net income attributable to equity holders was $7,410 million.",
        "The Basel III CET1 ratio was 13.0% as at October 31, 2023.",
    ],
}


def _load(conn: psycopg.Connection) -> None:
    for (company, year), texts in TEXTS.items():
        doc = ParsedDocument(
            company=company,
            year=year,
            source_path=f"data/raw/{company}/{year}.pdf",
            sha256="x",
            converter_version="1",
            extract_version="1",
            num_pages=len(texts),
            blocks=[],
        )
        chunks = [
            Chunk(
                chunk_id=f"{company}-{year}-{i:05d}",
                company=company,
                year=year,
                text=text,
                pages=[i + 1],
                section=None,
                num_tokens=10,
            )
            for i, text in enumerate(texts)
        ]
        load_document(conn, doc, chunks, embed=lambda t: np.eye(len(t), 384))


def _ids(results) -> list[str]:
    return [r.chunk_id for r in results]


def test_exact_term_is_found(conn: psycopg.Connection) -> None:
    _load(conn)
    results = keyword_search(conn, "What is the CET1 ratio?", k=5)
    assert set(_ids(results)[:2]) == {"rbc-2023-00000", "scotiabank-2023-00001"}


def test_any_term_matches_but_more_terms_rank_higher(conn: psycopg.Connection) -> None:
    _load(conn)
    results = keyword_search(conn, "net income attributable to equity holders", k=5)

    # Both net-income chunks match ("any term" OR query), the one with every term ranks first.
    assert _ids(results)[:2] == ["scotiabank-2023-00000", "rbc-2023-00001"]
    assert results[0].score > results[1].score


def test_stemming_matches_word_forms(conn: psycopg.Connection) -> None:
    _load(conn)
    assert _ids(keyword_search(conn, "boards meeting", k=1)) == ["rbc-2023-00002"]


def test_question_of_only_stopwords_returns_nothing(conn: psycopg.Connection) -> None:
    _load(conn)
    assert keyword_search(conn, "what was it?", k=5) == []


def test_filters_restrict_results(conn: psycopg.Connection) -> None:
    _load(conn)
    results = keyword_search(conn, "CET1 ratio", k=5, filters=QueryFilters(companies=["rbc"]))
    assert {r.company for r in results} == {"rbc"}


def test_filter_matching_nothing_falls_back_to_unfiltered(conn: psycopg.Connection) -> None:
    _load(conn)
    results = keyword_search(conn, "CET1 ratio", k=5, filters=QueryFilters(years=[2019]))
    assert len(results) == 2


@pytest.mark.parametrize("mode", ["vector", "keyword"])
def test_search_mode_setting_picks_the_retriever(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reportlens.core.config import get_settings
    from reportlens.retrieval import search as search_mod

    monkeypatch.setenv("RETRIEVAL_MODE", mode)
    get_settings.cache_clear()
    called: list[str] = []
    monkeypatch.setattr(search_mod, "vector_search", lambda *a, **kw: called.append("vector"))
    monkeypatch.setattr(search_mod, "keyword_search", lambda *a, **kw: called.append("keyword"))

    search_mod.search(None, "q")
    assert called == [mode]
