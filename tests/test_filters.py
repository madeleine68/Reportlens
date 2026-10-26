from pathlib import Path
from typing import Any

import numpy as np
import psycopg
import pytest
from pydantic import BaseModel

from reportlens.ingest.load import load_document
from reportlens.ingest.models import Chunk, ParsedDocument
from reportlens.retrieval.filters import (
    FilterCache,
    QueryFilters,
    extract_filters,
    extract_filters_cached,
    known_values,
)
from reportlens.retrieval.vector import vector_search

KNOWN = (["bce", "rbc", "scotiabank"], [2021, 2022, 2023, 2024])


class FakeStructured:
    """Stands in for llm.structured: returns the given filters and keeps the prompt."""

    def __init__(self, companies: list[str], years: list[int]) -> None:
        self.reply = QueryFilters(companies=companies, years=years)
        self.messages: list[dict[str, str]] = []

    def __call__(self, messages: Any, schema: type[BaseModel], tier: str) -> BaseModel:
        assert schema is QueryFilters and tier == "fast"
        self.messages = messages
        return self.reply


def test_prompt_lists_the_companies_and_years_in_the_database() -> None:
    fake = FakeStructured(["rbc"], [2023])
    extract_filters("What was RBC's 2023 net income?", KNOWN, structured=fake)

    system = fake.messages[0]["content"]
    assert "bce, rbc, scotiabank" in system
    assert "[2021, 2022, 2023, 2024]" in system
    assert fake.messages[1]["content"] == "What was RBC's 2023 net income?"


def test_values_are_normalized_deduplicated_and_validated() -> None:
    fake = FakeStructured([" RBC", "rbc", "TD", "apple"], [2023, 2023, 2019])
    filters = extract_filters("q", KNOWN, structured=fake)
    assert filters == QueryFilters(companies=["rbc"], years=[2023])


def test_question_without_company_or_year_gives_empty_filters() -> None:
    filters = extract_filters("Who won the Stanley Cup?", KNOWN, structured=FakeStructured([], []))
    assert filters.is_empty()


# --- Filtered search against the reportlens_test database -----------------------------------

REPORTS = [("rbc", 2023), ("rbc", 2024), ("bce", 2024)]
CHUNKS_PER_REPORT = 8
QUERY = np.eye(1, 384)[0]


def _load_reports(conn: psycopg.Connection) -> None:
    """Three small reports. Every chunk points near QUERY; bce chunks point closest."""
    for company, year in REPORTS:
        doc = ParsedDocument(
            company=company,
            year=year,
            source_path=f"data/raw/{company}/{year}.pdf",
            sha256="x",
            converter_version="1",
            extract_version="1",
            num_pages=CHUNKS_PER_REPORT,
            blocks=[],
        )
        chunks = [
            Chunk(
                chunk_id=f"{company}-{year}-{i:05d}",
                company=company,
                year=year,
                text=f"{company} {year} chunk {i}",
                pages=[i + 1],
                section=None,
                num_tokens=4,
            )
            for i in range(CHUNKS_PER_REPORT)
        ]
        closeness = 3.0 if company == "bce" else 1.0
        rng = np.random.default_rng(len(company) + year)

        def embed(texts: list[str], closeness: float = closeness, rng=rng) -> np.ndarray:
            vectors = rng.normal(size=(len(texts), 384)) * 0.1 + QUERY * closeness
            return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        load_document(conn, doc, chunks, embed=embed)


def _search(conn: psycopg.Connection, filters: QueryFilters | None, k: int = 5):
    return vector_search(conn, "anything", k=k, embed=lambda q: QUERY, filters=filters)


def test_unfiltered_search_prefers_the_closest_report(conn: psycopg.Connection) -> None:
    _load_reports(conn)
    assert {r.company for r in _search(conn, None)} == {"bce"}


def test_filters_restrict_company_and_year(conn: psycopg.Connection) -> None:
    _load_reports(conn)
    results = _search(conn, QueryFilters(companies=["rbc"], years=[2023]))

    assert len(results) == 5  # iterative scan still fills k despite the strict filter
    assert {(r.company, r.year) for r in results} == {("rbc", 2023)}


def test_multiple_years_are_allowed(conn: psycopg.Connection) -> None:
    _load_reports(conn)
    results = _search(conn, QueryFilters(companies=["rbc"], years=[2023, 2024]), k=16)
    assert {r.year for r in results} == {2023, 2024}


def test_filter_matching_nothing_falls_back_to_unfiltered(conn: psycopg.Connection) -> None:
    _load_reports(conn)
    results = _search(conn, QueryFilters(companies=["scotiabank"], years=[2021]))
    assert len(results) == 5 and {r.company for r in results} == {"bce"}


def test_empty_filters_behave_like_no_filters(conn: psycopg.Connection) -> None:
    _load_reports(conn)
    assert _search(conn, QueryFilters()) == _search(conn, None)


def test_known_values_reads_the_database(conn: psycopg.Connection) -> None:
    _load_reports(conn)
    assert known_values(conn) == (["bce", "rbc"], [2023, 2024])


@pytest.mark.parametrize("enabled", [True, False])
def test_search_uses_filters_only_when_enabled(
    enabled: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reportlens.core.config import get_settings
    from reportlens.retrieval import search as search_mod

    monkeypatch.setenv("RETRIEVAL_USE_FILTERS", str(enabled).lower())
    get_settings.cache_clear()
    seen: dict[str, object] = {}
    monkeypatch.setattr(search_mod, "known_values", lambda conn: KNOWN)
    monkeypatch.setattr(search_mod, "filter_cache", lambda: None)
    monkeypatch.setattr(
        search_mod,
        "extract_filters_cached",
        lambda q, known, cache: QueryFilters(companies=["rbc"]),
    )

    def fake_vector_search(conn, question, k, filters, kind=None):
        seen["f"] = filters
        return []

    monkeypatch.setattr(search_mod, "vector_search", fake_vector_search)

    search_mod.search(None, "What was RBC's net income?")
    assert seen["f"] == (QueryFilters(companies=["rbc"]) if enabled else None)


# --- Filter cache ----------------------------------------------------------------------------


class CountingStructured(FakeStructured):
    def __init__(self, companies: list[str], years: list[int]) -> None:
        super().__init__(companies, years)
        self.calls = 0

    def __call__(self, messages: Any, schema: type[BaseModel], tier: str) -> BaseModel:
        self.calls += 1
        return super().__call__(messages, schema, tier)


def test_cache_skips_the_llm_for_a_repeated_question(tmp_path: Path) -> None:
    llm = CountingStructured(["rbc"], [2023])
    cache = FilterCache(tmp_path / "filters.json")

    first = extract_filters_cached("RBC 2023 net income?", KNOWN, cache, structured=llm)
    second = extract_filters_cached("RBC 2023 net income?", KNOWN, cache, structured=llm)

    assert first == second == QueryFilters(companies=["rbc"], years=[2023])
    assert llm.calls == 1


def test_cache_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "cache" / "filters.json"
    extract_filters_cached("q", KNOWN, FilterCache(path), structured=FakeStructured(["bce"], []))

    llm = CountingStructured([], [])
    reloaded = extract_filters_cached("q", KNOWN, FilterCache(path), structured=llm)

    assert reloaded == QueryFilters(companies=["bce"])
    assert llm.calls == 0
    assert '"question": "q"' in path.read_text()  # stored readably


def test_cache_key_changes_with_question_known_values_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from reportlens.core.config import get_settings

    base = FilterCache.key("q", KNOWN)
    assert FilterCache.key("q2", KNOWN) != base
    assert FilterCache.key("q", (["bce"], [2024])) != base
    monkeypatch.setenv("LLM_MODEL_FAST", "another-model")
    get_settings.cache_clear()
    assert FilterCache.key("q", KNOWN) != base


def test_failed_extraction_is_not_cached(tmp_path: Path) -> None:
    cache = FilterCache(tmp_path / "filters.json")

    def failing(*args: Any, **kwargs: Any) -> BaseModel:
        raise RuntimeError("rate limited")

    with pytest.raises(RuntimeError):
        extract_filters_cached("q", KNOWN, cache, structured=failing)
    assert cache.get(FilterCache.key("q", KNOWN)) is None
