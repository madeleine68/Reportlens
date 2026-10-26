"""Chart slots in search(): charts are searched on their own, then given reserved places."""

import pytest

from reportlens.core.config import get_settings
from reportlens.retrieval import search as search_mod
from reportlens.retrieval.filters import QueryFilters, filter_conditions
from reportlens.retrieval.models import RetrievedChunk


def chunk(chunk_id: str, score: float, company: str = "rbc", year: int = 2023) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, company=company, year=year, section=None, pages=[1], text="", score=score
    )


TEXT = [chunk(f"text-{i}", score=10.0 - i) for i in range(5)]  # scores 10, 9, 8, 7, 6


@pytest.fixture
def fake_retrieve(monkeypatch: pytest.MonkeyPatch):
    """Replace the retriever: text and chart results come from the `results` dict by kind."""
    results: dict[str | None, list[RetrievedChunk]] = {"text": TEXT, "figure": [], None: TEXT}
    calls: list[tuple[int, str | None]] = []

    def retrieve(conn, question, k, filters, kind=None):
        calls.append((k, kind))
        return results[kind][:k]

    monkeypatch.setattr(search_mod, "_retrieve", retrieve)
    monkeypatch.setenv("RETRIEVAL_RERANK", "true")
    return results, calls


def use_slots(monkeypatch: pytest.MonkeyPatch, slots: int, filters: QueryFilters | None = None):
    monkeypatch.setenv("RETRIEVAL_FIGURE_SLOTS", str(slots))
    if filters is not None:
        monkeypatch.setenv("RETRIEVAL_USE_FILTERS", "true")
        monkeypatch.setattr(search_mod, "known_values", lambda conn: ([], []))
        monkeypatch.setattr(search_mod, "filter_cache", lambda: None)
        monkeypatch.setattr(search_mod, "extract_filters_cached", lambda q, known, cache: filters)
    get_settings.cache_clear()


def ids(chunks: list[RetrievedChunk]) -> list[str]:
    return [c.chunk_id for c in chunks]


def test_without_slots_one_search_runs_as_before(fake_retrieve, monkeypatch) -> None:
    _, calls = fake_retrieve
    use_slots(monkeypatch, 0)
    assert ids(search_mod.search(None, "q")) == ids(TEXT)
    assert calls == [(5, None)]


def test_best_chart_replaces_the_last_text_chunk_and_is_ranked_by_score(
    fake_retrieve, monkeypatch
) -> None:
    results, _ = fake_retrieve
    results["figure"] = [chunk("chart", score=8.5)]
    use_slots(monkeypatch, 1)
    assert ids(search_mod.search(None, "q")) == ["text-0", "text-1", "chart", "text-2", "text-3"]


def test_text_fills_the_slot_when_there_is_no_chart(fake_retrieve, monkeypatch) -> None:
    use_slots(monkeypatch, 1)
    assert ids(search_mod.search(None, "q")) == ids(TEXT)


def test_chart_from_another_report_is_dropped(fake_retrieve, monkeypatch) -> None:
    # The retrievers fall back to every report when nothing matches the filters.
    results, _ = fake_retrieve
    results["figure"] = [chunk("bce-chart", score=50.0, company="bce")]
    use_slots(monkeypatch, 1, filters=QueryFilters(companies=["rbc"], years=[2023]))
    assert ids(search_mod.search(None, "q")) == ids(TEXT)


def test_without_rerank_charts_go_last(fake_retrieve, monkeypatch) -> None:
    # Scores from different searches (cosine, RRF) don't compare, so keep the reserved order.
    results, _ = fake_retrieve
    results["figure"] = [chunk("chart", score=50.0)]
    monkeypatch.setenv("RETRIEVAL_RERANK", "false")
    use_slots(monkeypatch, 1)
    assert ids(search_mod.search(None, "q"))[-1] == "chart"


@pytest.mark.parametrize("figures", [True, False])
def test_kind_overrides_the_figures_setting(figures: bool, monkeypatch) -> None:
    monkeypatch.setenv("RETRIEVAL_FIGURES", str(figures).lower())
    get_settings.cache_clear()
    assert filter_conditions(None, kind="figure")[0] == ["kind = 'figure'"]
    assert filter_conditions(None, kind="text")[0] == ["kind = 'text'"]
    assert filter_conditions(None)[0] == ([] if figures else ["kind = 'text'"])
