import pytest

from reportlens.retrieval import hybrid
from reportlens.retrieval.filters import QueryFilters
from reportlens.retrieval.hybrid import rrf_merge
from reportlens.retrieval.models import RetrievedChunk


def _c(chunk_id: str, score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, company="rbc", year=2023, section=None, pages=[1], text="", score=score
    )


def _ids(chunks: list[RetrievedChunk]) -> list[str]:
    return [c.chunk_id for c in chunks]


def test_scores_follow_the_rrf_formula() -> None:
    vector = [_c("a"), _c("b"), _c("c")]
    keyword = [_c("b"), _c("d")]

    merged = rrf_merge([vector, keyword], k=60)

    # b: rank 2 + rank 1 -> 1/62 + 1/61 ; a: 1/61 ; d: 1/62 ; c: 1/63
    assert _ids(merged) == ["b", "a", "d", "c"]
    assert merged[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert [round(c.score, 6) for c in merged[1:]] == [
        round(1 / 61, 6),
        round(1 / 62, 6),
        round(1 / 63, 6),
    ]


def test_agreement_beats_a_single_first_place() -> None:
    # x is 1st in one list only; y is 3rd in both. 2/63 > 1/61, so y wins.
    merged = rrf_merge([[_c("x"), _c("p"), _c("y")], [_c("q"), _c("r"), _c("y")]], k=60)
    assert _ids(merged)[0] == "y"


def test_raw_retriever_scores_are_ignored() -> None:
    # A huge keyword score must not outweigh rank: only positions count.
    merged = rrf_merge([[_c("a", score=0.9)], [_c("b", score=500.0)]], k=60)
    assert [c.score for c in merged] == [pytest.approx(1 / 61)] * 2


def test_ties_keep_first_seen_order() -> None:
    merged = rrf_merge([[_c("a"), _c("b")], [_c("c"), _c("d")]], k=60)
    assert _ids(merged) == ["a", "c", "b", "d"]


def test_top_n_truncates_and_single_list_keeps_its_order() -> None:
    assert _ids(rrf_merge([[_c("a"), _c("b"), _c("c")]], top_n=2)) == ["a", "b"]


def test_empty_inputs() -> None:
    assert rrf_merge([]) == []
    assert rrf_merge([[], [_c("a")]]) == [rrf_merge([[_c("a")]])[0]]


def test_inputs_are_not_modified() -> None:
    original = _c("a", score=0.42)
    rrf_merge([[original]])
    assert original.score == 0.42


def test_hybrid_search_fuses_deep_candidates_from_both_retrievers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int, QueryFilters | None]] = []

    def fake(name: str, ids: list[str]):
        def search(conn, question, k, filters):
            calls.append((name, k, filters))
            return [_c(i) for i in ids]

        return search

    monkeypatch.setattr(hybrid, "vector_search", fake("vector", ["a", "b", "c"]))
    monkeypatch.setattr(hybrid, "keyword_search", fake("keyword", ["c", "d"]))
    filters = QueryFilters(companies=["rbc"])

    results = hybrid.hybrid_search(None, "q", k=2, filters=filters)

    assert calls == [("vector", 20, filters), ("keyword", 20, filters)]  # depth from settings
    assert _ids(results) == ["c", "a"]  # c is in both lists
