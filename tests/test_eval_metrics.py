from pathlib import Path

import pytest

from reportlens.evals import metrics
from reportlens.evals.judge import FaithfulnessVerdict, faithfulness_score
from reportlens.evals.questions import ExpectedSource, load_questions
from reportlens.retrieval.models import RetrievedChunk


def _chunk(company: str, year: int, pages: list[int]) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{company}-{year}-x",
        company=company,
        year=year,
        section=None,
        pages=pages,
        text="",
        score=0.5,
    )


RETRIEVED = [
    _chunk("rbc", 2022, [12]),  # right page, wrong year
    _chunk("rbc", 2023, [40, 41]),
    _chunk("td", 2023, [12]),
    _chunk("rbc", 2023, [12, 13]),
]
NET_INCOME_P12 = ExpectedSource(company="RBC", year=2023, page=12)
CAPITAL_P41 = ExpectedSource(company="RBC", year=2023, page=41)


def test_chunk_match_needs_company_year_and_page() -> None:
    assert metrics.chunk_matches(RETRIEVED[3], NET_INCOME_P12)  # company match is case-insensitive
    assert not metrics.chunk_matches(RETRIEVED[0], NET_INCOME_P12)
    assert not metrics.chunk_matches(RETRIEVED[2], NET_INCOME_P12)


def test_recall_counts_sources_found_in_top_k() -> None:
    sources = [NET_INCOME_P12, CAPITAL_P41]
    assert metrics.recall_at_k(RETRIEVED, sources, k=4) == 1.0
    assert metrics.recall_at_k(RETRIEVED, sources, k=3) == 0.5  # p12 chunk is 4th
    assert metrics.recall_at_k(RETRIEVED, sources, k=1) == 0.0


def test_reciprocal_rank_uses_first_matching_chunk() -> None:
    assert metrics.reciprocal_rank(RETRIEVED, [NET_INCOME_P12]) == 0.25
    assert metrics.reciprocal_rank(RETRIEVED, [NET_INCOME_P12, CAPITAL_P41]) == 0.5
    assert (
        metrics.reciprocal_rank(RETRIEVED, [ExpectedSource(company="bce", year=2023, page=1)]) == 0
    )


def test_retrieval_metrics_do_not_apply_without_sources() -> None:
    assert metrics.recall_at_k(RETRIEVED, [], k=5) is None
    assert metrics.reciprocal_rank(RETRIEVED, []) is None


def test_mean_skips_not_applicable_values() -> None:
    assert metrics.mean([1.0, None, 0.0]) == 0.5
    assert metrics.mean([None]) is None


def test_p50_and_cost() -> None:
    assert metrics.p50([3.0, 1.0, 2.0, 10.0]) == 2.5
    assert metrics.p50([]) is None
    assert metrics.cost_usd(2_000_000, 500_000, price_in=0.15, price_out=0.60) == pytest.approx(0.6)


@pytest.mark.parametrize(
    ("total", "unsupported", "expected"),
    [(4, [], 1.0), (4, ["x"], 0.75), (0, [], 1.0), (1, ["a", "b"], 0.0)],
)
def test_faithfulness_score(total: int, unsupported: list[str], expected: float) -> None:
    verdict = FaithfulnessVerdict(reason="", total_claims=total, unsupported_claims=unsupported)
    assert faithfulness_score(verdict) == expected


def test_load_questions_skips_comments_and_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    path.write_text(
        "// smoke test only\n\n"
        '{"id": "q1", "question": "Net income?", "type": "numeric", "expected_answer": "$1", '
        '"sources": [{"company": "rbc", "year": 2023, "page": 12}]}\n'
        '{"id": "q2", "question": "Stanley Cup?", "type": "out_of_corpus", "expected_answer": ""}\n'
    )
    questions = load_questions(path)
    assert [(q.id, q.type, len(q.sources)) for q in questions] == [
        ("q1", "numeric", 1),
        ("q2", "out_of_corpus", 0),
    ]


def test_load_questions_reports_bad_line(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    path.write_text('{"id": "q1", "question": "?", "type": "guess", "expected_answer": ""}\n')
    with pytest.raises(ValueError, match="q.jsonl:1"):
        load_questions(path)
