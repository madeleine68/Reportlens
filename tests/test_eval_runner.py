from typing import Any

import pytest
from pydantic import BaseModel

from reportlens.evals.judge import CorrectnessVerdict, FaithfulnessVerdict
from reportlens.evals.questions import EvalQuestion, ExpectedSource
from reportlens.evals.runner import QuestionResult, evaluate_question, results_row, summarize
from reportlens.generation.answer import Answer, Citation
from reportlens.generation.prompts import NOT_FOUND
from reportlens.retrieval.models import RetrievedChunk

CHUNK = RetrievedChunk(
    chunk_id="rbc-2023-00007",
    company="rbc",
    year=2023,
    section=None,
    pages=[12],
    text="t",
    score=0.8,
)
NUMERIC_Q = EvalQuestion(
    id="q1",
    question="What was RBC's net income in 2023?",
    type="numeric",
    expected_answer="$16.2 billion",
    sources=[ExpectedSource(company="RBC", year=2023, page=12)],
)
OUT_OF_CORPUS_Q = EvalQuestion(
    id="q2", question="Who won the 2023 Stanley Cup?", type="out_of_corpus", expected_answer=""
)


def _answer(text: str, found: bool = True) -> Answer:
    return Answer(
        question="q",
        text=text,
        found=found,
        citations=[
            Citation(number=1, chunk_id=CHUNK.chunk_id, company="rbc", year=2023, pages=[12])
        ]
        if found
        else [],
        chunks=[CHUNK],
        model="main-model",
        input_tokens=2_000,
        output_tokens=100,
        latency_s=1.0,
    )


class FakeJudge:
    """Stands in for llm.structured: returns canned verdicts and records which judges ran."""

    def __init__(self, verdict: str = "correct", unsupported: list[str] | None = None) -> None:
        self.verdict = verdict
        self.unsupported = unsupported or []
        self.calls: list[str] = []

    def __call__(self, messages: Any, schema: type[BaseModel], tier: str) -> BaseModel:
        assert tier == "fast"
        self.calls.append(schema.__name__)
        if schema is CorrectnessVerdict:
            return CorrectnessVerdict(reason="matches", verdict=self.verdict)
        return FaithfulnessVerdict(reason="", total_claims=2, unsupported_claims=self.unsupported)


def _evaluate(q: EvalQuestion, answer: Answer, judge: FakeJudge) -> QuestionResult:
    return evaluate_question(None, q, answer_fn=lambda conn, text: answer, structured=judge)


def test_answered_question_gets_all_metrics() -> None:
    judge = FakeJudge(verdict="partially_correct", unsupported=["extra claim"])
    r = _evaluate(NUMERIC_Q, _answer("Net income was $16.2B [1]."), judge)

    assert judge.calls == ["CorrectnessVerdict", "FaithfulnessVerdict"]
    assert (r.recall, r.reciprocal_rank) == (1.0, 1.0)
    assert r.correctness == 0.5
    assert r.faithfulness == 0.5
    assert r.unsupported_claims == ["extra claim"]
    assert r.cited == ["rbc-2023-00007"]
    assert (r.input_tokens, r.output_tokens) == (2_000, 100)


def test_refusal_on_answerable_question_scores_zero_without_judging() -> None:
    judge = FakeJudge()
    r = _evaluate(NUMERIC_Q, _answer(NOT_FOUND, found=False), judge)

    assert judge.calls == []
    assert (r.correctness, r.faithfulness) == (0.0, None)
    assert r.recall == 1.0  # retrieval found the page even though the answer refused


@pytest.mark.parametrize(("found", "score"), [(False, 1.0), (True, 0.0)])
def test_out_of_corpus_is_correct_only_when_refused(found: bool, score: float) -> None:
    judge = FakeJudge()
    text = "Some answer [1]." if found else NOT_FOUND
    r = _evaluate(OUT_OF_CORPUS_Q, _answer(text, found=found), judge)

    assert r.correctness == score
    assert "CorrectnessVerdict" not in judge.calls
    assert (r.recall, r.reciprocal_rank) == (None, None)


def test_cost_uses_configured_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    from reportlens.core.config import get_settings

    monkeypatch.setenv("LLM_PRICE_INPUT_PER_MTOK", "0.15")
    monkeypatch.setenv("LLM_PRICE_OUTPUT_PER_MTOK", "0.60")
    get_settings.cache_clear()
    r = _evaluate(NUMERIC_Q, _answer("x [1]."), FakeJudge())
    assert r.cost_usd == pytest.approx((2_000 * 0.15 + 100 * 0.60) / 1e6)


def test_summary_averages_applicable_values_and_skips_errors() -> None:
    results = [
        QuestionResult(
            id="a",
            type="numeric",
            question="",
            expected_answer="",
            recall=1.0,
            reciprocal_rank=0.5,
            correctness=1.0,
            faithfulness=1.0,
            latency_s=1.0,
            input_tokens=1000,
            output_tokens=100,
        ),
        QuestionResult(
            id="b",
            type="out_of_corpus",
            question="",
            expected_answer="",
            correctness=0.0,
            latency_s=3.0,
            input_tokens=900,
            output_tokens=50,
        ),
        QuestionResult(id="c", type="numeric", question="", expected_answer="", error="RateLimit"),
    ]
    s = summarize(results)

    assert (s.n, s.errors) == (3, 1)
    assert (s.recall, s.mrr) == (1.0, 0.5)  # out_of_corpus has no retrieval score
    assert (s.correctness, s.faithfulness) == (0.5, 1.0)
    assert s.p50_latency_s == 2.0
    assert s.tokens_per_query == 1025


def test_results_row_matches_results_md_columns() -> None:
    s = summarize(
        [
            QuestionResult(
                id="a",
                type="numeric",
                question="",
                expected_answer="",
                recall=0.8,
                reciprocal_rank=0.65,
                correctness=0.5,
                faithfulness=0.9,
                latency_s=1.4,
                input_tokens=2600,
                output_tokens=250,
            ),
        ]
    )
    row = results_row("2026-10-09", "baseline", s, notes="vector top-5")
    assert row == (
        "| 2026-10-09 | baseline | 0.80 | 0.65 | 0.50 | 0.90 | 1.4s | $0.0000 (2,850 tok) "
        "| n=1; vector top-5 |"
    )
    assert row.count("|") == 10  # 9 columns, like the results.md header
