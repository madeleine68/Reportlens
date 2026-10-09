"""Run the RAG pipeline over eval questions and score each answer."""

import time
from collections.abc import Callable

import psycopg
from pydantic import BaseModel

from reportlens.core.config import get_settings
from reportlens.core.logging import get_logger
from reportlens.evals import judge, metrics
from reportlens.evals.questions import EvalQuestion
from reportlens.generation.answer import Answer, answer_question
from reportlens.retrieval.models import RetrievedChunk
from reportlens.retrieval.search import search

log = get_logger(__name__)

AnswerFn = Callable[[psycopg.Connection, str], Answer]
RetrieveFn = Callable[[psycopg.Connection, str], list[RetrievedChunk]]


class QuestionResult(BaseModel):
    """Everything about one question's run, saved so failures can be inspected later."""

    id: str
    type: str
    question: str
    expected_answer: str
    answer: str | None = None
    found: bool | None = None
    cited: list[str] = []  # chunk ids the answer cited
    retrieved: list[str] = []  # chunk ids retrieved, best first, with their pages
    recall: float | None = None
    reciprocal_rank: float | None = None
    correctness: float | None = None
    correctness_reason: str = ""
    faithfulness: float | None = None
    unsupported_claims: list[str] = []
    latency_s: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None


class Summary(BaseModel):
    n: int
    errors: int
    recall: float | None
    mrr: float | None
    correctness: float | None
    faithfulness: float | None
    p50_latency_s: float | None
    cost_per_query: float | None
    tokens_per_query: float | None


def evaluate_question(
    conn: psycopg.Connection,
    q: EvalQuestion,
    answer_fn: AnswerFn = answer_question,
    structured: judge.Structured | None = None,
) -> QuestionResult:
    """Answer one question and score retrieval, correctness and faithfulness."""
    settings = get_settings()
    judge_kwargs = {"structured": structured} if structured else {}
    result = QuestionResult(
        id=q.id, type=q.type, question=q.question, expected_answer=q.expected_answer
    )

    start = time.perf_counter()
    ans = answer_fn(conn, q.question)
    result.latency_s = round(time.perf_counter() - start, 3)

    result.answer, result.found = ans.text, ans.found
    result.cited = [c.chunk_id for c in ans.citations]
    result.retrieved = [f"{c.chunk_id} p{c.pages}" for c in ans.chunks]
    result.input_tokens, result.output_tokens = ans.input_tokens, ans.output_tokens
    result.cost_usd = metrics.cost_usd(
        ans.input_tokens,
        ans.output_tokens,
        settings.llm_price_input_per_mtok,
        settings.llm_price_output_per_mtok,
    )

    k = settings.retrieval_top_k
    result.recall = metrics.recall_at_k(ans.chunks, q.sources, k)
    result.reciprocal_rank = metrics.reciprocal_rank(ans.chunks, q.sources)

    # Correctness: refusals are scored without a judge call.
    if q.type == "out_of_corpus":
        result.correctness = 0.0 if ans.found else 1.0
        result.correctness_reason = "answered" if ans.found else "correctly refused"
    elif not ans.found:
        result.correctness, result.correctness_reason = 0.0, "refused"
    else:
        verdict = judge.judge_correctness(q.question, q.expected_answer, ans.text, **judge_kwargs)
        result.correctness = judge.CORRECTNESS_SCORE[verdict.verdict]
        result.correctness_reason = f"{verdict.verdict}: {verdict.reason}"

    # Faithfulness only applies when the system actually answered.
    if ans.found:
        verdict = judge.judge_faithfulness(ans.text, ans.chunks, **judge_kwargs)
        result.faithfulness = judge.faithfulness_score(verdict)
        result.unsupported_claims = verdict.unsupported_claims
    return result


def evaluate_retrieval(
    conn: psycopg.Connection, q: EvalQuestion, retrieve: RetrieveFn = search
) -> QuestionResult:
    """Score retrieval only (recall, MRR): no answer model and no judge, so it is cheap to run."""
    result = QuestionResult(
        id=q.id, type=q.type, question=q.question, expected_answer=q.expected_answer
    )
    start = time.perf_counter()
    chunks = retrieve(conn, q.question)
    result.latency_s = round(time.perf_counter() - start, 3)
    result.retrieved = [f"{c.chunk_id} p{c.pages}" for c in chunks]
    result.recall = metrics.recall_at_k(chunks, q.sources, get_settings().retrieval_top_k)
    result.reciprocal_rank = metrics.reciprocal_rank(chunks, q.sources)
    return result


def summarize(results: list[QuestionResult]) -> Summary:
    """Average each metric over the questions it applies to; errored questions are excluded."""
    ok = [r for r in results if r.error is None]
    return Summary(
        n=len(results),
        errors=len(results) - len(ok),
        recall=metrics.mean([r.recall for r in ok]),
        mrr=metrics.mean([r.reciprocal_rank for r in ok]),
        correctness=metrics.mean([r.correctness for r in ok]),
        faithfulness=metrics.mean([r.faithfulness for r in ok]),
        p50_latency_s=metrics.p50([r.latency_s for r in ok if r.latency_s is not None]),
        cost_per_query=metrics.mean([r.cost_usd for r in ok]),
        tokens_per_query=metrics.mean([float(r.input_tokens + r.output_tokens) for r in ok]),
    )


def _fmt(value: float | None, pattern: str) -> str:
    return "n/a" if value is None else pattern.format(value)


def results_row(date: str, name: str, s: Summary, notes: str = "") -> str:
    """One markdown table row for evals/results.md."""
    cost = f"{_fmt(s.cost_per_query, '${:.4f}')} ({_fmt(s.tokens_per_query, '{:,.0f}')} tok)"
    auto_notes = f"n={s.n}" + (f", {s.errors} errors" if s.errors else "")
    cells = [
        date,
        name,
        _fmt(s.recall, "{:.2f}"),
        _fmt(s.mrr, "{:.2f}"),
        _fmt(s.correctness, "{:.2f}"),
        _fmt(s.faithfulness, "{:.2f}"),
        _fmt(s.p50_latency_s, "{:.1f}s"),
        cost,
        "; ".join(filter(None, [auto_notes, notes])),
    ]
    return "| " + " | ".join(cells) + " |"
