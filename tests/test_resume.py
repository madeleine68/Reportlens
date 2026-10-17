import json
from pathlib import Path

import httpx
import openai
import pytest

from reportlens.agents.corrective import Tools, answer_corrective
from reportlens.agents.routed import answer_routed
from reportlens.agents.router import Route, route_question
from reportlens.core import llm
from reportlens.evals.resume import latest_run, load_finished
from reportlens.evals.runner import QuestionResult
from reportlens.retrieval.models import RetrievedChunk


def _rate_limit(message: str) -> openai.RateLimitError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return openai.RateLimitError(message, response=httpx.Response(429, request=request), body=None)


DAILY = _rate_limit("Rate limit reached for model `m` on tokens per day (TPD): Limit 200000")
MINUTE = _rate_limit("Rate limit reached for model `m` on tokens per minute (TPM): Limit 8000")


def _raise(err: Exception):
    def fn(*args: object, **kwargs: object):
        raise err

    return fn


def test_daily_limit_is_told_apart_from_the_per_minute_one() -> None:
    assert llm.is_daily_limit(DAILY)
    assert not llm.is_daily_limit(MINUTE) and not llm.is_daily_limit(ValueError("per day"))


def test_router_raises_rate_limits_instead_of_routing_to_vector() -> None:
    with pytest.raises(openai.RateLimitError):
        route_question("q", (["rbc"], [2023]), None, _raise(DAILY))


def test_routed_answer_raises_rate_limits_instead_of_falling_back() -> None:
    with pytest.raises(openai.RateLimitError):
        answer_routed(
            "q", Route(source="sql", reason="r"), {"sql": _raise(MINUTE), "vector": print}
        )


def test_corrective_grading_raises_rate_limits_instead_of_keeping_all_chunks() -> None:
    chunk = RetrievedChunk(
        chunk_id="a", company="rbc", year=2023, section=None, pages=[1], text="a", score=1
    )
    tools = Tools(
        retrieve=lambda q: [chunk],
        grade=_raise(DAILY),
        rewrite=lambda q, m: q,
        generate=_raise(AssertionError("must not answer")),
        web=None,
    )
    with pytest.raises(openai.RateLimitError):
        answer_corrective("q", tools)


def _result(qid: str, error: str | None = None) -> QuestionResult:
    return QuestionResult(id=qid, type="lookup", question=qid, expected_answer="x", error=error)


def test_resume_keeps_successes_and_their_routes_only(tmp_path: Path) -> None:
    older = tmp_path / "2026-10-16-routed.jsonl"
    newer = tmp_path / "2026-10-17-routed.jsonl"
    older.write_text(_result("q1").model_dump_json() + "\n")
    newer.write_text(
        _result("q1").model_dump_json() + "\n" + _result("q2", "boom").model_dump_json()
    )
    newer.with_suffix(".routes.json").write_text(json.dumps({"q1": "sql", "q2": "vector"}))
    (tmp_path / "2026-10-18-routed-plain.jsonl").write_text("")  # another experiment's name

    assert latest_run(tmp_path, "routed") == newer and latest_run(tmp_path, "nope") is None
    finished, routes = load_finished(newer)
    assert list(finished) == ["q1"] and routes == {"q1": "sql"}
