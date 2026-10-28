"""The routed-answer LangGraph graph: route, ask the picked source, fall back, check."""

import httpx
import openai
import pytest

from reportlens.agents.graph import FALLBACK, run_routed
from reportlens.agents.router import Route
from reportlens.agents.self_rag import CheckedAnswer
from reportlens.core import llm
from reportlens.generation.answer import Answer


def answer(text: str, found: bool = True, tokens: int = 100) -> Answer:
    return Answer(
        question="q",
        text=text,
        found=found,
        citations=[],
        chunks=[],
        model="m",
        input_tokens=tokens,
        output_tokens=10,
        latency_s=1.0,
    )


def routed_to(source: str):
    return lambda question: Route(source=source, reason="because")


def test_found_answer_skips_the_fallback() -> None:
    result = run_routed(
        "q", routed_to("sql"), {"sql": lambda q: answer("rows"), FALLBACK: lambda q: answer("text")}
    )
    assert (result.answer.text, result.path) == ("rows", ["sql"])


def test_nothing_found_falls_back_and_keeps_the_first_attempts_cost() -> None:
    answerers = {
        "sql": lambda q: answer("none", found=False, tokens=300),
        FALLBACK: lambda q: answer("text", tokens=500),
    }
    result = run_routed("q", routed_to("sql"), answerers)
    assert (result.answer.text, result.path) == ("text", ["sql", FALLBACK])
    assert result.answer.input_tokens == 800 and result.answer.latency_s == 2.0


def test_broken_tool_falls_back_and_is_marked_in_the_path() -> None:
    def broken(question: str) -> Answer:
        raise ConnectionError("neo4j down")

    result = run_routed(
        "q", routed_to("graph"), {"graph": broken, FALLBACK: lambda q: answer("text")}
    )
    assert result.path == ["graph (error)", FALLBACK]


def test_rate_limit_stops_the_question_instead_of_falling_back() -> None:
    def out_of_quota(question: str) -> Answer:
        request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        raise openai.RateLimitError(
            "per day", response=httpx.Response(429, request=request), body=None
        )

    with pytest.raises(llm.RateLimitError):
        run_routed("q", routed_to("sql"), {"sql": out_of_quota, FALLBACK: lambda q: answer("x")})


def test_source_that_is_not_on_goes_straight_to_text_search() -> None:
    result = run_routed("q", routed_to("web"), {FALLBACK: lambda q: answer("text")})
    assert result.path == [FALLBACK] and result.route.source == "web"


def test_check_runs_after_either_branch_only_when_given() -> None:
    def check(ans: Answer) -> CheckedAnswer:
        return CheckedAnswer(answer=answer("checked"), supported=True, path=["supported"])

    answerers = {"sql": lambda q: answer("rows"), FALLBACK: lambda q: answer("text")}
    for source in ("sql", FALLBACK):
        result = run_routed("q", routed_to(source), answerers, check)
        assert (result.answer.text, result.checks, result.supported) == (
            "checked",
            ["supported"],
            True,
        )
    unchecked = run_routed("q", routed_to("sql"), answerers)
    assert (unchecked.checks, unchecked.supported) == ([], None)


def test_route_and_steps_are_streamed(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "reportlens.agents.graph.emit", lambda kind, **data: sent.append((kind, data))
    )
    run_routed(
        "q",
        routed_to("sql"),
        {"sql": lambda q: answer("none", found=False), FALLBACK: lambda q: answer("text")},
    )
    assert sent == [
        ("route", {"source": "sql", "reason": "because"}),
        ("step", {"stage": "routed", "text": "asking sql"}),
        ("step", {"stage": "routed", "text": "asking vector"}),
    ]
