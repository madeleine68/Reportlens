from pathlib import Path

import pytest
from pydantic import BaseModel

import reportlens.agents.routed as routed_module
from reportlens.agents.corrective import CorrectiveAnswer
from reportlens.agents.routed import (
    RoutedAnswer,
    answer_routed,
    ask,
    default_answerers,
    no_lookup_answer,
)
from reportlens.agents.router import Route, route_question, router_messages
from reportlens.core import llm
from reportlens.core.config import get_settings
from reportlens.generation import prompts
from reportlens.generation.answer import Answer
from reportlens.ingest.graph import FactCache

KNOWN = (["bce", "rbc", "scotiabank"], [2021, 2022, 2023, 2024])


def _router(*sources: str):
    """Stub for the routing LLM: returns the given sources in turn, counting calls."""
    calls: list[list[llm.Message]] = []

    def structured(messages: list[llm.Message], schema: type[BaseModel], tier: str) -> Route:
        assert (schema, tier) == (Route, "fast")
        calls.append(messages)
        return Route(source=sources[len(calls) - 1], reason="stub")

    return structured, calls


def _answer(question: str, found: bool, text: str, tokens: int = 10) -> Answer:
    return Answer(
        question=question,
        text=text,
        found=found,
        citations=[],
        chunks=[],
        model="main-model",
        input_tokens=tokens,
        output_tokens=1,
        latency_s=0.5,
    )


def _answerers(found: dict[str, bool]) -> tuple[dict, list[str]]:
    """One stub answer function per source; `found` says which of them find an answer."""
    called: list[str] = []

    def make(source: str):
        def answer(question: str) -> Answer:
            called.append(source)
            return _answer(question, found[source], f"from {source}", tokens=100)

        return answer

    return {source: make(source) for source in found}, called


def test_router_prompt_names_the_sources_companies_and_years() -> None:
    system = router_messages("q", KNOWN)[0]["content"]
    assert "bce, rbc, scotiabank" in system and "2021-2024" in system
    for source in ("- sql:", "- graph:", "- vector:", "- web:", "- none:"):
        assert source in system


def test_routes_are_cached_per_question(tmp_path: Path) -> None:
    structured, calls = _router("sql", "graph")
    cache = FactCache(tmp_path / "routes.json")
    first = route_question("RBC net income 2023?", KNOWN, cache, structured)
    again = route_question("RBC net income 2023?", KNOWN, cache, structured)
    other = route_question("Who replaced RBC's CFO?", KNOWN, cache, structured)
    assert (first.source, again.source, other.source, len(calls)) == ("sql", "sql", "graph", 2)


def test_a_router_failure_routes_to_vector() -> None:
    def broken(*args: object, **kwargs: object) -> Route:
        raise ValueError("bad JSON")

    route = route_question("q", KNOWN, None, broken)
    assert route.source == "vector" and "ValueError" in route.reason


@pytest.mark.parametrize("source", ["vector", "sql", "graph", "none"])
def test_a_source_that_finds_the_answer_is_used_alone(source: str) -> None:
    answerers, called = _answerers({s: True for s in ("vector", "sql", "graph", "none")})
    routed = answer_routed("q", Route(source=source, reason="r"), answerers)
    assert (routed.path, called, routed.answer.text) == ([source], [source], f"from {source}")


def test_sql_finding_nothing_falls_back_to_vector_and_costs_add_up() -> None:
    answerers, called = _answerers({"vector": True, "sql": False})
    routed = answer_routed("Why did RBC's profit fall?", Route(source="sql", reason="r"), answerers)
    assert (routed.path, called) == (["sql", "vector"], ["sql", "vector"])
    assert routed.answer.text == "from vector"
    assert (routed.answer.input_tokens, routed.answer.latency_s) == (200, 1.0)


def test_vector_finding_nothing_is_the_final_answer() -> None:
    answerers, called = _answerers({"vector": False})
    routed = answer_routed("q", Route(source="vector", reason="r"), answerers)
    assert (routed.path, routed.answer.found) == (["vector"], False)


def test_a_source_without_a_tool_goes_to_vector() -> None:
    answerers, called = _answerers({"vector": True})  # e.g. Neo4j down: no graph answerer
    routed = answer_routed("q", Route(source="graph", reason="r"), answerers)
    assert (routed.path, called) == (["vector"], ["vector"])


def test_ask_routes_then_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    structured, _ = _router("graph")
    answerers, called = _answerers({"vector": True, "graph": True})
    monkeypatch.setattr(
        routed_module,
        "route_question",
        lambda question, known, cache: route_question(question, known, cache, structured),
    )
    routed = ask("Who replaced Nadine Ahn?", KNOWN, answerers)
    assert isinstance(routed, RoutedAnswer) and routed.route.source == "graph"
    assert called == ["graph"]


def test_no_lookup_answer_uses_the_main_model_without_sources() -> None:
    def chat(messages: list[llm.Message], tier: str) -> llm.LLMResponse:
        assert tier == "main" and "general" in messages[0]["content"]
        return llm.LLMResponse(
            text=" CET1 is a capital ratio. ",
            model="m",
            input_tokens=5,
            output_tokens=7,
            latency_s=0.1,
        )

    answer = no_lookup_answer("What is CET1?", chat)
    assert (answer.text, answer.found, answer.chunks) == ("CET1 is a capital ratio.", True, [])


def test_no_lookup_refusal_is_not_found() -> None:
    def chat(messages: list[llm.Message], tier: str) -> llm.LLMResponse:
        return llm.LLMResponse(
            text=prompts.NOT_FOUND, model="m", input_tokens=5, output_tokens=7, latency_s=0.1
        )

    assert not no_lookup_answer("Who won the 2023 Stanley Cup?", chat).found


def test_the_vector_route_uses_corrective_rag_unless_switched_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(routed_module, "default_tools", lambda conn, cache: "tools")
    monkeypatch.setattr(routed_module, "crag_cache", lambda: None)
    monkeypatch.setattr(
        routed_module,
        "answer_corrective",
        lambda q, tools: (
            calls.append(f"corrective {tools}")
            or CorrectiveAnswer(answer=_answer(q, True, "graded"), path=["generate"])
        ),
    )
    monkeypatch.setattr(
        routed_module,
        "answer_question",
        lambda conn, q: calls.append("plain") or _answer(q, True, "plain"),
    )
    assert default_answerers(None, None, None)["vector"]("q").text == "graded"
    monkeypatch.setenv("ROUTED_CORRECTIVE", "false")
    get_settings.cache_clear()
    try:
        assert default_answerers(None, None, None)["vector"]("q").text == "plain"
    finally:
        get_settings.cache_clear()
    assert calls == ["corrective tools", "plain"]
