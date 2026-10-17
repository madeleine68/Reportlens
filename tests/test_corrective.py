from pathlib import Path

import pytest
from pydantic import BaseModel

from reportlens.agents.corrective import (
    Grade,
    Rewrite,
    Tools,
    answer_corrective,
    grade_chunks,
    grade_messages,
    rewrite_query,
)
from reportlens.core.config import get_settings
from reportlens.generation import prompts
from reportlens.generation.answer import Answer
from reportlens.ingest.graph import FactCache
from reportlens.retrieval.models import RetrievedChunk

KNOWN = (["bce", "rbc", "scotiabank"], [2021, 2022, 2023, 2024])


def _chunk(chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, company="rbc", year=2023, section=None, pages=[4], text=chunk_id, score=1
    )


def _answer(question: str, text: str, chunks: list[RetrievedChunk]) -> Answer:
    return Answer(
        question=question,
        text=text,
        found=text != prompts.NOT_FOUND,
        citations=[],
        chunks=chunks,
        model="m",
        input_tokens=10,
        output_tokens=1,
        latency_s=0.1,
    )


class Recorder:
    """Stub tools: search results per query and grades per round, recording every call."""

    def __init__(self, results: dict[str, list[str]], grades: list[Grade], web: bool = False):
        self.results, self.grades, self.calls = results, grades, []
        self.web = web

    def tools(self) -> Tools:
        return Tools(
            retrieve=self.retrieve,
            grade=self.grade,
            rewrite=self.rewrite,
            generate=self.generate,
            web=self.web_answer if self.web else None,
        )

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        self.calls.append(f"retrieve {query}")
        return [_chunk(c) for c in self.results.get(query, [])]

    def grade(self, question: str, chunks: list[RetrievedChunk]) -> Grade:
        self.calls.append("grade")
        return self.grades.pop(0)

    def rewrite(self, question: str, missing: str) -> str:
        self.calls.append(f"rewrite ({missing})")
        return "better query"

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> Answer:
        self.calls.append("generate " + ",".join(c.chunk_id for c in chunks))
        return _answer(question, "the answer [1]", chunks)

    def web_answer(self, question: str) -> Answer:
        self.calls.append("web")
        return _answer(question, f"{prompts.WEB_PREFIX} something", [])


def test_enough_relevant_chunks_are_answered_from_alone() -> None:
    rec = Recorder({"q": ["a", "b", "c"]}, [Grade(relevant=[1, 3], enough=True)])
    result = answer_corrective("q", rec.tools())
    assert rec.calls == ["retrieve q", "grade", "generate a,c"]
    assert result.path == ["retrieve (3)", "grade 2/3 relevant, enough", "generate"]


def test_not_enough_rewrites_once_and_keeps_chunks_from_both_rounds() -> None:
    rec = Recorder(
        {"q": ["a", "b"], "better query": ["b", "d"]},
        [Grade(relevant=[1], enough=False, missing="BCE"), Grade(relevant=[1, 2], enough=True)],
    )
    result = answer_corrective("q", rec.tools())
    assert rec.calls == [
        "retrieve q",
        "grade",
        "rewrite (BCE)",
        "retrieve better query",
        "grade",
        "generate a,b,d",
    ]
    assert result.path[2] == "rewrite: better query" and result.answer.found


def test_partial_chunks_after_the_rewrite_still_get_an_answer() -> None:
    rec = Recorder(
        {"q": ["a"], "better query": ["b"]},
        [Grade(relevant=[1], enough=False), Grade(relevant=[], enough=False)],
    )
    answer_corrective("q", rec.tools())
    assert rec.calls[-1] == "generate a"


def test_nothing_relevant_refuses_when_web_search_is_off() -> None:
    rec = Recorder({"q": ["a"]}, [Grade(relevant=[], enough=False)])
    result = answer_corrective("q", rec.tools())
    assert rec.calls == ["retrieve q", "grade", "rewrite ()", "retrieve better query"]
    assert result.path[-2:] == ["grade (nothing found)", "refuse"]
    assert (result.answer.text, result.answer.found) == (prompts.NOT_FOUND, False)


def test_nothing_relevant_goes_to_the_web_when_it_is_on() -> None:
    nothing = [Grade(relevant=[], enough=False)] * 2
    rec = Recorder({"q": ["a"], "better query": ["b"]}, nothing, web=True)
    result = answer_corrective("q", rec.tools())
    assert rec.calls[-1] == "web" and result.answer.text.startswith(prompts.WEB_PREFIX)


def test_a_failed_grade_keeps_every_chunk() -> None:
    rec = Recorder({"q": ["a", "b"]}, [])  # popping from an empty list raises
    result = answer_corrective("q", rec.tools())
    assert rec.calls[-1] == "generate a,b" and "grade failed: kept all" in result.path


def test_a_failed_rewrite_searches_again_with_the_same_query() -> None:
    rec = Recorder({"q": ["a"]}, [Grade(relevant=[1], enough=False)] * 2)
    tools = rec.tools()

    def broken(question: str, missing: str) -> str:
        raise TimeoutError

    tools.rewrite = broken
    result = answer_corrective("q", tools)
    assert "rewrite failed" in result.path and rec.calls[-1] == "generate a"


def test_the_step_limit_stops_with_a_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRAG_MAX_REWRITES", "100")
    monkeypatch.setenv("AGENT_MAX_STEPS", "5")
    get_settings.cache_clear()
    try:
        rec = Recorder({}, [])
        result = answer_corrective("q", rec.tools())
    finally:
        get_settings.cache_clear()
    assert result.path == ["step limit reached"] and not result.answer.found


def _structured(reply: BaseModel):
    calls: list[list] = []

    def structured(messages: list, schema: type[BaseModel], tier: str, **kwargs: object):
        assert tier == "fast" and isinstance(reply, schema)
        calls.append(messages)
        return reply

    return structured, calls


def test_grades_drop_numbers_out_of_range_and_are_cached(tmp_path: Path) -> None:
    structured, calls = _structured(Grade(relevant=[3, 1, 9, 1], enough=True))
    cache = FactCache(tmp_path / "c.json")
    chunks = [_chunk("a"), _chunk("b"), _chunk("c")]
    first = grade_chunks("q", chunks, cache, structured)
    again = grade_chunks("q", chunks, cache, structured)
    assert first.relevant == again.relevant == [1, 3] and len(calls) == 1


def test_grade_prompt_numbers_chunks_with_company_year_and_pages() -> None:
    user = grade_messages("q", [_chunk("a"), _chunk("b")])[1]["content"]
    assert "[1] RBC 2023, page 4\na" in user and "[2] RBC 2023, page 4\nb" in user


def test_rewrite_names_what_is_missing() -> None:
    structured, calls = _structured(Rewrite(query=" BCE net earnings 2023 "))
    assert rewrite_query("q", "BCE figure", KNOWN, None, structured) == "BCE net earnings 2023"
    assert "BCE figure" in calls[0][1]["content"] and "2021-2024" in calls[0][0]["content"]
