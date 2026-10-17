from pathlib import Path

import pytest
from pydantic import BaseModel

import reportlens.agents.routed as routed_module
from reportlens.agents.routed import ask
from reportlens.agents.router import Route
from reportlens.agents.self_rag import (
    CheckTools,
    SupportCheck,
    check_messages,
    check_support,
    revise_answer,
    revise_messages,
    self_check,
    source_label,
)
from reportlens.core import llm
from reportlens.core.config import get_settings
from reportlens.generation import prompts
from reportlens.generation.answer import Answer
from reportlens.ingest.graph import FactCache
from reportlens.retrieval.models import RetrievedChunk


def _chunk(chunk_id: str, company: str = "rbc", year: int = 2023, pages=(4,)) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        company=company,
        year=year,
        section="Results",
        pages=list(pages),
        text=f"text of {chunk_id}",
        score=1,
    )


def _answer(text: str = "Net income was $14.9B [1].", chunks=None, found: bool = True) -> Answer:
    return Answer(
        question="What was RBC's net income in 2023?",
        text=text,
        found=found,
        citations=[],
        chunks=[_chunk("a"), _chunk("b")] if chunks is None else chunks,
        model="m",
        input_tokens=100,
        output_tokens=10,
        latency_s=1.0,
    )


class Recorder:
    """Stub tools: verdicts in turn, recording every call."""

    def __init__(self, verdicts: list[SupportCheck]):
        self.verdicts, self.calls = verdicts, []

    def tools(self) -> CheckTools:
        return CheckTools(check=self.check, revise=self.revise)

    def check(self, answer: Answer) -> SupportCheck:
        self.calls.append(f"check {answer.text}")
        return self.verdicts.pop(0)

    def revise(self, answer: Answer, claims: list[str]) -> Answer:
        self.calls.append(f"revise {claims}")
        return answer.model_copy(update={"text": "revised [1]"})


BAD = SupportCheck(supported=False, unsupported_claims=["$14.9B"])
GOOD = SupportCheck(supported=True)


def test_a_supported_answer_is_returned_as_is() -> None:
    rec = Recorder([GOOD])
    result = self_check(_answer(), rec.tools())
    assert (result.supported, result.path, len(rec.calls)) == (True, ["supported"], 1)


def test_an_unsupported_answer_is_revised_once_and_checked_again() -> None:
    rec = Recorder([BAD, GOOD])
    result = self_check(_answer(), rec.tools())
    assert rec.calls == [
        "check Net income was $14.9B [1].",
        "revise ['$14.9B']",
        "check revised [1]",
    ]
    assert (result.answer.text, result.supported) == ("revised [1]", True)
    assert result.path == ["1 unsupported", "revise", "supported"]


def test_still_unsupported_after_the_revision_is_returned_marked() -> None:
    rec = Recorder([BAD, BAD])
    result = self_check(_answer(), rec.tools())
    assert (result.answer.text, result.supported) == ("revised [1]", False)
    assert result.path == ["1 unsupported", "revise", "1 unsupported"]


def test_supported_true_with_claims_listed_counts_as_unsupported() -> None:
    rec = Recorder([SupportCheck(supported=True, unsupported_claims=["x"]), GOOD])
    assert self_check(_answer(), rec.tools()).path[0] == "1 unsupported"


@pytest.mark.parametrize(
    "answer", [_answer(prompts.NOT_FOUND, found=False), _answer("CET1 is a ratio.", chunks=[])]
)
def test_refusals_and_answers_without_sources_are_not_checked(answer: Answer) -> None:
    rec = Recorder([])
    result = self_check(answer, rec.tools())
    assert (result.supported, rec.calls) == (None, [])


def test_a_failed_check_keeps_the_answer() -> None:
    rec = Recorder([])  # popping from an empty list raises
    result = self_check(_answer(), rec.tools())
    assert (result.answer.text, result.supported, result.path) == (
        "Net income was $14.9B [1].",
        None,
        ["check failed"],
    )


def test_a_failed_revision_keeps_the_earlier_answer() -> None:
    rec = Recorder([BAD, BAD])
    tools = rec.tools()

    def broken(answer: Answer, claims: list[str]) -> Answer:
        raise TimeoutError

    tools.revise = broken
    result = self_check(_answer(), tools)
    assert result.answer.text == "Net income was $14.9B [1]." and "revise failed" in result.path
    assert result.supported is False


def test_sources_are_labelled_like_the_answer_saw_them() -> None:
    assert source_label(_chunk("a", pages=(4, 5))) == "RBC 2023, pages 4, 5"
    web = RetrievedChunk(
        chunk_id="https://x.com", company="web", year=0, section="X", pages=[], text="t", score=0
    )
    assert source_label(web) == "X (https://x.com)"
    user = check_messages(_answer())[1]["content"]
    assert "[1] RBC 2023, page 4\ntext of a" in user and "Answer: Net income" in user


def test_check_is_cached(tmp_path: Path) -> None:
    calls: list[object] = []

    def structured(messages: list, schema: type[BaseModel], tier: str, **kwargs: object):
        assert (schema, tier) == (SupportCheck, "fast")
        calls.append(messages)
        return GOOD

    cache = FactCache(tmp_path / "c.json")
    check_support(_answer(), cache, structured)
    check_support(_answer(), cache, structured)
    assert len(calls) == 1


def test_revision_lists_the_claims_reparses_citations_and_adds_costs() -> None:
    def chat(messages: list[llm.Message], tier: str) -> llm.LLMResponse:
        assert tier == "main" and "- $14.9B" in messages[1]["content"]
        return llm.LLMResponse(
            text="Net income was $14,866 million [2].",
            model="m",
            input_tokens=50,
            output_tokens=5,
            latency_s=0.5,
        )

    revised = revise_answer(_answer(), ["$14.9B"], chat)
    assert [c.chunk_id for c in revised.citations] == ["b"]
    assert (revised.input_tokens, revised.output_tokens, revised.latency_s) == (150, 15, 1.5)
    assert "Earlier answer: Net income was $14.9B" in revise_messages(_answer(), [])[1]["content"]


def test_ask_checks_the_answer_only_when_self_rag_is_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        routed_module, "route_question", lambda q, known, cache: Route(source="vector", reason="r")
    )
    answerers = {"vector": lambda q: _answer()}
    off = ask("q", (["rbc"], [2023]), answerers, check_tools=Recorder([]).tools())
    assert (off.checks, off.supported) == ([], None)

    monkeypatch.setenv("SELF_RAG", "true")
    get_settings.cache_clear()
    try:
        on = ask("q", (["rbc"], [2023]), answerers, check_tools=Recorder([BAD, GOOD]).tools())
    finally:
        get_settings.cache_clear()
    assert (on.answer.text, on.checks, on.supported) == (
        "revised [1]",
        ["1 unsupported", "revise", "supported"],
        True,
    )
