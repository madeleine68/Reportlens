"""Multi-source answers: SQL/graph rows and text chunks numbered together, in one answer call."""

from typing import Any

import pytest

from reportlens.agents import routed
from reportlens.agents.combined import answer_combined, rows_as_chunks
from reportlens.core.config import get_settings
from reportlens.core.llm import LLMResponse
from reportlens.generation.prompts import COMBINED_SYSTEM, NOT_FOUND
from reportlens.retrieval.models import RetrievedChunk

COLUMNS = ["company", "report_year", "metric", "value", "unit", "source_page"]
ROWS = [["bce", 2024, "employees", 40000, "count", 33]]
TEXT = [
    RetrievedChunk(
        chunk_id="bce-2024-00123",
        company="bce",
        year=2024,
        section="Our people",
        pages=[61],
        text="BCE had 40,390 employees at the end of 2024.",
        score=4.2,
    )
]


class FakeChat:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[list[dict[str, str]]] = []

    def __call__(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        self.calls.append(messages)
        return LLMResponse(
            text=self.reply, model="main-model", input_tokens=700, output_tokens=30, latency_s=1.0
        )


def fetch(rows: list[list[Any]], error: str | None = None):
    return lambda question: ("SELECT ...", COLUMNS if rows else [], rows, error)


def test_rows_become_citable_sources() -> None:
    [chunk] = rows_as_chunks("sql", COLUMNS, ROWS)
    assert chunk.chunk_id == "sql-row-1"
    assert (chunk.company, chunk.year, chunk.pages) == ("bce", 2024, [33])
    assert chunk.section == "financial table"
    assert chunk.text.startswith("company=bce, report_year=2024")
    assert "[1]" not in chunk.text  # numbering comes from the combined source list


def test_a_computed_row_without_a_page_still_becomes_a_source() -> None:
    # q043's retried query added a "difference" row with no year or page; it used to crash
    rows = [*ROWS, ["difference", None, "net_income", 5633, "CAD millions", None]]
    chunks = rows_as_chunks("sql", COLUMNS, rows)
    assert (chunks[1].company, chunks[1].year, chunks[1].pages) == ("difference", 0, [])


def test_rows_and_text_are_numbered_together_and_both_cited() -> None:
    chat = FakeChat("BCE had 40,390 employees [2]; the table rounds this to 40,000 [1].")
    answer = answer_combined("How many employees?", "sql", fetch(ROWS), lambda q: TEXT, chat)

    system, user = chat.calls[0]
    assert system["content"] == COMBINED_SYSTEM
    assert "[1] BCE 2024, page 33, section: financial table\ncompany=bce" in user["content"]
    assert "[2] BCE 2024, page 61, section: Our people\nBCE had 40,390" in user["content"]
    assert answer.found
    assert [c.chunk_id for c in answer.citations] == ["bce-2024-00123", "sql-row-1"]
    assert [c.chunk_id for c in answer.chunks] == ["sql-row-1", "bce-2024-00123"]


def test_no_rows_answers_from_text_alone() -> None:
    chat = FakeChat("BCE had 40,390 employees [1].")
    answer = answer_combined(
        "How many employees?", "sql", fetch([], error="no such metric"), lambda q: TEXT, chat
    )
    assert "financial table" not in chat.calls[0][1]["content"]
    assert answer.found and answer.citations[0].chunk_id == "bce-2024-00123"


def test_nothing_found_anywhere_refuses_without_an_llm_call() -> None:
    chat = FakeChat("unused")
    answer = answer_combined("q", "graph", fetch([]), lambda q: [], chat)
    assert not answer.found and answer.text == NOT_FOUND
    assert chat.calls == []


def test_refusal_has_no_citations() -> None:
    answer = answer_combined("q", "sql", fetch(ROWS), lambda q: TEXT, FakeChat(NOT_FOUND))
    assert not answer.found and answer.citations == []


@pytest.mark.parametrize("enabled", [True, False])
def test_setting_swaps_the_sql_answerer(enabled: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MULTI_SOURCE", str(enabled).lower())
    get_settings.cache_clear()
    calls: list[str] = []
    monkeypatch.setattr(routed, "answer_combined", lambda q, source, *a: calls.append(source))
    monkeypatch.setattr(routed, "answer_with_sql", lambda reader, q: calls.append("sql only"))
    monkeypatch.setattr(routed, "as_answer", lambda result: result)

    routed.default_answerers(None, None, None)["sql"]("How many employees?")
    assert calls == (["sql"] if enabled else ["sql only"])
