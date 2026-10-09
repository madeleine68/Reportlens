from typing import Any

import pytest

from reportlens.core.llm import LLMResponse
from reportlens.generation.answer import (
    answer_question,
    build_messages,
    format_pages,
    parse_citations,
)
from reportlens.generation.prompts import NOT_FOUND
from reportlens.retrieval.models import RetrievedChunk


def _chunk(i: int, company: str = "rbc", year: int = 2023, pages: list[int] | None = None):
    return RetrievedChunk(
        chunk_id=f"{company}-{year}-{i:05d}",
        company=company,
        year=year,
        section=f"Section {i}",
        pages=pages or [10 + i],
        text=f"text of chunk {i}",
        score=0.9 - i / 10,
    )


CHUNKS = [_chunk(0, pages=[52, 53]), _chunk(1, company="td", year=2022), _chunk(2)]


class FakeChat:
    """Stands in for llm.chat: records the call and returns a canned reply."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    def __call__(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        self.calls.append({"messages": messages, **kwargs})
        return LLMResponse(
            text=self.reply, model="main-model", input_tokens=900, output_tokens=40, latency_s=1.2
        )


def test_prompt_numbers_sources_with_citation_headers() -> None:
    system, user = build_messages("What was net income?", CHUNKS)

    assert system["role"] == "system" and NOT_FOUND in system["content"]
    assert "[1] RBC 2023, pages 52, 53, section: Section 0\ntext of chunk 0" in user["content"]
    assert "[2] TD 2022, page 11, section: Section 1\ntext of chunk 1" in user["content"]
    assert user["content"].endswith("Question: What was net income?")


def test_citations_follow_order_of_first_use_without_duplicates() -> None:
    citations = parse_citations("Net income rose [3]. It was $16.2B [1][3]. Also [1, 2].", CHUNKS)
    assert [(c.number, c.chunk_id, c.pages) for c in citations] == [
        (3, "rbc-2023-00002", [12]),
        (1, "rbc-2023-00000", [52, 53]),
        (2, "td-2022-00001", [11]),
    ]


def test_citations_to_missing_sources_are_dropped() -> None:
    citations = parse_citations("Claim [1]. Invented [7]. Zero [0].", CHUNKS)
    assert [c.number for c in citations] == [1]


def test_answer_uses_main_model_and_returns_cited_sources() -> None:
    chat = FakeChat("RBC's net income was $16.2 billion in 2023 [1].")
    result = answer_question(None, "q?", retrieve=lambda conn, q: CHUNKS, chat=chat)

    assert chat.calls[0]["tier"] == "main"
    assert result.found is True
    assert [c.chunk_id for c in result.citations] == ["rbc-2023-00000"]
    assert result.chunks == CHUNKS
    assert (result.input_tokens, result.output_tokens, result.latency_s) == (900, 40, 1.2)


def test_full_width_markers_are_normalized_and_cited() -> None:
    chat = FakeChat("Bell had 10.29 million mobile phone subscribers \u30102\u3011.")
    result = answer_question(None, "q?", retrieve=lambda conn, q: CHUNKS, chat=chat)

    assert result.text == "Bell had 10.29 million mobile phone subscribers [2]."
    assert [c.number for c in result.citations] == [2]


def test_dagger_markers_are_normalized_and_cited() -> None:
    chat = FakeChat("Net income fell 6% [3\u2020L13-L16][1\u2020L1-L4].")
    result = answer_question(None, "q?", retrieve=lambda conn, q: CHUNKS, chat=chat)

    assert result.text == "Net income fell 6% [3][1]."
    assert [c.number for c in result.citations] == [3, 1]


@pytest.mark.parametrize("reply", [NOT_FOUND, "I couldn’t find this in the annual reports."])
def test_refusal_is_marked_not_found_with_no_citations(reply: str) -> None:
    result = answer_question(
        None, "q?", retrieve=lambda conn, q: CHUNKS, chat=FakeChat(reply + " [1]")
    )
    assert result.found is False
    assert result.citations == []


def test_no_retrieved_chunks_skips_the_llm() -> None:
    chat = FakeChat("should not be used")
    result = answer_question(None, "q?", retrieve=lambda conn, q: [], chat=chat)

    assert chat.calls == []
    assert (result.text, result.found, result.model) == (NOT_FOUND, False, None)


def test_format_pages() -> None:
    assert format_pages([4]) == "page 4"
    assert format_pages([4, 5]) == "pages 4, 5"
