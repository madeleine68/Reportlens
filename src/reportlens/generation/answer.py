"""Answer a question from retrieved chunks, with [n] citation markers mapped back to pages."""

import re
from collections.abc import Callable

import psycopg
from pydantic import BaseModel

from reportlens.core import llm
from reportlens.core.logging import get_logger
from reportlens.generation import prompts
from reportlens.retrieval.models import RetrievedChunk
from reportlens.retrieval.vector import vector_search

log = get_logger(__name__)

Retriever = Callable[[psycopg.Connection, str], list[RetrievedChunk]]
Chat = Callable[..., llm.LLMResponse]

# Matches [1] and grouped markers like [1, 3].
_MARKER = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
# Some models (e.g. gpt-oss) write full-width brackets like 【1】; normalize them to [1].
_FULL_WIDTH_BRACKETS = str.maketrans({"\u3010": "[", "\u3011": "]"})


class Citation(BaseModel):
    """A source the answer actually cited."""

    number: int  # the [n] used in the answer text
    chunk_id: str
    company: str
    year: int
    pages: list[int]


class Answer(BaseModel):
    """The answer plus everything needed to show sources and measure cost."""

    question: str
    text: str
    found: bool  # False when the model (or retrieval) found nothing to answer from
    citations: list[Citation]
    chunks: list[RetrievedChunk]  # everything retrieved, cited or not
    model: str | None
    input_tokens: int
    output_tokens: int
    latency_s: float


def format_pages(pages: list[int]) -> str:
    """'page 4' or 'pages 4, 5'."""
    label = "page" if len(pages) == 1 else "pages"
    return f"{label} {', '.join(map(str, pages))}"


def build_messages(question: str, chunks: list[RetrievedChunk]) -> list[llm.Message]:
    """System rules plus the numbered sources and the question."""
    sources = "\n\n".join(
        prompts.SOURCE_HEADER.format(
            number=i,
            company=chunk.company.upper(),
            year=chunk.year,
            pages=format_pages(chunk.pages),
            section=chunk.section or "unknown",
        )
        + "\n"
        + chunk.text
        for i, chunk in enumerate(chunks, start=1)
    )
    return [
        {"role": "system", "content": prompts.ANSWER_SYSTEM},
        {"role": "user", "content": prompts.ANSWER_USER.format(sources=sources, question=question)},
    ]


def parse_citations(text: str, chunks: list[RetrievedChunk]) -> list[Citation]:
    """Turn [n] markers into citations in order of first use, dropping markers with no source."""
    citations: list[Citation] = []
    seen: set[int] = set()
    for match in _MARKER.finditer(text):
        for number in (int(n) for n in match.group(1).split(",")):
            if number in seen:
                continue
            seen.add(number)
            if not 1 <= number <= len(chunks):
                log.warning("answer.invalid_citation number=%d sources=%d", number, len(chunks))
                continue
            chunk = chunks[number - 1]
            citations.append(
                Citation(
                    number=number,
                    chunk_id=chunk.chunk_id,
                    company=chunk.company,
                    year=chunk.year,
                    pages=chunk.pages,
                )
            )
    return citations


def is_not_found(text: str) -> bool:
    """True if the reply is the fixed refusal sentence (tolerating curly apostrophes)."""
    return prompts.NOT_FOUND.lower() in text.replace("’", "'").lower()


def answer_question(
    conn: psycopg.Connection,
    question: str,
    retrieve: Retriever = vector_search,
    chat: Chat = llm.chat,
) -> Answer:
    """Retrieve chunks, ask the main LLM to answer from them, and attach citations."""
    chunks = retrieve(conn, question)
    if not chunks:
        return Answer(
            question=question,
            text=prompts.NOT_FOUND,
            found=False,
            citations=[],
            chunks=[],
            model=None,
            input_tokens=0,
            output_tokens=0,
            latency_s=0.0,
        )

    reply = chat(build_messages(question, chunks), tier="main")
    text = reply.text.translate(_FULL_WIDTH_BRACKETS).strip()
    found = not is_not_found(text)
    return Answer(
        question=question,
        text=text,
        found=found,
        citations=parse_citations(text, chunks) if found else [],
        chunks=chunks,
        model=reply.model,
        input_tokens=reply.input_tokens,
        output_tokens=reply.output_tokens,
        latency_s=reply.latency_s,
    )
