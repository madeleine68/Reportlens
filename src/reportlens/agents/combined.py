"""Multi-source answers (Phase 4): rows from SQL or the graph, plus the text around them.

The router sends each question to one source. For the questions it sends to SQL or the graph,
the evals showed that the report text often holds what the rows lack: the exact headcount (the
table rounds it), the context the expected answer mentions, or a chart's breakdown. With
MULTI_SOURCE=true those questions get both: the routed tool's rows and the top text chunks,
numbered as one list of sources and answered in a single main-LLM call. That call replaces the
tool's own answer call, so a question still costs one main-model call, with longer input.
"""

from collections.abc import Callable
from typing import Any, Literal

from reportlens.agents.events import emit
from reportlens.core import llm
from reportlens.core.logging import get_logger
from reportlens.generation import prompts
from reportlens.generation.answer import (
    Answer,
    build_messages,
    is_not_found,
    normalize_markers,
    parse_citations,
)
from reportlens.retrieval.models import RetrievedChunk
from reportlens.sources.sql_tool import row_text

log = get_logger(__name__)

RowSource = Literal["sql", "graph"]
# (query that ran, columns, rows, error): the tools' query_for, with its connection bound.
FetchRows = Callable[[str], tuple[str | None, list[str], list[list[Any]], str | None]]
SearchText = Callable[[str], list[RetrievedChunk]]
Chat = Callable[..., llm.LLMResponse]

ROW_SECTIONS: dict[RowSource, str] = {"sql": "financial table", "graph": "knowledge graph"}


def rows_as_chunks(
    source: RowSource, columns: list[str], rows: list[list[Any]]
) -> list[RetrievedChunk]:
    """Rows as sources: each keeps its company, report year and page, so it can be cited."""
    index = {name: i for i, name in enumerate(columns)}
    page_col = index.get("source_page", index.get("page"))
    return [
        RetrievedChunk(
            chunk_id=f"{source}-row-{i}",
            company=row[index["company"]] if "company" in index else "",
            year=row[index["report_year"]] if "report_year" in index else 0,
            section=ROW_SECTIONS[source],
            pages=[row[page_col]] if page_col is not None else [],
            text=row_text(columns, row),
            score=0.0,
        )
        for i, row in enumerate(rows, start=1)
    ]


def answer_combined(
    question: str,
    source: RowSource,
    fetch_rows: FetchRows,
    search_text: SearchText,
    chat: Chat = llm.chat,
) -> Answer:
    """Answer from the tool's rows and the text search's chunks together, citing either."""
    _, columns, rows, error = fetch_rows(question)
    if error:
        log.info("combined.no_rows source=%s error=%s", source, error)
    chunks = rows_as_chunks(source, columns, rows) + search_text(question)
    emit(
        "step",
        stage="combined",
        text=f"{len(rows)} {source} rows + {len(chunks) - len(rows)} text chunks",
    )
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

    reply = chat(build_messages(question, chunks, prompts.COMBINED_SYSTEM), tier="main")
    text = normalize_markers(reply.text).strip()
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
