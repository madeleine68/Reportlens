"""Text-to-SQL over the financials and table_cells tables (Phase 3).

1. The fast LLM writes one SELECT, given the tables' columns, metrics and segments.
2. sources/sql_safety.py checks it and sets a row limit; it runs on a read-only connection.
3. If it is rejected or fails, the error goes back to the LLM for one more try.
4. The main LLM answers from the returned rows, citing them as [n]; each row carries its page.
"""

import time
from collections.abc import Callable
from typing import Any

import psycopg
from pydantic import BaseModel

from reportlens.core import llm
from reportlens.core.config import get_settings
from reportlens.core.logging import get_logger
from reportlens.generation import prompts
from reportlens.generation.answer import (
    CITATION_MARKER,
    Answer,
    Citation,
    is_not_found,
    normalize_markers,
)
from reportlens.ingest.metrics import METRICS, SEGMENTS
from reportlens.retrieval.models import RetrievedChunk
from reportlens.sources.sql_safety import UnsafeSQL, check_sql

log = get_logger(__name__)

SQL_ATTEMPTS = 2  # first try plus one retry with the error fed back


class SqlQuery(BaseModel):
    """What the SQL-writing LLM returns. An empty string means the tables can't answer."""

    sql: str


class SqlCitation(BaseModel):
    """A row the answer cited, and where its figure is printed."""

    number: int
    company: str
    year: int  # the report's fiscal year
    page: int


class RowsAnswer(BaseModel):
    """An answer written from query rows (SQL here, Cypher in sources/graph_tool.py)."""

    question: str
    text: str
    found: bool
    columns: list[str]
    rows: list[list[Any]]
    citations: list[SqlCitation]
    error: str | None  # why no query ran, if none did
    model: str | None
    input_tokens: int  # main-model tokens, as for text answers
    output_tokens: int
    latency_s: float


class SqlAnswer(RowsAnswer):
    """The answer plus the query and rows behind it."""

    sql: str | None  # the query that ran (after the safety check), if any


WriteSql = Callable[[list[llm.Message]], SqlQuery]
Chat = Callable[..., llm.LLMResponse]


def sql_messages(question: str) -> list[llm.Message]:
    """System prompt describing the tables, plus the question."""
    metrics = "\n".join(f"  - '{m.name}' ({m.unit}): {m.description}" for m in METRICS)
    segments = "; ".join(f"{c}: {', '.join(names)}" for c, names in SEGMENTS.items())
    system = prompts.SQL_SYSTEM.format(
        companies=", ".join(f"'{c}'" for c in SEGMENTS),
        metrics=metrics,
        segments=segments,
        row_limit=get_settings().sql_row_limit,
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": question}]


def write_sql(messages: list[llm.Message]) -> SqlQuery:
    """Ask the fast model for a query."""
    return llm.structured(messages, SqlQuery, tier="fast")


def run_sql(conn: psycopg.Connection, query: str) -> tuple[list[str], list[list[Any]]]:
    """Run a checked query on the read-only connection; (column names, rows)."""
    start = time.perf_counter()
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            columns = [d.name for d in cur.description or []]
            rows = [list(r) for r in cur.fetchmany(get_settings().sql_row_limit)]
    finally:
        conn.rollback()  # nothing to keep, and a failed query must not poison the next one
    log.info("sql.run rows=%d latency=%.2fs", len(rows), time.perf_counter() - start)
    return columns, rows


def query_for(
    conn: psycopg.Connection, question: str, write: WriteSql = write_sql
) -> tuple[str | None, list[str], list[list[Any]], str | None]:
    """(query that ran, columns, rows, error): write, check and run, with one retry on failure."""
    messages = sql_messages(question)
    error = None
    for _ in range(SQL_ATTEMPTS):
        proposed = write(messages).sql.strip()
        if not proposed:
            return None, [], [], "the SQL model said these tables can't answer the question"
        try:
            checked = check_sql(proposed, get_settings().sql_row_limit)
            columns, rows = run_sql(conn, checked)
            return checked, columns, rows, None
        except (UnsafeSQL, psycopg.Error) as err:
            error = str(err).strip()
            log.warning("sql.failed sql=%r error=%s", proposed, error)
            messages = [
                *messages,
                {"role": "assistant", "content": SqlQuery(sql=proposed).model_dump_json()},
                {"role": "user", "content": prompts.SQL_RETRY.format(error=error)},
            ]
    return None, [], [], error


def format_row(number: int, columns: list[str], row: list[Any]) -> str:
    """One row as '[n] column=value, ...' on a single line (line breaks in values become spaces)."""
    pairs = ", ".join(f"{c}={' '.join(str(v).split())}" for c, v in zip(columns, row, strict=True))
    return f"[{number}] {pairs}"


def format_rows(columns: list[str], rows: list[list[Any]]) -> str:
    """Rows as numbered lines of column=value pairs, the way the answer model cites them."""
    return "\n".join(format_row(i, columns, row) for i, row in enumerate(rows, start=1))


def row_citations(text: str, columns: list[str], rows: list[list[Any]]) -> list[SqlCitation]:
    """Citations for the [n] markers in the answer, using each row's company, report and page."""
    index = {name: i for i, name in enumerate(columns)}
    page_col = index.get("source_page", index.get("page"))
    if "company" not in index or "report_year" not in index or page_col is None:
        return []
    citations, seen = [], set()
    for match in CITATION_MARKER.finditer(text):
        for number in (int(n) for n in match.group(1).split(",")):
            if number in seen or not 1 <= number <= len(rows):
                continue
            seen.add(number)
            row = rows[number - 1]
            citations.append(
                SqlCitation(
                    number=number,
                    company=row[index["company"]],
                    year=row[index["report_year"]],
                    page=row[page_col],
                )
            )
    return citations


def answer_with_sql(
    conn: psycopg.Connection,
    question: str,
    write: WriteSql = write_sql,
    chat: Chat = llm.chat,
) -> SqlAnswer:
    """Answer a question from the financial tables, citing the rows used."""
    query, columns, rows, error = query_for(conn, question, write)
    empty = SqlAnswer(
        question=question,
        text=prompts.NOT_FOUND,
        found=False,
        sql=query,
        columns=columns,
        rows=rows,
        citations=[],
        error=error,
        model=None,
        input_tokens=0,
        output_tokens=0,
        latency_s=0.0,
    )
    if not rows:
        return empty
    return empty.model_copy(update=answer_from_rows(question, query, columns, rows, chat))


def answer_from_rows(
    question: str, query: str | None, columns: list[str], rows: list[list[Any]], chat: Chat
) -> dict[str, Any]:
    """The main LLM's answer from the rows, as RowsAnswer fields (text, citations, tokens...)."""
    user = prompts.SQL_ANSWER_USER.format(
        sql=query, rows=format_rows(columns, rows), question=question
    )
    reply = chat(
        [
            {"role": "system", "content": prompts.SQL_ANSWER_SYSTEM},
            {"role": "user", "content": user},
        ],
        tier="main",
    )
    text = normalize_markers(reply.text).strip()
    found = not is_not_found(text)
    return {
        "text": text,
        "found": found,
        "citations": row_citations(text, columns, rows) if found else [],
        "model": reply.model,
        "input_tokens": reply.input_tokens,
        "output_tokens": reply.output_tokens,
        "latency_s": reply.latency_s,
    }


def as_answer(result: RowsAnswer) -> Answer:
    """A SQL or graph answer in the shape the eval runner scores: each row becomes a pseudo-chunk.

    A row's company, report and page let recall check it against the expected pages, and its
    column=value text is what the faithfulness judge reads as the source.
    """
    index = {name: i for i, name in enumerate(result.columns)}
    page_col = index.get("source_page", index.get("page"))
    chunks = [
        RetrievedChunk(
            chunk_id=f"sql-row-{i}",
            company=row[index["company"]] if "company" in index else "",
            year=row[index["report_year"]] if "report_year" in index else 0,
            section=None,
            pages=[row[page_col]] if page_col is not None else [],
            text=format_row(i, result.columns, row),
            score=0.0,
        )
        for i, row in enumerate(result.rows, start=1)
    ]
    return Answer(
        question=result.question,
        text=result.text,
        found=result.found,
        citations=[
            Citation(
                number=c.number,
                chunk_id=f"sql-row-{c.number}",
                company=c.company,
                year=c.year,
                pages=[c.page],
            )
            for c in result.citations
        ],
        chunks=chunks,
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_s=result.latency_s,
    )
