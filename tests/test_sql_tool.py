import psycopg
import pytest

from reportlens.core import llm
from reportlens.core.db import connect_reader
from reportlens.generation import prompts
from reportlens.ingest.financials import Financial, load_financials
from reportlens.sources import sql_tool
from reportlens.sources.sql_tool import (
    SqlQuery,
    answer_with_sql,
    format_rows,
    query_for,
    row_citations,
)

ROW = Financial(
    company="rbc",
    year=2024,
    report_year=2024,
    metric="net_income",
    scope="total",
    basis="reported",
    value=16240.0,
    unit="millions",
    source_page=26,
    source_table="Table 2",
    agreeing_cells=8,
    candidate_values=1,
)
GOOD_SQL = (
    "SELECT company, report_year, source_page, value, unit FROM financials"
    " WHERE company = 'rbc' AND metric = 'net_income' AND year = 2024"
)


@pytest.fixture
def reader(conn: psycopg.Connection):
    """Read-only connection to the test database, with one financials row loaded."""
    load_financials(conn, [ROW])
    with connect_reader() as reader_conn:
        yield reader_conn


def _writer(*queries: str):
    """Stub for the SQL-writing LLM: returns the given queries in turn, recording its prompts."""
    seen: list[list[llm.Message]] = []

    def write(messages: list[llm.Message]) -> SqlQuery:
        seen.append(messages)
        return SqlQuery(sql=queries[len(seen) - 1])

    return write, seen


def _chat(text: str):
    def chat(messages: list[llm.Message], tier: str) -> llm.LLMResponse:
        assert tier == "main"
        return llm.LLMResponse(
            text=text, model="main-model", input_tokens=100, output_tokens=20, latency_s=0.5
        )

    return chat


def test_reader_role_cannot_write_even_outside_a_read_only_transaction(reader) -> None:
    reader.read_only = False
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        reader.execute("DELETE FROM financials")
    reader.rollback()
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        reader.execute("SELECT * FROM chunks")


def test_reader_queries_time_out(reader) -> None:
    reader.execute("SET statement_timeout = 100")
    with pytest.raises(psycopg.errors.QueryCanceled):
        reader.execute("SELECT pg_sleep(1)")


def test_sql_prompt_lists_every_metric_and_segment() -> None:
    system = sql_tool.sql_messages("q")[0]["content"]
    assert "'net_income' (millions)" in system and "'cet1_ratio' (percent)" in system
    assert "Canadian Banking" in system and "Bell Media" in system


def test_query_runs_with_a_row_limit(reader) -> None:
    write, _ = _writer(GOOD_SQL)
    query, columns, rows, error = query_for(reader, "RBC net income 2024?", write)
    assert query.endswith("LIMIT 50")
    assert columns == ["company", "report_year", "source_page", "value", "unit"]
    assert rows == [["rbc", 2024, 26, 16240.0, "millions"]]
    assert error is None


def test_rejected_or_failing_sql_is_retried_with_the_error(reader) -> None:
    write, seen = _writer("SELECT * FROM chunks", GOOD_SQL)
    query, _, rows, _ = query_for(reader, "q", write)
    assert rows and "table 'chunks' is not allowed" in seen[1][-1]["content"]

    write, seen = _writer("SELECT nope FROM financials", GOOD_SQL)
    _, _, rows, _ = query_for(reader, "q", write)
    assert rows and 'column "nope" does not exist' in seen[1][-1]["content"]


def test_gives_up_after_two_failures(reader) -> None:
    write, seen = _writer("DROP TABLE financials", "SELECT * FROM documents")
    query, _, rows, error = query_for(reader, "q", write)
    assert (query, rows, len(seen)) == (None, [], 2)
    assert "documents" in error


def test_answer_cites_rows_by_company_report_and_page(reader) -> None:
    write, _ = _writer(GOOD_SQL)
    answer = answer_with_sql(
        reader, "RBC net income 2024?", write, _chat("RBC earned $16,240 million in 2024 [1].")
    )
    assert answer.found and answer.text.startswith("RBC earned")
    assert [(c.number, c.company, c.year, c.page) for c in answer.citations] == [
        (1, "rbc", 2024, 26)
    ]
    assert (answer.model, answer.input_tokens) == ("main-model", 100)


def test_no_rows_or_no_sql_means_not_found_without_calling_the_answer_model(reader) -> None:
    def no_chat(*args: object, **kwargs: object) -> llm.LLMResponse:
        raise AssertionError("the answer model should not be called")

    write, _ = _writer(GOOD_SQL.replace("2024", "1999"))
    assert answer_with_sql(reader, "q", write, no_chat).text == prompts.NOT_FOUND

    write, _ = _writer("")
    answer = answer_with_sql(reader, "Why did profit fall?", write, no_chat)
    assert (answer.found, answer.sql) == (False, None)
    assert "can't answer" in answer.error


def test_format_rows_numbers_rows_as_column_value_pairs() -> None:
    assert format_rows(["company", "value"], [["rbc", 1.0], ["bce", 2.0]]) == (
        "[1] company=rbc, value=1.0\n[2] company=bce, value=2.0"
    )


def test_row_citations_need_company_report_and_page_columns() -> None:
    rows = [["rbc", 2024, 26], ["bce", 2023, 47]]
    cites = row_citations("x [2][1] y [9]", ["company", "report_year", "page"], rows)
    assert [(c.number, c.company, c.page) for c in cites] == [(2, "bce", 47), (1, "rbc", 26)]
    assert row_citations("x [1]", ["company", "value"], [["rbc", 1.0]]) == []
