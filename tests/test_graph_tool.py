from collections.abc import Iterator
from typing import Any

import pytest
from neo4j import Driver
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from reportlens.core import llm
from reportlens.core.config import get_settings
from reportlens.core.graph_db import connect_graph
from reportlens.generation import prompts
from reportlens.sources import graph_tool
from reportlens.sources.graph_tool import CypherQuery, answer_with_graph, query_for, run_cypher
from reportlens.sources.sql_tool import as_answer

GOOD_CYPHER = (
    "MATCH (c:Company {id: 'rbc'})-[r:OWNS]->(s:Subsidiary)"
    " WHERE toLower(s.name) CONTAINS 'city national'"
    " RETURN c.id AS company, r.report_year AS report_year, r.source_page AS source_page,"
    " s.name AS subsidiary"
)
COLUMNS = ["company", "report_year", "source_page", "subsidiary"]
ROWS = [["rbc", 2023, 233, "City National Bank"], ["rbc", 2024, 247, "City National Bank"]]


@pytest.fixture
def driver() -> Iterator[Driver]:
    """Driver for the real graph, used read-only; skips the test if Neo4j is down."""
    try:
        with connect_graph() as graph_driver:
            yield graph_driver
    except ServiceUnavailable:
        pytest.skip("Neo4j not reachable; start it with `docker compose up -d`")


@pytest.fixture
def ran(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stub out the database: every query returns ROWS. Records the queries that reached it."""
    queries: list[str] = []

    def fake_run(_driver: object, query: str) -> tuple[list[str], list[list[Any]]]:
        queries.append(query)
        return COLUMNS, ROWS

    monkeypatch.setattr(graph_tool, "run_cypher", fake_run)
    return queries


def _writer(*queries: str):
    """Stub for the Cypher-writing LLM: returns the given queries in turn, recording its prompts."""
    seen: list[list[llm.Message]] = []

    def write(messages: list[llm.Message]) -> CypherQuery:
        seen.append(messages)
        return CypherQuery(cypher=queries[len(seen) - 1])

    return write, seen


def _chat(text: str):
    def chat(messages: list[llm.Message], tier: str) -> llm.LLMResponse:
        assert tier == "main"
        return llm.LLMResponse(
            text=text, model="main-model", input_tokens=80, output_tokens=15, latency_s=0.4
        )

    return chat


def test_server_refuses_writes_even_if_the_check_is_bypassed(driver: Driver) -> None:
    with pytest.raises(Neo4jError, match="read access mode"):
        run_cypher(driver, "CREATE (:Company {id: 'test-write'})")


def test_rows_are_capped_even_without_a_limit(driver: Driver) -> None:
    columns, rows = run_cypher(driver, "UNWIND range(1, 500) AS x RETURN x")
    assert columns == ["x"] and len(rows) == get_settings().cypher_row_limit


def test_slow_queries_time_out(driver: Driver, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CYPHER_TIMEOUT_S", "0.2")
    get_settings.cache_clear()
    with pytest.raises(Neo4jError):
        run_cypher(driver, "UNWIND range(1, 1000000000) AS x WITH x WHERE x = -1 RETURN x")


def test_cypher_prompt_lists_the_schema_and_company_ids() -> None:
    system = graph_tool.cypher_messages("q")[0]["content"]
    assert "[:OWNS {report_year, source_page, ownership_pct, location}]" in system
    assert "'scotiabank' (The Bank of Nova Scotia (Scotiabank))" in system


def test_query_is_checked_and_given_a_row_limit(ran: list[str]) -> None:
    write, _ = _writer(GOOD_CYPHER)
    query, columns, rows, error = query_for(None, "Which American bank does RBC own?", write)
    assert query.endswith("LIMIT 50") and ran == [query]
    assert (columns, rows, error) == (COLUMNS, ROWS, None)


def test_rejected_cypher_is_retried_with_the_reason(ran: list[str]) -> None:
    write, seen = _writer("MATCH (c:Company) DETACH DELETE c", GOOD_CYPHER)
    _, _, rows, _ = query_for(None, "q", write)
    assert rows and "DETACH is not allowed" in seen[1][-1]["content"]
    assert len(ran) == 1  # the rejected query never reached the database


def test_failing_cypher_is_retried_with_the_server_error(driver: Driver) -> None:
    write, seen = _writer("MATCH (c:Company) RETURN x", GOOD_CYPHER)
    _, columns, rows, error = query_for(driver, "Which American bank does RBC own?", write)
    assert "Variable `x` not defined (line 1, column 26" in seen[1][-1]["content"]
    assert error is None and columns == COLUMNS
    assert {row[3] for row in rows} == {"City National Bank"}  # in the real graph


def test_gives_up_after_two_failures(ran: list[str]) -> None:
    write, seen = _writer("CALL db.labels()", "MATCH (u:User) RETURN u")
    query, _, rows, error = query_for(None, "q", write)
    assert (query, rows, len(seen), ran) == (None, [], 2, [])
    assert "'User' does not exist" in error


def test_answer_cites_rows_and_scores_like_a_sql_answer(ran: list[str]) -> None:
    write, _ = _writer(GOOD_CYPHER)
    answer = answer_with_graph(
        None, "Which American bank does RBC own?", write, _chat("City National Bank [1][2].")
    )
    assert answer.found and answer.cypher.endswith("LIMIT 50")
    assert [(c.number, c.year, c.page) for c in answer.citations] == [
        (1, 2023, 233),
        (2, 2024, 247),
    ]

    scored = as_answer(answer)
    assert [(c.company, c.year, c.pages) for c in scored.chunks] == [
        ("rbc", 2023, [233]),
        ("rbc", 2024, [247]),
    ]
    assert scored.input_tokens == 80


def test_no_query_means_not_found_without_calling_the_answer_model(ran: list[str]) -> None:
    def no_chat(*args: object, **kwargs: object) -> llm.LLMResponse:
        raise AssertionError("the answer model should not be called")

    write, _ = _writer("")
    answer = answer_with_graph(None, "What was RBC's net income?", write, no_chat)
    assert (answer.found, answer.text, answer.cypher) == (False, prompts.NOT_FOUND, None)
    assert "can't answer" in answer.error and ran == []
