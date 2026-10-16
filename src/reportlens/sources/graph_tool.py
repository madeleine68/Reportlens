"""Text-to-Cypher over the Neo4j knowledge graph (Phase 3).

1. The fast LLM writes one Cypher query, given the graph's labels, relationships and company ids.
2. sources/cypher_safety.py checks it and sets a row limit; it runs in a read transaction with a
   timeout, so the server refuses any write that got past the check.
3. If it is rejected or fails, the error goes back to the LLM for one more try.
4. The main LLM answers from the returned rows, citing them as [n]; each row carries its page.

Same steps as sources/sql_tool.py, whose answer step and row citations this reuses.
"""

import time
from collections.abc import Callable
from itertools import islice
from typing import Any

from neo4j import Driver, Query, Result, RoutingControl
from neo4j.exceptions import Neo4jError
from pydantic import BaseModel

from reportlens.core import llm
from reportlens.core.config import get_settings
from reportlens.core.logging import get_logger
from reportlens.generation import prompts
from reportlens.ingest.graph import COMPANY_NAMES
from reportlens.sources.cypher_safety import UnsafeCypher, check_cypher
from reportlens.sources.sql_tool import Chat, RowsAnswer, answer_from_rows

log = get_logger(__name__)

CYPHER_ATTEMPTS = 2  # first try plus one retry with the error fed back


class CypherQuery(BaseModel):
    """What the Cypher-writing LLM returns. An empty string means the graph can't answer."""

    cypher: str


class GraphAnswer(RowsAnswer):
    """The answer plus the Cypher query and rows behind it."""

    cypher: str | None  # the query that ran (after the safety check), if any


WriteCypher = Callable[[list[llm.Message]], CypherQuery]


def cypher_messages(question: str) -> list[llm.Message]:
    """System prompt describing the graph, plus the question."""
    system = prompts.CYPHER_SYSTEM.format(
        companies=", ".join(f"'{c}' ({name})" for c, name in COMPANY_NAMES.items()),
        row_limit=get_settings().cypher_row_limit,
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": question}]


def write_cypher(messages: list[llm.Message]) -> CypherQuery:
    """Ask the fast model for a query.

    Two retries: Groq sometimes rejects the JSON (which uses one retry to drop JSON mode) and then
    returns an empty reply (q027 on 2026-10-16), which needs another.
    """
    return llm.structured(messages, CypherQuery, tier="fast", max_retries=2)


def run_cypher(driver: Driver, query: str) -> tuple[list[str], list[list[Any]]]:
    """Run a checked query in a read transaction with a timeout; (column names, rows)."""
    settings = get_settings()

    def first_rows(result: Result) -> tuple[list[str], list[list[Any]]]:
        # The row limit is enforced here too: a LIMIT after UNION only limits the last part.
        rows = [record.values() for record in islice(result, settings.cypher_row_limit)]
        return list(result.keys()), rows

    start = time.perf_counter()
    columns, rows = driver.execute_query(
        Query(query, timeout=settings.cypher_timeout_s),  # type: ignore[arg-type]  # LLM-written
        routing_=RoutingControl.READ,
        result_transformer_=first_rows,
    )
    log.info("cypher.run rows=%d latency=%.2fs", len(rows), time.perf_counter() - start)
    return columns, rows


def query_for(
    driver: Driver, question: str, write: WriteCypher = write_cypher
) -> tuple[str | None, list[str], list[list[Any]], str | None]:
    """(query that ran, columns, rows, error): write, check and run, with one retry on failure."""
    messages = cypher_messages(question)
    error = None
    for _ in range(CYPHER_ATTEMPTS):
        proposed = write(messages).cypher.strip()
        if not proposed:
            return None, [], [], "the Cypher model said the graph can't answer the question"
        try:
            checked = check_cypher(proposed, get_settings().cypher_row_limit)
            columns, rows = run_cypher(driver, checked)
            return checked, columns, rows, None
        except (UnsafeCypher, Neo4jError) as err:
            error = (err.message if isinstance(err, Neo4jError) else None) or str(err).strip()
            log.warning("cypher.failed cypher=%r error=%s", proposed, error)
            messages = [
                *messages,
                {"role": "assistant", "content": CypherQuery(cypher=proposed).model_dump_json()},
                {"role": "user", "content": prompts.CYPHER_RETRY.format(error=error)},
            ]
    return None, [], [], error


def answer_with_graph(
    driver: Driver,
    question: str,
    write: WriteCypher = write_cypher,
    chat: Chat = llm.chat,
) -> GraphAnswer:
    """Answer a question from the knowledge graph, citing the rows used."""
    query, columns, rows, error = query_for(driver, question, write)
    empty = GraphAnswer(
        question=question,
        text=prompts.NOT_FOUND,
        found=False,
        cypher=query,
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
