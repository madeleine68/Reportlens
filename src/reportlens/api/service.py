"""What the API does for one question, kept apart from HTTP so it is easy to read and test.

Resources live for the whole server: a Postgres connection pool (a psycopg connection must not
be shared between requests running in parallel threads) and one Neo4j driver (thread-safe), or
None when Neo4j is down, in which case graph routes fall back to text search as in the CLI.
"""

from collections.abc import Callable
from dataclasses import dataclass

import psycopg
from neo4j import Driver
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from reportlens.agents.corrective import answer_corrective, crag_cache, default_tools
from reportlens.agents.routed import ask, default_answerers
from reportlens.agents.router import route_cache
from reportlens.api.schemas import AskResponse, RouteOut, Source, response_fields
from reportlens.core.config import get_settings
from reportlens.core.db import connect, connect_reader
from reportlens.core.graph_db import connect_graph
from reportlens.core.logging import get_logger
from reportlens.generation.answer import answer_question
from reportlens.retrieval.filters import known_values
from reportlens.sources.graph_tool import answer_with_graph
from reportlens.sources.sql_tool import answer_with_sql, as_answer

log = get_logger(__name__)


class SourceUnavailable(Exception):
    """The requested source can't be reached (e.g. Neo4j is down and source=graph)."""


@dataclass
class Resources:
    pool: ConnectionPool
    driver: Driver | None


def open_resources() -> Resources:
    """Apply the schema once, then open the pool and (if reachable) the Neo4j driver."""
    connect().close()  # creates the schema and the SQL tool's read-only role if missing
    pool = ConnectionPool(
        get_settings().database_url,
        kwargs={"autocommit": True},
        configure=register_vector,
        min_size=1,
        max_size=4,
        open=True,
    )
    try:
        driver = connect_graph()
    except Exception as err:  # noqa: BLE001  # the API still answers without the graph
        log.warning("api.neo4j_unavailable error=%s", err)
        driver = None
    return Resources(pool=pool, driver=driver)


def take_daily_slot(conn: psycopg.Connection, cap: int) -> bool:
    """Count one demo question against today's cap (UTC); False, uncounted, if it is reached.

    One statement, so parallel requests can't both take the last slot.
    """
    row = conn.execute(
        """
        INSERT INTO demo_usage (day, questions) VALUES ((now() AT TIME ZONE 'utc')::date, 1)
        ON CONFLICT (day) DO UPDATE SET questions = demo_usage.questions + 1
            WHERE demo_usage.questions < %s
        RETURNING questions
        """,
        (cap,),
    ).fetchone()
    return row is not None


def daily_counter(resources: Resources) -> Callable[[int], bool]:
    """take_daily_slot on a pooled connection, for api/access.py."""

    def take(cap: int) -> bool:
        with resources.pool.connection() as conn:
            return take_daily_slot(conn, cap)

    return take


def close_resources(resources: Resources) -> None:
    resources.pool.close()
    if resources.driver is not None:
        resources.driver.close()


def warm_up_models() -> None:
    """Load the embedding and reranking models now, so the first question isn't a minute slower."""
    from reportlens.ingest.embed import embed_query
    from reportlens.retrieval.models import RetrievedChunk
    from reportlens.retrieval.rerank import rerank

    embed_query("warm up")
    chunk = RetrievedChunk(
        chunk_id="warm", company="", year=0, section=None, pages=[], text="warm up", score=0
    )
    rerank("warm up", [chunk], top_n=1)


def answer(question: str, source: Source, resources: Resources) -> AskResponse:
    """Answer one question from the requested source (auto: the router picks)."""
    if source == "graph" and resources.driver is None:
        raise SourceUnavailable("Neo4j is not reachable, so source=graph is unavailable")
    with resources.pool.connection() as conn:
        if source == "auto":
            with connect_reader() as reader:
                return _routed(question, conn, reader, resources.driver)
        if source == "text":
            return AskResponse(source=source, **response_fields(answer_question(conn, question)))
        if source == "corrective":
            result = answer_corrective(question, default_tools(conn, crag_cache()))
            return AskResponse(source=source, path=result.path, **response_fields(result.answer))
    if source == "sql":
        with connect_reader() as reader:
            rows = answer_with_sql(reader, question)
        return AskResponse(source=source, query=rows.sql, **response_fields(as_answer(rows)))
    rows = answer_with_graph(resources.driver, question)
    return AskResponse(source=source, query=rows.cypher, **response_fields(as_answer(rows)))


def _routed(
    question: str, conn: psycopg.Connection, reader: psycopg.Connection, driver: Driver | None
) -> AskResponse:
    answerers = default_answerers(conn, reader, driver)
    routed = ask(question, known_values(conn), answerers, route_cache())
    return AskResponse(
        source="auto",
        route=RouteOut(source=routed.route.source, reason=routed.route.reason),
        path=routed.path,
        checks=routed.checks,
        supported=routed.supported,
        **response_fields(routed.answer),
    )


def meta(resources: Resources) -> tuple[list[str], list[int]]:
    """Companies and years with chunks in the database."""
    with resources.pool.connection() as conn:
        return known_values(conn)


def health(resources: Resources) -> tuple[bool, bool]:
    """Whether Postgres and Neo4j answer right now."""
    try:
        with resources.pool.connection() as conn:
            conn.execute("SELECT 1")
        postgres = True
    except Exception:  # noqa: BLE001
        postgres = False
    neo4j = False
    if resources.driver is not None:
        try:
            resources.driver.verify_connectivity()
            neo4j = True
        except Exception:  # noqa: BLE001
            neo4j = False
    return postgres, neo4j
