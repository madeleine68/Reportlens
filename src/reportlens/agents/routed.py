"""Answer a question from the source the router picks, falling back to text search (Phase 4).

SQL, the graph and the web answer only what they hold; when they find nothing (no rows, or the
answer model says the rows don't answer it) or fail (a network or database error), text search
gets the question. That keeps a wrong route or a broken tool from costing the answer: SQL
rightly declines "why" questions, for example.

Text search here is corrective RAG (agents/corrective.py: grade the chunks, rewrite the query if
they fall short) unless ROUTED_CORRECTIVE=false, which uses plain search.
"""

from collections.abc import Callable

import psycopg
from neo4j import Driver
from pydantic import BaseModel

from reportlens.agents.corrective import Tools, answer_corrective, crag_cache, default_tools
from reportlens.agents.router import Known, Route, route_question
from reportlens.core import llm
from reportlens.core.config import get_settings
from reportlens.core.logging import get_logger
from reportlens.generation import prompts
from reportlens.generation.answer import Answer, answer_question, is_not_found
from reportlens.ingest.graph import FactCache
from reportlens.sources.graph_tool import answer_with_graph
from reportlens.sources.sql_tool import answer_with_sql, as_answer
from reportlens.sources.web_tool import answer_with_web, web_enabled

log = get_logger(__name__)

AnswerFn = Callable[[str], Answer]
FALLBACK = "vector"


class RoutedAnswer(BaseModel):
    """The answer, the router's decision, and the sources tried in order (e.g. sql → vector)."""

    answer: Answer
    route: Route
    path: list[str]


def no_lookup_answer(question: str, chat: Callable[..., llm.LLMResponse] = llm.chat) -> Answer:
    """A general-knowledge answer for questions that need no report (route 'none')."""
    reply = chat(
        [
            {"role": "system", "content": prompts.NO_LOOKUP_SYSTEM},
            {"role": "user", "content": question},
        ],
        tier="main",
    )
    text = reply.text.strip()
    return Answer(
        question=question,
        text=text,
        found=not is_not_found(text),
        citations=[],
        chunks=[],
        model=reply.model,
        input_tokens=reply.input_tokens,
        output_tokens=reply.output_tokens,
        latency_s=reply.latency_s,
    )


def corrective_answerer(conn: psycopg.Connection) -> AnswerFn:
    """Text search through corrective RAG; its tools are built on the first question."""
    tools: list[Tools] = []

    def answer(question: str) -> Answer:
        if not tools:
            tools.append(default_tools(conn, crag_cache()))
        return answer_corrective(question, tools[0]).answer

    return answer


def default_answerers(
    conn: psycopg.Connection, reader: psycopg.Connection, driver: Driver | None
) -> dict[str, AnswerFn]:
    """One answer function per source. Web search is off unless WEB_SEARCH=true: then text
    search, which refuses out-of-corpus questions, takes the 'web' route."""
    vector = (
        corrective_answerer(conn)
        if get_settings().routed_corrective
        else lambda q: answer_question(conn, q)
    )
    answerers: dict[str, AnswerFn] = {
        "vector": vector,
        "sql": lambda q: as_answer(answer_with_sql(reader, q)),
        "none": no_lookup_answer,
    }
    if driver is not None:
        answerers["graph"] = lambda q: as_answer(answer_with_graph(driver, q))
    if web_enabled():
        answerers["web"] = answer_with_web
    return answerers


def _add_costs(answer: Answer, earlier: list[Answer]) -> Answer:
    """The answer, with the tokens and time of the attempts that came before it added in."""
    return answer.model_copy(
        update={
            "input_tokens": answer.input_tokens + sum(a.input_tokens for a in earlier),
            "output_tokens": answer.output_tokens + sum(a.output_tokens for a in earlier),
            "latency_s": round(answer.latency_s + sum(a.latency_s for a in earlier), 3),
        }
    )


def answer_routed(question: str, route: Route, answerers: dict[str, AnswerFn]) -> RoutedAnswer:
    """Answer from the routed source; if it finds nothing (or isn't available), use text search."""
    path, attempts = [], []
    for source in dict.fromkeys([route.source, FALLBACK]):  # the routed source, then vector
        if source not in answerers:
            continue
        path.append(source)
        if source == FALLBACK:
            answer = answerers[source](question)
        else:
            try:
                answer = answerers[source](question)
            except llm.RateLimitError:
                raise  # out of quota, not a broken tool: falling back would hide it
            except Exception as err:  # noqa: BLE001  # a broken tool falls back to text search
                log.warning("routed.source_failed source=%s error=%s", source, err)
                path[-1] = f"{source} (error)"
                continue
        if answer.found or source == FALLBACK:
            return RoutedAnswer(answer=_add_costs(answer, attempts), route=route, path=path)
        attempts.append(answer)
    raise AssertionError("unreachable: the fallback source always answers")


def ask(
    question: str, known: Known, answerers: dict[str, AnswerFn], cache: FactCache | None = None
) -> RoutedAnswer:
    """Route the question, then answer it."""
    return answer_routed(question, route_question(question, known, cache), answerers)
