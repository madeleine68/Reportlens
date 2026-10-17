"""Corrective RAG (Phase 4): grade what text search found before answering from it.

A LangGraph state graph with five nodes:
1. retrieve: text search (the best stack: filters, hybrid, rerank) for the current query.
2. grade: the fast model reads the chunks and says which are relevant and whether together they
   are enough to answer. Relevant chunks are kept across rounds; the rest are dropped.
3. rewrite: if they are not enough, the fast model rewrites the question into a search query
   aimed at what is missing (at most CRAG_MAX_REWRITES times), and the graph retrieves again.
4. generate: the main model answers from the relevant chunks only, with page citations.
5. fallback: nothing relevant even after rewriting: web search if WEB_SEARCH=true, otherwise
   the fixed refusal.

Grading and rewriting are improvements, not requirements: if either fails, the graph carries on
as plain text search would (all chunks kept / the query unchanged). A run longer than
AGENT_MAX_STEPS stops with the refusal instead of looping.
"""

import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Literal, TypedDict

import psycopg
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from reportlens.agents.router import Known
from reportlens.core import llm
from reportlens.core.config import get_settings
from reportlens.core.logging import get_logger
from reportlens.generation import prompts
from reportlens.generation.answer import Answer, answer_question, format_pages
from reportlens.ingest.graph import FactCache
from reportlens.retrieval.filters import known_values
from reportlens.retrieval.models import RetrievedChunk
from reportlens.retrieval.search import search
from reportlens.sources.web_tool import answer_with_web, web_enabled

log = get_logger(__name__)

Structured = Callable[..., BaseModel]
GRADE_MAX_TOKENS = 2000  # prompt (~2.5k) + this stays under the 8,000 tokens-per-minute limit


class Grade(BaseModel):
    """The grader's verdict on one round of retrieved chunks."""

    relevant: list[int]  # 1-based numbers of the relevant chunks
    enough: bool
    missing: str = ""


class Rewrite(BaseModel):
    """A new search query."""

    query: str


@dataclass
class Tools:
    """What the graph's nodes call; tests pass stubs."""

    retrieve: Callable[[str], list[RetrievedChunk]]
    grade: Callable[[str, list[RetrievedChunk]], Grade]
    rewrite: Callable[[str, str], str]  # (question, what is missing) -> query
    generate: Callable[[str, list[RetrievedChunk]], Answer]
    web: Callable[[str], Answer] | None  # None when web search is off


class CragState(TypedDict, total=False):
    question: str
    query: str  # what retrieve searches for: the question, then each rewrite
    retrieved: list[RetrievedChunk]  # the latest round
    relevant: list[RetrievedChunk]  # graded relevant, across all rounds
    enough: bool
    missing: str
    rewrites: int
    path: Annotated[list[str], operator.add]  # each node appends what it did
    answer: Answer


class CorrectiveAnswer(BaseModel):
    """The answer plus the steps taken, e.g. retrieve → grade 1/5 → rewrite → ... → generate."""

    answer: Answer
    path: list[str]


# --- grading and rewriting (fast model, cached) ------------------------------------------------


def _cached[T: BaseModel](
    messages: list[llm.Message], schema: type[T], cache: FactCache | None, structured: Structured
) -> T:
    key = FactCache.key(messages) if cache else ""
    if cache and (hit := cache.get(key)) is not None:
        return schema.model_validate(hit)
    result = structured(messages, schema, tier="fast", max_tokens=GRADE_MAX_TOKENS)
    if cache:
        cache.put(key, result.model_dump())
    return result


def grade_messages(question: str, chunks: list[RetrievedChunk]) -> list[llm.Message]:
    """The grading prompt: the question and the numbered chunks with company, year and pages."""
    passages = "\n\n".join(
        f"[{i}] {c.company.upper()} {c.year}, {format_pages(c.pages)}\n{c.text}"
        for i, c in enumerate(chunks, start=1)
    )
    user = prompts.GRADE_USER.format(question=question, passages=passages)
    return [{"role": "system", "content": prompts.GRADE_SYSTEM}, {"role": "user", "content": user}]


def grade_chunks(
    question: str,
    chunks: list[RetrievedChunk],
    cache: FactCache | None = None,
    structured: Structured = llm.structured,
) -> Grade:
    """Which chunks are relevant, and whether they are enough (numbers out of range dropped)."""
    grade = _cached(grade_messages(question, chunks), Grade, cache, structured)
    relevant = sorted({n for n in grade.relevant if 1 <= n <= len(chunks)})
    return grade.model_copy(update={"relevant": relevant})


def rewrite_query(
    question: str,
    missing: str,
    known: Known,
    cache: FactCache | None = None,
    structured: Structured = llm.structured,
) -> str:
    """A search query aimed at what the first search missed."""
    companies, years = known
    system = prompts.REWRITE_SYSTEM.format(
        companies=", ".join(companies), years=f"{min(years)}-{max(years)}" if years else "none"
    )
    user = prompts.REWRITE_USER.format(question=question, missing=missing or "unknown")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return _cached(messages, Rewrite, cache, structured).query.strip()


# --- the graph --------------------------------------------------------------------------------


def refusal(question: str) -> Answer:
    """The fixed not-found answer, at no cost."""
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


def _merge(kept: list[RetrievedChunk], new: list[RetrievedChunk]) -> list[RetrievedChunk]:
    seen = {c.chunk_id for c in kept}
    return kept + [c for c in new if c.chunk_id not in seen]


def build_graph(tools: Tools, max_rewrites: int):  # noqa: ANN201  # LangGraph's compiled type
    """The corrective RAG state graph, compiled and ready to invoke."""

    def retrieve(state: CragState) -> CragState:
        chunks = tools.retrieve(state["query"])
        return {"retrieved": chunks, "path": [f"retrieve ({len(chunks)})"]}

    def grade(state: CragState) -> CragState:
        chunks = state["retrieved"]
        if not chunks:
            return {"enough": False, "missing": "no results", "path": ["grade (nothing found)"]}
        try:
            verdict = tools.grade(state["question"], chunks)
        except Exception as err:  # noqa: BLE001  # grading is optional: keep everything
            log.warning("corrective.grade_failed error=%s", err)
            relevant = _merge(state.get("relevant", []), chunks)
            return {"relevant": relevant, "enough": True, "path": ["grade failed: kept all"]}
        picked = [chunks[n - 1] for n in verdict.relevant]
        return {
            "relevant": _merge(state.get("relevant", []), picked),
            "enough": verdict.enough and bool(picked),
            "missing": verdict.missing,
            "path": [f"grade {len(picked)}/{len(chunks)} relevant{', enough' * verdict.enough}"],
        }

    def decide(state: CragState) -> Literal["generate", "rewrite", "fallback"]:
        if state["enough"]:
            return "generate"
        if state.get("rewrites", 0) < max_rewrites:
            return "rewrite"
        return "generate" if state.get("relevant") else "fallback"

    def rewrite(state: CragState) -> CragState:
        rewrites = state.get("rewrites", 0) + 1
        try:
            query = tools.rewrite(state["question"], state.get("missing", ""))
        except Exception as err:  # noqa: BLE001  # rewriting is optional: search again as is
            log.warning("corrective.rewrite_failed error=%s", err)
            return {"rewrites": rewrites, "path": ["rewrite failed"]}
        query = query or state["query"]
        return {"query": query, "rewrites": rewrites, "path": [f"rewrite: {query}"]}

    def generate(state: CragState) -> CragState:
        answer = tools.generate(state["question"], state["relevant"])
        return {"answer": answer, "path": ["generate"]}

    def fallback(state: CragState) -> CragState:
        if tools.web is None:
            return {"answer": refusal(state["question"]), "path": ["refuse"]}
        return {"answer": tools.web(state["question"]), "path": ["web"]}

    graph = StateGraph(CragState)
    for name, node in [
        ("retrieve", retrieve),
        ("grade", grade),
        ("rewrite", rewrite),
        ("generate", generate),
        ("fallback", fallback),
    ]:
        graph.add_node(name, node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", decide)
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", END)
    graph.add_edge("fallback", END)
    return graph.compile()


def answer_corrective(question: str, tools: Tools) -> CorrectiveAnswer:
    """Run the graph for one question; past the step limit, refuse rather than loop."""
    settings = get_settings()
    graph = build_graph(tools, settings.crag_max_rewrites)
    start: CragState = {"question": question, "query": question, "rewrites": 0, "path": []}
    try:
        state = graph.invoke(start, {"recursion_limit": settings.agent_max_steps})
    except GraphRecursionError:
        log.warning("corrective.step_limit steps=%d", settings.agent_max_steps)
        return CorrectiveAnswer(answer=refusal(question), path=["step limit reached"])
    log.info("corrective.path %s", " -> ".join(state["path"]))
    return CorrectiveAnswer(answer=state["answer"], path=state["path"])


def crag_cache() -> FactCache:
    """The grades-and-rewrites cache file (data/cache/corrective.json)."""
    return FactCache(get_settings().crag_cache_path)


def default_tools(conn: psycopg.Connection, cache: FactCache | None = None) -> Tools:
    """The real tools: the best text-search stack, fast-model grading, main-model answers."""
    known = known_values(conn)
    return Tools(
        retrieve=lambda query: search(conn, query),
        grade=lambda question, chunks: grade_chunks(question, chunks, cache),
        rewrite=lambda question, missing: rewrite_query(question, missing, known, cache),
        generate=lambda question, chunks: answer_question(
            conn, question, retrieve=lambda _conn, _q: chunks
        ),
        web=answer_with_web if web_enabled() else None,
    )
