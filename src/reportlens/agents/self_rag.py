"""Self-RAG (Phase 4): check that an answer is supported by its sources, and revise it once if not.

"Is retrieval needed at all?" is the router's job (its 'none' route answers without a lookup), so
this module does the second half: a LangGraph state graph with two nodes.
1. check: the fast model reads the answer and the sources it was written from (text chunks, or
   SQL/graph rows as pseudo-chunks) and lists facts no cited source states.
2. revise: if some are unsupported, the main model rewrites the answer from the same sources,
   told which claims to fix or drop; then it is checked again. At most SELF_RAG_MAX_REVISIONS.

An answer that still fails is returned anyway, marked unsupported: the checker can be wrong too,
and refusing would throw away correct answers. Refusals and answers with no sources are not
checked. A check or revision that fails leaves the answer as it was (rate limits are raised).
"""

import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Literal, TypedDict

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from reportlens.core import llm
from reportlens.core.config import get_settings
from reportlens.core.logging import get_logger
from reportlens.generation import prompts
from reportlens.generation.answer import (
    Answer,
    format_pages,
    is_not_found,
    normalize_markers,
    parse_citations,
)
from reportlens.ingest.graph import FactCache
from reportlens.retrieval.models import RetrievedChunk

log = get_logger(__name__)

Structured = Callable[..., BaseModel]
Chat = Callable[..., llm.LLMResponse]
CHECK_MAX_TOKENS = 2000  # prompt (~3k) + this stays under the 8,000 tokens-per-minute limit


class SupportCheck(BaseModel):
    """The checker's verdict on one answer."""

    supported: bool
    unsupported_claims: list[str] = []


@dataclass
class CheckTools:
    """What the graph's nodes call; tests pass stubs."""

    check: Callable[[Answer], SupportCheck]
    revise: Callable[[Answer, list[str]], Answer]  # (answer, unsupported claims) -> new answer


class SelfRagState(TypedDict, total=False):
    answer: Answer
    supported: bool | None  # None: not checked
    claims: list[str]
    revisions: int
    path: Annotated[list[str], operator.add]


class CheckedAnswer(BaseModel):
    """The (possibly revised) answer, whether it passed the check, and the steps taken."""

    answer: Answer
    supported: bool | None  # None when the answer was not checked
    path: list[str]


# --- prompts --------------------------------------------------------------------------------


def source_label(chunk: RetrievedChunk) -> str:
    """'RBC 2023, page 4' for report chunks and rows, 'title (url)' for web results."""
    if chunk.company == "web":
        return f"{chunk.section} ({chunk.chunk_id})"
    company = chunk.company.upper() or "SOURCE"
    pages = f", {format_pages(chunk.pages)}" if chunk.pages else ""
    return f"{company} {chunk.year}{pages}" if chunk.year else f"{company}{pages}"


def numbered_sources(chunks: list[RetrievedChunk]) -> str:
    """The sources as the answer saw them: [1] label, then the text."""
    return "\n\n".join(f"[{i}] {source_label(c)}\n{c.text}" for i, c in enumerate(chunks, start=1))


def check_messages(answer: Answer) -> list[llm.Message]:
    """The support-check prompt for one answer."""
    user = prompts.SUPPORT_USER.format(
        sources=numbered_sources(answer.chunks), question=answer.question, answer=answer.text
    )
    return [
        {"role": "system", "content": prompts.SUPPORT_SYSTEM},
        {"role": "user", "content": user},
    ]


def revise_messages(answer: Answer, claims: list[str]) -> list[llm.Message]:
    """The revision prompt: same sources and question, the earlier answer and what to fix."""
    user = prompts.REVISE_USER.format(
        sources=numbered_sources(answer.chunks),
        question=answer.question,
        answer=answer.text,
        claims="\n".join(f"- {c}" for c in claims) or "- (not specified)",
    )
    return [{"role": "system", "content": prompts.REVISE_SYSTEM}, {"role": "user", "content": user}]


# --- the two LLM steps -----------------------------------------------------------------------


def check_support(
    answer: Answer, cache: FactCache | None = None, structured: Structured = llm.structured
) -> SupportCheck:
    """The fast model's verdict, cached by model, prompt and answer."""
    messages = check_messages(answer)
    key = FactCache.key(messages) if cache else ""
    if cache and (hit := cache.get(key)) is not None:
        return SupportCheck.model_validate(hit)
    verdict = structured(messages, SupportCheck, tier="fast", max_tokens=CHECK_MAX_TOKENS)
    if cache:
        cache.put(key, verdict.model_dump())
    return verdict


def revise_answer(answer: Answer, claims: list[str], chat: Chat = llm.chat) -> Answer:
    """The main model's rewrite from the same sources; its tokens are added to the answer's."""
    reply = chat(revise_messages(answer, claims), tier="main")
    text = normalize_markers(reply.text).strip()
    found = not is_not_found(text)
    return answer.model_copy(
        update={
            "text": text,
            "found": found,
            "citations": parse_citations(text, answer.chunks) if found else [],
            "input_tokens": answer.input_tokens + reply.input_tokens,
            "output_tokens": answer.output_tokens + reply.output_tokens,
            "latency_s": round(answer.latency_s + reply.latency_s, 3),
        }
    )


# --- the graph --------------------------------------------------------------------------------


def needs_check(answer: Answer) -> bool:
    """Only answers with sources are checked: not refusals or no-lookup answers."""
    return answer.found and bool(answer.chunks)


def build_graph(tools: CheckTools, max_revisions: int):  # noqa: ANN201  # LangGraph's type
    """The Self-RAG state graph, compiled and ready to invoke."""

    def check(state: SelfRagState) -> SelfRagState:
        answer = state["answer"]
        if not needs_check(answer):
            return {"supported": None, "path": ["check skipped (no sources)"]}
        try:
            verdict = tools.check(answer)
        except llm.RateLimitError:
            raise  # out of quota: passing the answer through would quietly skip the check
        except Exception as err:  # noqa: BLE001  # the check is optional: keep the answer
            log.warning("self_rag.check_failed error=%s", err)
            return {"supported": None, "path": ["check failed"]}
        supported = verdict.supported and not verdict.unsupported_claims
        label = "supported" if supported else f"{len(verdict.unsupported_claims)} unsupported"
        return {"supported": supported, "claims": verdict.unsupported_claims, "path": [label]}

    def decide(state: SelfRagState) -> Literal["revise", "__end__"]:
        if state.get("supported") is False and state.get("revisions", 0) < max_revisions:
            return "revise"
        return END

    def revise(state: SelfRagState) -> SelfRagState:
        revisions = state.get("revisions", 0) + 1
        try:
            answer = tools.revise(state["answer"], state.get("claims", []))
        except llm.RateLimitError:
            raise
        except Exception as err:  # noqa: BLE001  # keep the earlier answer, marked unsupported
            log.warning("self_rag.revise_failed error=%s", err)
            return {"revisions": max_revisions, "path": ["revise failed"]}
        return {"answer": answer, "revisions": revisions, "path": ["revise"]}

    graph = StateGraph(SelfRagState)
    graph.add_node("check", check)
    graph.add_node("revise", revise)
    graph.add_edge(START, "check")
    graph.add_conditional_edges("check", decide)
    graph.add_edge("revise", "check")
    return graph.compile()


def self_check(answer: Answer, tools: CheckTools) -> CheckedAnswer:
    """Check the answer and revise it if needed; past the step limit, return it unchecked."""
    settings = get_settings()
    graph = build_graph(tools, settings.self_rag_max_revisions)
    start: SelfRagState = {"answer": answer, "revisions": 0, "path": []}
    try:
        state = graph.invoke(start, {"recursion_limit": settings.agent_max_steps})
    except GraphRecursionError:
        log.warning("self_rag.step_limit steps=%d", settings.agent_max_steps)
        return CheckedAnswer(answer=answer, supported=None, path=["step limit reached"])
    log.info("self_rag.path %s", " -> ".join(state["path"]))
    return CheckedAnswer(
        answer=state["answer"], supported=state.get("supported"), path=state["path"]
    )


def default_check_tools() -> CheckTools:
    """Fast-model checks (cached in data/cache/self_rag.json) and main-model revisions."""
    cache = FactCache(get_settings().self_rag_cache_path)
    return CheckTools(check=lambda answer: check_support(answer, cache), revise=revise_answer)
