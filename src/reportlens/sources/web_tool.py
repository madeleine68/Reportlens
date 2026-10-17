"""Web search with Tavily (Phase 4), for questions the annual reports don't cover.

Off unless WEB_SEARCH=true (and TAVILY_API_KEY is set): the evals score out-of-corpus questions
as correct only when they are refused. When on, the router's 'web' route lands here:
1. Tavily returns the top results (title, URL, a text extract) for the question.
2. The main LLM answers only from them, citing [n], and starts with "From the web, not the
   annual reports:" so a web answer is never mistaken for one from the reports.
Web pages are untrusted text: the prompt tells the model to ignore instructions inside them.
The API is one JSON POST, so this uses the standard library rather than another dependency.
"""

import json
import time
import urllib.request
from collections.abc import Callable

from pydantic import BaseModel

from reportlens.core import llm
from reportlens.core.config import get_settings
from reportlens.core.logging import get_logger
from reportlens.generation import prompts
from reportlens.generation.answer import (
    Answer,
    is_not_found,
    normalize_markers,
    parse_citations,
)
from reportlens.retrieval.models import RetrievedChunk

log = get_logger(__name__)

TAVILY_URL = "https://api.tavily.com/search"
MAX_RESULT_CHARS = 1500  # per result, so five results stay a small prompt


class WebResult(BaseModel):
    """One search result."""

    title: str
    url: str
    content: str


PostJson = Callable[[str, dict, dict[str, str], float], dict]
Search = Callable[[str], list[WebResult]]
Chat = Callable[..., llm.LLMResponse]


def _api_key() -> str:
    """The Tavily key, or '' if unset (an empty TAVILY_API_KEY= counts as unset)."""
    key = get_settings().tavily_api_key
    return key.get_secret_value().strip() if key is not None else ""


def web_enabled() -> bool:
    """True when web search is switched on and has a key."""
    return get_settings().web_search and bool(_api_key())


def post_json(url: str, body: dict, headers: dict[str, str], timeout_s: float) -> dict:
    """POST a JSON body and return the JSON reply."""
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read())


def search_web(question: str, post: PostJson = post_json) -> list[WebResult]:
    """Tavily's top results for the question."""
    settings = get_settings()
    if not (key := _api_key()):
        raise RuntimeError("TAVILY_API_KEY is not set")
    start = time.perf_counter()
    reply = post(
        TAVILY_URL,
        {"query": question, "max_results": settings.web_max_results, "search_depth": "basic"},
        {"Authorization": f"Bearer {key}"},
        settings.web_timeout_s,
    )
    results = [
        WebResult(title=r.get("title") or r["url"], url=r["url"], content=r.get("content") or "")
        for r in reply.get("results", [])
        if r.get("url")
    ]
    log.info("web.search results=%d latency=%.2fs", len(results), time.perf_counter() - start)
    return results


def as_chunks(results: list[WebResult]) -> list[RetrievedChunk]:
    """Results in the shape answers cite: the URL is the chunk id, the title the section."""
    return [
        RetrievedChunk(
            chunk_id=r.url,
            company="web",
            year=0,
            section=r.title,
            pages=[],
            text=r.content[:MAX_RESULT_CHARS],
            score=0.0,
        )
        for r in results
    ]


def web_messages(question: str, chunks: list[RetrievedChunk]) -> list[llm.Message]:
    """System rules plus the numbered results and the question."""
    sources = "\n\n".join(
        prompts.WEB_RESULT_HEADER.format(number=i, title=c.section, url=c.chunk_id) + "\n" + c.text
        for i, c in enumerate(chunks, start=1)
    )
    user = prompts.ANSWER_USER.format(sources=sources, question=question)
    return [
        {"role": "system", "content": prompts.WEB_ANSWER_SYSTEM},
        {"role": "user", "content": user},
    ]


def answer_with_web(question: str, search: Search = search_web, chat: Chat = llm.chat) -> Answer:
    """Answer from web search results, citing their URLs; not found if nothing answers it."""
    chunks = as_chunks(search(question))
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
    reply = chat(web_messages(question, chunks), tier="main")
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
