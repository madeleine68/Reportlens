"""Router (Phase 4): the fast LLM picks which source should answer a question.

The sources and what the evals showed each is good at:
- sql: figures, trends and comparisons (0.80 vs 0.67 for text on the 33 numeric questions)
- graph: officer roles, subsidiaries and segments over time (it won the CFO-change question)
- vector: text search, the default (0.83 vs 0.56 for the graph on relationship questions)
- web: outside the reports (no web tool yet: agents/routed.py sends these to vector, which
  refuses out-of-corpus questions)
- none: no lookup needed

Routes are cached by model, prompt and question, like the metadata filters, so eval runs are
repeatable and cost nothing the second time. If the router itself fails, the question goes to
vector: a router error should never stop a question from being answered.
"""

from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel

from reportlens.core import llm
from reportlens.core.config import get_settings
from reportlens.core.logging import get_logger
from reportlens.generation import prompts
from reportlens.ingest.graph import FactCache

log = get_logger(__name__)

Source = Literal["vector", "sql", "graph", "web", "none"]
Structured = Callable[..., BaseModel]
Known = tuple[list[str], list[int]]  # companies and years in the database


class Route(BaseModel):
    """Where to look up the answer, and why."""

    source: Source
    reason: str


def router_messages(question: str, known: Known) -> list[llm.Message]:
    """System prompt naming the sources and the companies/years covered, plus the question."""
    companies, years = known
    system = prompts.ROUTER_SYSTEM.format(
        companies=", ".join(companies), years=f"{min(years)}-{max(years)}" if years else "none"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": question}]


def route_question(
    question: str,
    known: Known,
    cache: FactCache | None = None,
    structured: Structured = llm.structured,
) -> Route:
    """The source for this question; vector if the router fails."""
    messages = router_messages(question, known)
    key = FactCache.key(messages) if cache else ""
    if cache and (cached := cache.get(key)) is not None:
        route = Route.model_validate(cached)
        log.info("router.cache_hit source=%s", route.source)
        return route
    try:
        route = structured(messages, Route, tier="fast")
    except llm.RateLimitError:
        raise  # out of quota: falling back would silently change the route (and the evals)
    except Exception as err:  # noqa: BLE001  # any router failure falls back to the default
        log.warning("router.failed error=%s", err)
        return Route(source="vector", reason=f"router failed ({type(err).__name__}); default")
    if cache:
        cache.put(key, route.model_dump())
    log.info("router.route source=%s reason=%s", route.source, route.reason)
    return route


def route_cache() -> FactCache:
    """The router's cache file (data/cache/routes.json)."""
    return FactCache(get_settings().route_cache_path)
