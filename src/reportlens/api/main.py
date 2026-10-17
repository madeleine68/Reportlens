"""The HTTP API (Phase 5).

    uv run uvicorn reportlens.api.main:app --reload

- GET  /            the web UI (api/static/index.html: one page, no build step)
- GET  /health      Postgres and Neo4j reachable? web search and Self-RAG on?
- GET  /meta        companies and years loaded, and the sources a question can ask for
- POST /ask         {"question": "...", "source": "auto"} -> the answer, route, steps and sources
- POST /ask/stream  the same, as server-sent events: route, step..., answer, done (or error)

The stream reports progress, not tokens: the router's pick as soon as it is made, then each
corrective RAG and Self-RAG step, then the whole answer (citations are parsed and the answer is
checked only once it is complete).
"""

import json
import queue
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from reportlens.agents.events import listening
from reportlens.api import service
from reportlens.api.access import password_required, require_access
from reportlens.api.schemas import AskRequest, AskResponse, Health, Meta, Source
from reportlens.core import llm
from reportlens.core.config import get_settings
from reportlens.core.logging import get_logger, setup_logging
from reportlens.sources.web_tool import web_enabled

log = get_logger(__name__)

AnswerFn = Callable[[str, Source, service.Resources], AskResponse]
SOURCES: list[Source] = ["auto", "text", "corrective", "sql", "graph"]
UI_PAGE = Path(__file__).parent / "static" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    setup_logging(get_settings().log_level)
    app.state.resources = service.open_resources()
    app.state.take_day = service.daily_counter(app.state.resources)  # cap survives restarts
    service.warm_up_models()
    log.info("api.ready neo4j=%s", app.state.resources.driver is not None)
    yield
    service.close_resources(app.state.resources)


app = FastAPI(title="ReportLens", version="0.1.0", lifespan=lifespan)


def get_resources(request: Request) -> service.Resources:
    return request.app.state.resources


def get_answer_fn() -> AnswerFn:
    """The function that answers a question; tests replace it with a stub."""
    return service.answer


ResourcesDep = Annotated[service.Resources, Depends(get_resources)]
AnswerDep = Annotated[AnswerFn, Depends(get_answer_fn)]


# --- errors ---------------------------------------------------------------------------------


def error_status(err: Exception) -> tuple[int, str]:
    """HTTP status and a short message for an error raised while answering."""
    if isinstance(err, llm.RateLimitError):
        limit = "daily" if llm.is_daily_limit(err) else "per-minute"
        return 429, f"Out of LLM quota ({limit} limit); try again later."
    if isinstance(err, service.SourceUnavailable):
        return 503, str(err)
    log.exception("api.error")  # details go to the log, not the response
    return 500, "Internal error while answering; see the server log."


@app.exception_handler(llm.RateLimitError)
@app.exception_handler(service.SourceUnavailable)
async def known_error(request: Request, err: Exception) -> JSONResponse:
    status, message = error_status(err)
    return JSONResponse(status_code=status, content={"detail": message})


# --- routes ---------------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
def ui() -> FileResponse:
    return FileResponse(UI_PAGE, media_type="text/html")


@app.get("/health")
def health(resources: ResourcesDep) -> Health:
    postgres, neo4j = service.health(resources)
    settings = get_settings()
    return Health(
        postgres=postgres,
        neo4j=neo4j,
        web_search=web_enabled(),
        self_rag=settings.self_rag,
        password_required=password_required(),
    )


@app.get("/meta")
def meta(resources: ResourcesDep) -> Meta:
    companies, years = service.meta(resources)
    return Meta(companies=companies, years=years, sources=SOURCES)


@app.post("/ask", dependencies=[Depends(require_access)])
def ask(
    body: AskRequest,
    resources: ResourcesDep,
    answer_fn: AnswerDep,
) -> AskResponse:
    try:
        return answer_fn(body.question, body.source, resources)
    except (llm.RateLimitError, service.SourceUnavailable):
        raise
    except Exception as err:  # noqa: BLE001
        status, message = error_status(err)
        raise HTTPException(status_code=status, detail=message) from err


def sse(kind: str, data: dict[str, Any]) -> str:
    """One server-sent event."""
    return f"event: {kind}\ndata: {json.dumps(data)}\n\n"


def stream_events(
    question: str, source: Source, resources: service.Resources, answer_fn: AnswerFn
) -> Iterator[str]:
    """Answer in a worker thread, yielding its progress events as they happen."""
    events: queue.Queue[tuple[str, dict[str, Any]] | None] = queue.Queue()

    def work() -> None:
        try:
            with listening(lambda kind, data: events.put((kind, data))):
                result = answer_fn(question, source, resources)
            events.put(("answer", result.model_dump()))
            events.put(("done", {}))
        except Exception as err:  # noqa: BLE001  # reported to the client as an error event
            status, message = error_status(err)
            events.put(("error", {"status": status, "detail": message}))
        finally:
            events.put(None)

    threading.Thread(target=work, daemon=True).start()
    while (event := events.get()) is not None:
        yield sse(*event)


@app.post("/ask/stream", dependencies=[Depends(require_access)])
def ask_stream(
    body: AskRequest,
    resources: ResourcesDep,
    answer_fn: AnswerDep,
) -> StreamingResponse:
    return StreamingResponse(
        stream_events(body.question, body.source, resources, answer_fn),
        media_type="text/event-stream",
        # no-buffering: proxies in front of the app must pass each event on at once
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
