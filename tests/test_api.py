import json

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

from reportlens.agents.events import emit
from reportlens.api import main, service
from reportlens.api.schemas import AskResponse, RouteOut, response_fields
from reportlens.generation.answer import Answer, Citation
from reportlens.retrieval.models import RetrievedChunk


def _answer(question: str) -> Answer:
    chunk = RetrievedChunk(
        chunk_id="rbc-2023-1",
        company="rbc",
        year=2023,
        section="Results",
        pages=[26],
        text="t",
        score=2,
    )
    web = RetrievedChunk(
        chunk_id="https://x.com", company="web", year=0, section="X", pages=[], text="w", score=0
    )
    return Answer(
        question=question,
        text="Net income was $14,866 million [1]; see also [2].",
        found=True,
        citations=[
            Citation(number=1, chunk_id="rbc-2023-1", company="rbc", year=2023, pages=[26]),
            Citation(number=2, chunk_id="https://x.com", company="web", year=0, pages=[]),
        ],
        chunks=[chunk, web],
        model="main-model",
        input_tokens=100,
        output_tokens=10,
        latency_s=1.5,
    )


def stub_answer(question: str, source: str, resources: object) -> AskResponse:
    emit("route", source="vector", reason="text question")
    emit("step", stage="corrective", text="grade 1/5 relevant, enough")
    return AskResponse(
        source=source,
        route=RouteOut(source="vector", reason="text question"),
        path=["vector"],
        **response_fields(_answer(question)),
    )


def _rate_limit(message: str) -> openai.RateLimitError:
    request = httpx.Request("POST", "https://api.groq.com")
    return openai.RateLimitError(message, response=httpx.Response(429, request=request), body=None)


def failing(err: Exception):
    def answer(question: str, source: str, resources: object) -> AskResponse:
        raise err

    return answer


@pytest.fixture
def client():
    """A client with stub resources and a stub answerer; the lifespan (models, DBs) never runs."""
    main.app.dependency_overrides[main.get_resources] = lambda: object()
    main.app.dependency_overrides[main.get_answer_fn] = lambda: stub_answer
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


def _use(answer_fn) -> None:
    main.app.dependency_overrides[main.get_answer_fn] = lambda: answer_fn


def test_ask_returns_the_answer_route_citations_and_chunks(client: TestClient) -> None:
    body = client.post("/ask", json={"question": "  RBC net income 2023?  "}).json()
    assert body["question"] == "RBC net income 2023?" and body["source"] == "auto"
    assert body["route"] == {"source": "vector", "reason": "text question"}
    assert body["citations"][0] == {
        "number": 1,
        "company": "rbc",
        "year": 2023,
        "pages": [26],
        "url": None,
    }
    assert body["citations"][1]["url"] == "https://x.com"
    assert [c["id"] for c in body["chunks"]] == ["rbc-2023-1", "https://x.com"]
    assert (body["input_tokens"], body["path"], body["supported"]) == (100, ["vector"], None)


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "   "},
        {"question": "x" * 501},
        {"question": "q", "source": "db"},
    ],
)
def test_bad_questions_and_sources_are_rejected(client: TestClient, payload: dict) -> None:
    assert client.post("/ask", json=payload).status_code == 422


@pytest.mark.parametrize(
    ("err", "status", "words"),
    [
        (_rate_limit("Rate limit on tokens per day (TPD)"), 429, "daily"),
        (_rate_limit("Rate limit on tokens per minute (TPM)"), 429, "per-minute"),
        (service.SourceUnavailable("Neo4j is not reachable"), 503, "Neo4j"),
        (RuntimeError("password=secret in a stack trace"), 500, "Internal error"),
    ],
)
def test_errors_map_to_statuses_without_leaking_details(
    client: TestClient, err: Exception, status: int, words: str
) -> None:
    _use(failing(err))
    response = client.post("/ask", json={"question": "q"})
    assert response.status_code == status and words in response.json()["detail"]
    assert "secret" not in response.text


def _events(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        kind, data = block.split("\n")
        events.append((kind.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return events


def test_stream_sends_route_steps_answer_then_done(client: TestClient) -> None:
    response = client.post("/ask/stream", json={"question": "q", "source": "corrective"})
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _events(response.text)
    assert [kind for kind, _ in events] == ["route", "step", "answer", "done"]
    assert events[0][1] == {"source": "vector", "reason": "text question"}
    assert events[1][1] == {"stage": "corrective", "text": "grade 1/5 relevant, enough"}
    assert events[2][1]["source"] == "corrective" and events[2][1]["citations"][0]["pages"] == [26]


def test_stream_reports_errors_as_an_event(client: TestClient) -> None:
    _use(failing(_rate_limit("tokens per day")))
    events = _events(client.post("/ask/stream", json={"question": "q"}).text)
    assert events == [("error", {"status": 429, "detail": events[0][1]["detail"]})]
    assert "daily" in events[0][1]["detail"]


def test_events_outside_a_request_go_nowhere() -> None:
    emit("step", stage="x", text="no listener: nothing happens")


def test_health_and_meta_use_the_service(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service, "health", lambda resources: (True, False))
    monkeypatch.setattr(service, "meta", lambda resources: (["bce", "rbc"], [2023, 2024]))
    health = client.get("/health").json()
    assert (health["postgres"], health["neo4j"]) == (True, False)
    assert set(health) == {"postgres", "neo4j", "web_search", "self_rag", "password_required"}
    meta = client.get("/meta").json()
    assert meta["companies"] == ["bce", "rbc"] and "auto" in meta["sources"]


def test_graph_without_neo4j_is_unavailable() -> None:
    resources = service.Resources(pool=None, driver=None)  # type: ignore[arg-type]
    with pytest.raises(service.SourceUnavailable):
        service.answer("Who was RBC's CFO?", "graph", resources)


def test_the_ui_page_is_served_and_streams_from_the_api(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/html")
    page = response.text
    assert "<title>ReportLens</title>" in page and 'fetch("/ask/stream"' in page
    assert "innerHTML" not in page  # untrusted text is only ever inserted as text
