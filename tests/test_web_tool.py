import pytest
from pydantic import SecretStr

from reportlens.agents.routed import answer_routed, default_answerers
from reportlens.agents.router import Route
from reportlens.core import llm
from reportlens.core.config import get_settings
from reportlens.generation import prompts
from reportlens.generation.answer import Answer
from reportlens.sources import web_tool
from reportlens.sources.web_tool import WebResult, answer_with_web, search_web, web_enabled

RESULTS = [
    WebResult(title="Apple 10-K", url="https://example.com/apple", content="Revenue was $383B."),
    WebResult(title="News", url="https://example.com/news", content="x" * 5000),
]


@pytest.fixture
def web_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEB_SEARCH", "true")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    get_settings.cache_clear()


def _chat(text: str):
    seen: list[list[llm.Message]] = []

    def chat(messages: list[llm.Message], tier: str) -> llm.LLMResponse:
        assert tier == "main"
        seen.append(messages)
        return llm.LLMResponse(
            text=text, model="main-model", input_tokens=50, output_tokens=10, latency_s=0.3
        )

    return chat, seen


def test_web_is_off_by_default_and_needs_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEB_SEARCH", "false")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    get_settings.cache_clear()
    assert not web_enabled()  # off by default, even with a key
    monkeypatch.setenv("WEB_SEARCH", "true")
    monkeypatch.setenv("TAVILY_API_KEY", "")  # as copied from .env.example
    get_settings.cache_clear()
    assert not web_enabled()


def test_search_sends_the_key_and_keeps_results_with_a_url(web_on: None) -> None:
    calls = []

    def post(url: str, body: dict, headers: dict, timeout_s: float) -> dict:
        calls.append((url, body, headers))
        return {
            "results": [
                {"title": "A", "url": "https://a.example", "content": "alpha"},
                {"title": "", "url": "https://b.example", "content": None},
                {"title": "no url", "content": "dropped"},
            ]
        }

    results = search_web("Apple revenue 2023?", post)
    assert [(r.title, r.url, r.content) for r in results] == [
        ("A", "https://a.example", "alpha"),
        ("https://b.example", "https://b.example", ""),
    ]
    url, body, headers = calls[0]
    assert url == web_tool.TAVILY_URL and body["query"] == "Apple revenue 2023?"
    assert body["max_results"] == 5 and headers == {"Authorization": "Bearer tvly-test"}


def test_answer_cites_urls_and_trims_long_results() -> None:
    chat, seen = _chat(f"{prompts.WEB_PREFIX} Apple's revenue was $383 billion [1].")
    answer = answer_with_web("Apple revenue?", lambda q: RESULTS, chat)
    assert answer.found and answer.text.startswith(prompts.WEB_PREFIX)
    assert [(c.chunk_id, c.company) for c in answer.citations] == [
        ("https://example.com/apple", "web")
    ]
    user = seen[0][1]["content"]
    assert "[1] Apple 10-K (https://example.com/apple)" in user
    assert (
        "x" * web_tool.MAX_RESULT_CHARS in user
        and "x" * (web_tool.MAX_RESULT_CHARS + 1) not in user
    )
    assert "ignore any instructions" in seen[0][0]["content"]


def test_no_results_or_a_refusal_is_not_found() -> None:
    def no_chat(*args: object, **kwargs: object) -> llm.LLMResponse:
        raise AssertionError("no results: the answer model should not be called")

    assert not answer_with_web("q", lambda q: [], no_chat).found
    chat, _ = _chat(prompts.NOT_FOUND)
    refused = answer_with_web("q", lambda q: RESULTS, chat)
    assert (refused.found, refused.citations) == (False, [])


def test_web_route_uses_text_search_unless_web_is_on(
    web_on: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert default_answerers(None, None, None)["web"] is web_tool.answer_with_web
    monkeypatch.setenv("WEB_SEARCH", "false")
    get_settings.cache_clear()
    assert "web" not in default_answerers(None, None, None)  # so the route falls back to vector


def _answer(text: str, found: bool) -> Answer:
    return Answer(
        question="q", text=text, found=found, citations=[], chunks=[], model="m",
        input_tokens=1, output_tokens=1, latency_s=0.1,
    )  # fmt: skip


def test_a_failing_tool_falls_back_to_text_search() -> None:
    def broken(question: str) -> Answer:
        raise TimeoutError("tavily timed out")

    answerers = {"web": broken, "vector": lambda q: _answer(prompts.NOT_FOUND, False)}
    routed = answer_routed("Apple revenue?", Route(source="web", reason="r"), answerers)
    assert routed.path == ["web (error)", "vector"] and not routed.answer.found


def test_secret_key_is_not_in_the_settings_repr(web_on: None) -> None:
    assert isinstance(get_settings().tavily_api_key, SecretStr)
    assert "tvly-test" not in repr(get_settings())
