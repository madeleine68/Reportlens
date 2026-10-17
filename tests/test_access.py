import datetime as dt

import pytest
from fastapi.testclient import TestClient

from reportlens.api import access, main
from reportlens.api.access import RateLimiter
from reportlens.api.schemas import AskResponse
from reportlens.core.config import get_settings


def stub_answer(question: str, source: str, resources: object) -> AskResponse:
    return AskResponse(
        question=question,
        answer="a",
        found=True,
        source=source,
        citations=[],
        chunks=[],
        model=None,
        input_tokens=0,
        output_tokens=0,
        latency_s=0.0,
    )


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch):
    """Set env vars for one test, with the settings cache cleared before and after."""

    def set_env(**values: str) -> None:
        for name, value in values.items():
            monkeypatch.setenv(name, value)
        get_settings.cache_clear()

    yield set_env
    get_settings.cache_clear()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(access, "limiter", RateLimiter())  # fresh counts per test
    monkeypatch.setattr(main.service, "health", lambda resources: (True, True))
    main.app.dependency_overrides[main.get_resources] = lambda: object()
    main.app.dependency_overrides[main.get_answer_fn] = lambda: stub_answer
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


def _ask(client: TestClient, password: str | None = None, ip: str = "1.1.1.1"):
    headers = {"x-forwarded-for": f"{ip}, 10.0.0.1"}
    if password is not None:
        headers["x-demo-password"] = password
    return client.post("/ask", json={"question": "q"}, headers=headers)


def test_without_a_password_set_anyone_may_ask(client: TestClient, settings) -> None:
    settings(DEMO_PASSWORD="")  # empty counts as unset, as in .env.example
    assert _ask(client).status_code == 200
    assert client.get("/health").json()["password_required"] is False


def test_the_password_is_checked_before_any_limit(client: TestClient, settings) -> None:
    settings(DEMO_PASSWORD="open sesame", DAILY_QUESTION_CAP="1")
    assert client.get("/health").json()["password_required"] is True
    assert _ask(client).status_code == 401
    assert _ask(client, "wrong").status_code == 401
    assert _ask(client, "open sesame").status_code == 200  # wrong tries used none of the cap
    assert _ask(client, "open sesame").status_code == 429


def test_the_stream_needs_the_password_too(client: TestClient, settings) -> None:
    settings(DEMO_PASSWORD="pw")
    assert client.post("/ask/stream", json={"question": "q"}).status_code == 401


def test_per_visitor_hourly_limit(client: TestClient, settings) -> None:
    settings(RATE_LIMIT_PER_HOUR="2")
    assert [_ask(client, ip="1.1.1.1").status_code for _ in range(3)] == [200, 200, 429]
    assert _ask(client, ip="2.2.2.2").status_code == 200  # another visitor has their own count
    refused = _ask(client, ip="1.1.1.1")
    assert "2 questions per hour" in refused.json()["detail"] and refused.headers["retry-after"]


def test_the_hour_rolls_and_the_day_resets() -> None:
    now, today = [0.0], [dt.date(2026, 10, 17)]
    limiter = RateLimiter(clock=lambda: now[0], today=lambda: today[0])
    assert limiter.allow("a", 1, 2)[0] and not limiter.allow("a", 1, 2)[0]
    now[0] = 3600.0
    assert limiter.allow("a", 1, 2)[0]  # an hour later: allowed again
    assert not limiter.allow("b", 1, 2)[0]  # but the day's cap of 2 is used up
    today[0] = dt.date(2026, 10, 18)
    assert limiter.allow("b", 1, 2)[0]


def test_hosted_urls_replace_the_local_parts(settings) -> None:
    settings(
        DATABASE_URL="postgresql://owner:s3cret@ep-x.neon.tech/reportlens?sslmode=require",
        NEO4J_URI="neo4j+s://abc.databases.neo4j.io",
        SQL_READER_PASSWORD="r/p@ss",
    )
    s = get_settings()
    assert s.database_url.startswith("postgresql://owner:s3cret@ep-x.neon.tech/")
    assert s.reader_database_url == (
        "postgresql://reportlens_reader:r%2Fp%40ss@ep-x.neon.tech/reportlens?sslmode=require"
    )
    assert s.neo4j_uri == "neo4j+s://abc.databases.neo4j.io"


def test_local_reader_url_keeps_the_port(settings) -> None:
    settings(POSTGRES_HOST="localhost", POSTGRES_PORT="5433", SQL_READER_PASSWORD="r")
    assert get_settings().reader_database_url.endswith(
        "reportlens_reader:r@localhost:5433/" + get_settings().postgres_db
    )


def test_the_daily_count_can_live_elsewhere() -> None:
    taken: list[int] = []

    def take_day(cap: int) -> bool:
        taken.append(cap)
        return len(taken) <= cap

    limiter = RateLimiter(clock=lambda: 0.0)
    assert limiter.allow("a", 1, 2, take_day)[0]
    assert not limiter.allow("a", 1, 2, take_day)[0]  # the hourly limit refuses first...
    assert taken == [2]  # ...without using a daily slot
    assert limiter.allow("b", 1, 2, take_day)[0]
    allowed, reason, _ = limiter.allow("c", 1, 2, take_day)
    assert not allowed and "daily" in reason


def test_a_database_failure_falls_back_to_counting_in_memory(monkeypatch) -> None:
    monkeypatch.setattr(access, "limiter", RateLimiter())

    def broken(cap: int) -> bool:
        raise OSError("Neon is down")

    take = access._fall_back_to_memory(broken)
    assert [take(2), take(2), take(2)] == [True, True, False]


def test_the_daily_count_in_postgres(conn) -> None:
    from reportlens.api.service import take_daily_slot

    conn.execute("TRUNCATE demo_usage")
    assert [take_daily_slot(conn, 2) for _ in range(3)] == [True, True, False]
    assert conn.execute("SELECT questions FROM demo_usage").fetchone()[0] == 2  # refusals uncounted
