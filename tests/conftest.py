from collections.abc import Iterator

import psycopg
import pytest

from reportlens.core import llm
from reportlens.core.config import get_settings
from reportlens.core.db import connect

TEST_DB = "reportlens_test"


@pytest.fixture(autouse=True)
def fake_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every test a known LLM config and fresh cached singletons."""
    monkeypatch.setenv("LLM_BASE_URL", "http://fake-llm/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL_MAIN", "main-model")
    monkeypatch.setenv("LLM_MODEL_FAST", "fast-model")
    get_settings.cache_clear()
    llm._client.cache_clear()


@pytest.fixture
def conn(monkeypatch: pytest.MonkeyPatch) -> Iterator[psycopg.Connection]:
    """Connection to an empty reportlens_test database; skips the test if Postgres is down."""
    try:
        with psycopg.connect(get_settings().database_url, autocommit=True) as admin:
            exists = admin.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DB,)
            ).fetchone()
            if not exists:
                admin.execute(f"CREATE DATABASE {TEST_DB}")
    except psycopg.OperationalError:
        pytest.skip("Postgres not reachable; start it with `docker compose up -d`")

    monkeypatch.setenv("POSTGRES_DB", TEST_DB)
    get_settings.cache_clear()
    with connect() as test_conn:
        test_conn.execute("TRUNCATE documents CASCADE")
        yield test_conn
