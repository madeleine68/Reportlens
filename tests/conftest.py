import pytest

from reportlens.core import llm
from reportlens.core.config import get_settings


@pytest.fixture(autouse=True)
def fake_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every test a known LLM config and fresh cached singletons."""
    monkeypatch.setenv("LLM_BASE_URL", "http://fake-llm/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL_MAIN", "main-model")
    monkeypatch.setenv("LLM_MODEL_FAST", "fast-model")
    get_settings.cache_clear()
    llm._client.cache_clear()
