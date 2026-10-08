import pytest
from pydantic import ValidationError

from reportlens.core.config import Settings, get_settings


def test_settings_read_from_env() -> None:
    settings = get_settings()
    assert settings.llm_model_main == "main-model"
    assert settings.llm_model_fast == "fast-model"
    assert settings.llm_api_key.get_secret_value() == "test-key"


def test_api_key_hidden_in_repr() -> None:
    assert "test-key" not in repr(get_settings())


def test_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_USER", "u")
    monkeypatch.setenv("POSTGRES_PASSWORD", "p")
    monkeypatch.setenv("POSTGRES_HOST", "h")
    monkeypatch.setenv("POSTGRES_PORT", "6543")
    monkeypatch.setenv("POSTGRES_DB", "d")
    assert Settings(_env_file=None).database_url == "postgresql://u:p@h:6543/d"


def test_missing_model_name_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_MODEL_MAIN")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_data_paths() -> None:
    settings = get_settings()
    assert settings.raw_dir == settings.data_dir / "raw"
    assert settings.parsed_dir == settings.data_dir / "parsed"
