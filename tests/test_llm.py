from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from openai import BadRequestError
from pydantic import BaseModel, ValidationError

from reportlens.core import llm


def _fake_completion(text: str, prompt_tokens: int = 10, completion_tokens: int = 5):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    client = MagicMock()
    monkeypatch.setattr(llm, "_client", lambda: client)
    return client


class Filters(BaseModel):
    company: str
    year: int


def test_chat_uses_main_model_by_default(fake_client: MagicMock) -> None:
    fake_client.chat.completions.create.return_value = _fake_completion("hello")

    result = llm.chat([{"role": "user", "content": "hi"}])

    assert result.text == "hello"
    assert result.model == "main-model"
    assert (result.input_tokens, result.output_tokens) == (10, 5)
    kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "main-model"
    assert "response_format" not in kwargs


def test_chat_fast_tier_and_json_mode(fake_client: MagicMock) -> None:
    fake_client.chat.completions.create.return_value = _fake_completion("{}")

    llm.chat([{"role": "user", "content": "hi"}], tier="fast", json_mode=True)

    kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "fast-model"
    assert kwargs["response_format"] == {"type": "json_object"}


def test_chat_handles_missing_usage(fake_client: MagicMock) -> None:
    completion = _fake_completion("ok")
    completion.usage = None
    fake_client.chat.completions.create.return_value = completion

    result = llm.chat([{"role": "user", "content": "hi"}])

    assert (result.input_tokens, result.output_tokens) == (0, 0)


def test_structured_returns_validated_model(fake_client: MagicMock) -> None:
    fake_client.chat.completions.create.return_value = _fake_completion(
        '{"company": "RBC", "year": 2023}'
    )

    result = llm.structured([{"role": "user", "content": "RBC 2023"}], Filters)

    assert result == Filters(company="RBC", year=2023)
    kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "fast-model"
    assert "JSON schema" in kwargs["messages"][0]["content"]


def test_structured_retries_once_on_invalid_json(fake_client: MagicMock) -> None:
    fake_client.chat.completions.create.side_effect = [
        _fake_completion('{"company": "RBC"}'),
        _fake_completion('{"company": "RBC", "year": 2023}'),
    ]

    result = llm.structured([{"role": "user", "content": "RBC 2023"}], Filters)

    assert result.year == 2023
    retry_messages = fake_client.chat.completions.create.call_args.kwargs["messages"]
    assert "invalid" in retry_messages[-1]["content"]


def test_structured_raises_after_retries_exhausted(fake_client: MagicMock) -> None:
    fake_client.chat.completions.create.return_value = _fake_completion("not json")

    with pytest.raises(ValidationError):
        llm.structured([{"role": "user", "content": "x"}], Filters)
    assert fake_client.chat.completions.create.call_count == 2


def _json_rejected(code: str = "json_validate_failed") -> BadRequestError:
    response = MagicMock(status_code=400, headers={})
    return BadRequestError("rejected", response=response, body={"code": code})


def test_structured_retries_when_provider_rejects_json(fake_client: MagicMock) -> None:
    fake_client.chat.completions.create.side_effect = [
        _json_rejected(),
        _fake_completion('{"company": "RBC", "year": 2023}'),
    ]

    result = llm.structured([{"role": "user", "content": "RBC 2023"}], Filters)

    assert result.year == 2023
    assert fake_client.chat.completions.create.call_count == 2


def test_structured_does_not_retry_other_bad_requests(fake_client: MagicMock) -> None:
    fake_client.chat.completions.create.side_effect = _json_rejected(code="model_not_found")

    with pytest.raises(BadRequestError):
        llm.structured([{"role": "user", "content": "x"}], Filters)
    assert fake_client.chat.completions.create.call_count == 1
