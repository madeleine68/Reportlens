"""Single entry point for every LLM call, so timing and token usage are logged in one place.

Talks to any OpenAI-compatible endpoint (Groq, Gemini, Ollama, ...) chosen via .env.
"""

import base64
import json
import time
from functools import lru_cache
from typing import Any, Literal

from openai import BadRequestError, OpenAI, RateLimitError
from pydantic import BaseModel, ValidationError

from reportlens.core.config import get_settings
from reportlens.core.logging import get_logger

log = get_logger(__name__)

Tier = Literal["main", "fast", "vision"]
Message = dict[str, Any]  # content is a string, or a list of text and image parts (vision)


def is_daily_limit(err: BaseException) -> bool:
    """True for the provider's tokens-per-day limit (wait hours), not the per-minute one."""
    return isinstance(err, RateLimitError) and "per day" in str(err)


class LLMResponse(BaseModel):
    """Text plus the usage numbers we track for cost and latency."""

    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float


@lru_cache
def _client() -> OpenAI:
    settings = get_settings()
    return OpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key.get_secret_value(),
        max_retries=settings.llm_max_retries,
    )


def _model_for(tier: Tier) -> str:
    settings = get_settings()
    models = {
        "main": settings.llm_model_main,
        "fast": settings.llm_model_fast,
        "vision": settings.llm_model_vision,
    }
    return models[tier]


def image_message(text: str, png: bytes) -> Message:
    """A user message holding text plus one PNG image, for the vision tier."""
    url = "data:image/png;base64," + base64.b64encode(png).decode()
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": url}},
        ],
    }


def chat(
    messages: list[Message],
    tier: Tier = "main",
    temperature: float = 0.0,
    json_mode: bool = False,
    max_tokens: int | None = None,
) -> LLMResponse:
    """Send a chat completion and return the reply with usage stats.

    max_tokens caps the reply, reasoning included; None uses the provider's default.
    """
    model = _model_for(tier)
    extra: dict = {"response_format": {"type": "json_object"}} if json_mode else {}
    if max_tokens is not None:
        extra["max_tokens"] = max_tokens

    start = time.perf_counter()
    resp = _client().chat.completions.create(
        model=model, messages=messages, temperature=temperature, **extra
    )
    latency = time.perf_counter() - start

    usage = resp.usage
    result = LLMResponse(
        text=resp.choices[0].message.content or "",
        model=model,
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
        latency_s=round(latency, 3),
    )
    log.info(
        "llm.chat model=%s in=%d out=%d latency=%.2fs",
        model,
        result.input_tokens,
        result.output_tokens,
        result.latency_s,
    )
    return result


def structured[T: BaseModel](
    messages: list[Message],
    schema: type[T],
    tier: Tier = "fast",
    max_retries: int = 1,
    max_tokens: int | None = None,
) -> T:
    """Ask for JSON matching `schema` and return a validated Pydantic object.

    Uses JSON mode (widely supported) plus Pydantic validation, retrying once with the
    validation error fed back to the model. When the provider itself rejects the reply (Groq
    returns a 400 `json_validate_failed`), retries without JSON mode: some inputs fail Groq's
    check every time while the plain reply is valid JSON, and Pydantic still validates it.
    """
    schema_json = json.dumps(schema.model_json_schema())
    system = {
        "role": "system",
        "content": f"Respond only with a JSON object matching this JSON schema:\n{schema_json}",
    }
    convo = [system, *messages]
    json_mode = True

    for attempt in range(max_retries + 1):
        try:
            reply = chat(convo, tier=tier, json_mode=json_mode, max_tokens=max_tokens)
        except BadRequestError as err:
            if err.code != "json_validate_failed" or attempt == max_retries:
                raise
            log.warning("llm.structured provider rejected JSON (attempt %d)", attempt + 1)
            json_mode = False
            continue
        try:
            return schema.model_validate_json(reply.text)
        except ValidationError as err:
            log.warning("llm.structured invalid output (attempt %d): %s", attempt + 1, err)
            if attempt == max_retries:
                raise
            convo = [
                *convo,
                {"role": "assistant", "content": reply.text},
                {"role": "user", "content": f"That JSON was invalid: {err}. Try again."},
            ]
    raise AssertionError("unreachable")
