"""Single entry point for every LLM call, so timing and token usage are logged in one place.

Talks to any OpenAI-compatible endpoint (Groq, Gemini, Ollama, ...) chosen via .env.
"""

import json
import time
from functools import lru_cache
from typing import Literal

from openai import BadRequestError, OpenAI
from pydantic import BaseModel, ValidationError

from reportlens.core.config import get_settings
from reportlens.core.logging import get_logger

log = get_logger(__name__)

Tier = Literal["main", "fast"]
Message = dict[str, str]


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
    return settings.llm_model_main if tier == "main" else settings.llm_model_fast


def chat(
    messages: list[Message],
    tier: Tier = "main",
    temperature: float = 0.0,
    json_mode: bool = False,
) -> LLMResponse:
    """Send a chat completion and return the reply with usage stats."""
    model = _model_for(tier)
    extra = {"response_format": {"type": "json_object"}} if json_mode else {}

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
) -> T:
    """Ask for JSON matching `schema` and return a validated Pydantic object.

    Uses JSON mode (widely supported) plus Pydantic validation, retrying once with the
    validation error fed back to the model. Also retries when the provider itself rejects
    malformed JSON (Groq returns a 400 `json_validate_failed`).
    """
    schema_json = json.dumps(schema.model_json_schema())
    system = {
        "role": "system",
        "content": f"Respond only with a JSON object matching this JSON schema:\n{schema_json}",
    }
    convo = [system, *messages]

    for attempt in range(max_retries + 1):
        try:
            reply = chat(convo, tier=tier, json_mode=True)
        except BadRequestError as err:
            if err.code != "json_validate_failed" or attempt == max_retries:
                raise
            log.warning("llm.structured provider rejected JSON (attempt %d)", attempt + 1)
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
