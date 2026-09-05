from __future__ import annotations

from typing import Any


def create_openai_client(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
):
    """Create an OpenAI client without forcing optional arguments into the SDK."""
    from openai import OpenAI

    client_kwargs: dict[str, Any] = {}
    if base_url:
        client_kwargs["base_url"] = base_url
    if api_key:
        client_kwargs["api_key"] = api_key
    if timeout is not None:
        client_kwargs["timeout"] = timeout
    if max_retries is not None:
        client_kwargs["max_retries"] = max_retries
    return OpenAI(**client_kwargs)


def vision_completion_result(response, latency_seconds: float) -> dict[str, Any]:
    """Normalize response metadata shared by local and API vision runners."""
    choice = response.choices[0]
    usage = getattr(response, "usage", None)
    result: dict[str, Any] = {
        "raw_answer": (choice.message.content or "").strip(),
        "latency_seconds": round(latency_seconds, 3),
        "finish_reason": choice.finish_reason,
        "truncated": choice.finish_reason == "length",
    }
    if usage is not None:
        result["prompt_tokens"] = usage.prompt_tokens
        result["completion_tokens"] = usage.completion_tokens
    return result
