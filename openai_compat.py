from __future__ import annotations

import threading
import time
import math
from collections.abc import Callable
from typing import Any


DEFAULT_API_CALL_DELAY_SECONDS = 1.0
DEFAULT_API_CALL_MAX_RETRIES = 10


class APICallExhaustedError(RuntimeError):
    """Raised after one API request and all configured retries have failed."""


class _APICallPacer:
    """Keep request starts separated even when multiple runners share a process."""

    _lock = threading.Lock()
    _next_start = 0.0

    @classmethod
    def wait(cls, delay_seconds: float) -> None:
        with cls._lock:
            remaining = cls._next_start - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            cls._next_start = time.monotonic() + delay_seconds


class APICallController:
    """Apply visible pacing and retry behavior to an API request callable."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        delay_seconds: float = DEFAULT_API_CALL_DELAY_SECONDS,
        max_retries: int = DEFAULT_API_CALL_MAX_RETRIES,
        log_calls: bool = True,
    ) -> None:
        if isinstance(delay_seconds, bool) or not isinstance(delay_seconds, (int, float)):
            raise TypeError("API call delay must be a number of seconds.")
        if not math.isfinite(delay_seconds) or delay_seconds < 0:
            raise ValueError("API call delay must be finite and non-negative.")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int):
            raise TypeError("API call max retries must be an integer.")
        if max_retries < 0:
            raise ValueError("API call max retries must be non-negative.")
        self.provider = str(provider or "unknown")
        self.model = str(model or "unknown")
        self.delay_seconds = float(delay_seconds)
        self.max_retries = max_retries
        if type(log_calls) is not bool:
            raise TypeError("API call logging must be boolean.")
        self.log_calls = log_calls

    def call(self, request: Callable[[], Any], operation: str) -> Any:
        total_attempts = self.max_retries + 1
        for attempt in range(1, total_attempts + 1):
            _APICallPacer.wait(self.delay_seconds)
            started = time.perf_counter()
            if self.log_calls:
                print(
                    f"API CALL | provider={self.provider} | model={self.model} | "
                    f"operation={operation} | attempt={attempt}/{total_attempts}"
                )
            try:
                result = request()
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                print(
                    f"API ERROR | provider={self.provider} | model={self.model} | "
                    f"operation={operation} | attempt={attempt}/{total_attempts} | {error}"
                )
                if attempt == total_attempts:
                    message = (
                        f"API call failed for provider={self.provider}, model={self.model} "
                        f"after {total_attempts} attempts ({self.max_retries} retries): {error}"
                    )
                    print(f"API FATAL | {message}")
                    raise APICallExhaustedError(message) from exc
                print(
                    f"API RETRY | provider={self.provider} | model={self.model} | "
                    f"retry={attempt}/{self.max_retries}"
                )
                continue
            if self.log_calls:
                print(
                    f"API SUCCESS | provider={self.provider} | model={self.model} | "
                    f"operation={operation} | attempt={attempt}/{total_attempts} | "
                    f"latency={time.perf_counter() - started:.3f}s"
                )
            return result

        raise AssertionError("Unreachable API retry state")


def create_openai_client(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = 0,
):
    """Create an OpenAI client; callers normally use zero SDK retries.

    Visible retries are handled by APICallController so every failed attempt is
    logged consistently and the final provider error is not hidden by the SDK.
    """
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
    for field in ("model", "id"):
        if getattr(response, field, None) is not None:
            result[f"response_{field}"] = getattr(response, field)
    return result
