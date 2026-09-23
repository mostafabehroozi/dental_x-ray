"""Hosted-model access shared by the model runner and the location adapter.

The notebook keeps one registry of OpenAI-compatible providers (base URL plus API key),
then assigns small model specs to roles such as ANALYZER and ADAPTER. A model spec is:

    {"provider": "openrouter", "model": "qwen/qwen3-vl-235b-a22b-thinking"}

Optional model-spec overrides are "api_key", "base_url" and "api_key_env",
"token_param" and "temperature"
("max_completion_tokens" and None for OpenAI reasoning models), "request_options" (extra
request fields, e.g. OpenRouter routing under "extra_body"). The key never reaches a
manifest: record public(spec), not the spec.
"""
from __future__ import annotations

import os
import time

import run_monitor as mon

# The console side of a run lives in run_monitor; these names stay here because every
# caller already reaches for them through llm_api.
monitor = mon.monitor
failure_details = mon.failure_details

PROVIDERS: dict[str, dict] = {}
TOKEN_PARAMS = ("max_tokens", "max_completion_tokens")


def secret(name: str, required: bool = True) -> str | None:
    """Value of the environment variable `name`, else of the Kaggle secret `name` (Add-ons > Secrets)."""
    value = os.environ.get(name)
    if not value:
        try:
            from kaggle_secrets import UserSecretsClient

            value = UserSecretsClient().get_secret(name)
        except Exception:  # not on Kaggle, or no secret of that name attached
            value = None
    if isinstance(value, str):
        value = value.strip()
    if not value and required:
        raise RuntimeError(f"secret {name!r} not found: export it as an environment variable, attach it as a "
                           f"Kaggle secret (Add-ons > Secrets), or put the key in the spec under 'api_key'")
    return value or None


def configure_providers(providers: dict[str, dict]) -> None:
    """Replace the provider registry with the notebook's single source of configuration."""
    if not isinstance(providers, dict) or not providers:
        raise ValueError("providers must be a non-empty dictionary")
    normalized = {}
    for name, row in providers.items():
        if not isinstance(name, str) or not name or not isinstance(row, dict):
            raise ValueError("each provider needs a non-empty name and a configuration dictionary")
        if not row.get("base_url"):
            raise ValueError(f"provider {name!r} needs a base_url")
        normalized[name] = dict(row)
    PROVIDERS.clear()
    PROVIDERS.update(normalized)


def resolve(spec: dict) -> tuple[str, str]:
    """(base_url, api_key) for a spec: the provider's endpoint and secret, unless the spec overrides them."""
    provider = spec.get("provider")
    row = PROVIDERS.get(provider, {})
    if not row and not spec.get("base_url"):
        raise ValueError(f"unknown provider {provider!r}: use one of {sorted(PROVIDERS)} or give base_url")
    base_url = spec.get("base_url") or row["base_url"]
    api_key = spec.get("api_key") or row.get("api_key")
    if not api_key:
        key_env = spec.get("api_key_env") or row.get("api_key_env")
        if not key_env:
            if provider in PROVIDERS:
                raise RuntimeError(f"API key for provider {provider!r} is empty; configure it in PROVIDERS")
            raise ValueError("spec needs 'api_key', or 'api_key_env' naming the environment variable / Kaggle secret")
        api_key = secret(key_env)
    return base_url, api_key


def connect(base_url: str | None, api_key: str, timeout: float = 600.0, max_retries: int = 0):
    """OpenAI client for one endpoint. Retries are controlled visibly by call_with_retries()."""
    from openai import OpenAI

    kwargs = {"api_key": api_key, "timeout": timeout, "max_retries": max_retries}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def client(spec: dict, timeout: float = 600.0):
    """OpenAI client for a spec (see the module docstring)."""
    base_url, api_key = resolve(spec)
    return connect(base_url, api_key, timeout)


def public(spec: dict) -> dict:
    """The spec without its key, for manifests, provenance and printouts."""
    return {k: v for k, v in spec.items() if k != "api_key"}


def generation_fields(token_param: str, max_tokens: int, temperature: float | None) -> dict:
    """The request fields that bound one completion; temperature None is left out (reasoning models)."""
    if token_param not in TOKEN_PARAMS:
        raise ValueError(f"token_param must be one of {TOKEN_PARAMS}")
    fields = {token_param: max_tokens}
    if temperature is not None:
        fields["temperature"] = temperature
    return fields


def validate_parse_retries(retries: int) -> None:
    if type(retries) is not int or retries < 0:
        raise ValueError("parse_retries must be a non-negative integer")


def validate_api_retries(retries: int) -> None:
    if type(retries) is not int or retries < 0:
        raise ValueError("api_call_retries must be a non-negative integer")


def artifact_error(path, reason: str) -> ValueError:
    monitor("ARTIFACT ERROR", str(path), reason=reason)
    return ValueError(f"invalid artifact {path}: {reason}")


def call_with_retries(call, retries: int, context: str):
    """Retry transient API/transport failures visibly; permanent request errors fail immediately."""
    validate_api_retries(retries)
    for attempt in range(retries + 1):
        try:
            return call()
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            permanent = status in {400, 401, 403, 404, 405, 422} or str(exc).startswith(
                ("refusal:", "content_filter", "non_text_content:"))
            if permanent or attempt == retries:
                mon.tally("api_failed")
                monitor("API FAILED", context, attempt=f"{attempt + 1}/{retries + 1}",
                        reason=type(exc).__name__, status=status,
                        kind="permanent" if permanent else "retries exhausted")
                mon.error_details(exc)  # the provider's own message is the only thing that explains it
                raise
            delay = min(2 ** attempt, 8)
            mon.tally("api_retry")
            monitor("API RETRY", context, attempt=f"{attempt + 1}/{retries + 1}",
                    reason=type(exc).__name__, status=status, wait=f"{delay}s",
                    detail=mon.clip(exc, 120) or None)
            time.sleep(delay)


def chat_reply(response) -> dict:
    """Validate an OpenAI-compatible response envelope and return normalized reply metadata."""
    choices = getattr(response, "choices", None)
    if not choices:
        raise RuntimeError("invalid_response_shape: empty_choices")
    choice = choices[0]
    message = getattr(choice, "message", None)
    if message is None:
        raise RuntimeError("invalid_response_shape: missing_message")
    refusal = getattr(message, "refusal", None)
    finish = getattr(choice, "finish_reason", None)
    if refusal:
        raise RuntimeError(f"refusal: {refusal}")
    if finish == "content_filter":
        raise RuntimeError("content_filter")
    content = getattr(message, "content", None)
    if content is None:
        content = ""
    if not isinstance(content, str):
        raise RuntimeError(f"non_text_content: {type(content).__name__}")
    usage = getattr(response, "usage", None)
    return {"text": content.strip(), "finish_reason": finish, "truncated": finish == "length",
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None)}


def ask_parsed(runner, image, question, *, parse, record, retries, context, fallback):
    """Retry format failures only. Keep all attempts and return (value, final reply).

    parse(reply) returns (value, error_or_None). A value may be partially usable.
    fallback(value) supplies the exact next question; it never sees ground truth.
    Transport errors propagate to the caller (the client owns transport retries).
    """
    validate_parse_retries(retries)
    for attempt in range(retries + 1):
        reply = dict(runner.ask(image, question))
        value, error = parse(reply)
        if error:
            if not reply.get("text", "").strip():
                error = "empty_response"
            elif reply.get("truncated"):
                error = "truncated_output"
            elif "<think>" in reply.get("text", "").lower() and "</think>" not in reply["text"].lower():
                error = "unclosed_think"
        exhausted = bool(error) and attempt == retries
        reply["parse_recovery"] = {
            "attempt": attempt + 1, "max_attempts": retries + 1,
            "error": error, "value": value,
            "status": "exhausted" if exhausted else "retrying" if error else "parsed",
            "recovered": not error and attempt > 0,
        }
        record(question, reply)
        if not error:
            if attempt:
                mon.tally("parse_recovered")
                monitor("PARSE RECOVERED", context, attempt=f"{attempt + 1}/{retries + 1}")
            return value, reply
        mon.tally("parse_warning")
        monitor("PARSE WARNING", context, attempt=f"{attempt + 1}/{retries + 1}",
                reason=error, finish=reply.get("finish_reason"))
        failure_details(question, reply.get("text", ""))
        if exhausted:
            mon.tally("parse_exhausted")
            monitor("PARSE EXHAUSTED", context, policy="neutral/excluded")
            return value, reply
        monitor("PARSE RETRY", context, action="format reminder")
        question = fallback(value)


def parse_recovery_summary(calls):
    attempts = [c["parse_recovery"] for c in calls if "parse_recovery" in c]
    return {
        "first_pass_failures": sum(a["attempt"] == 1 and bool(a["error"]) for a in attempts),
        "retry_calls": sum(a["attempt"] > 1 for a in attempts),
        "recovered_checks": sum(a["recovered"] for a in attempts),
        "unresolved_checks": sum(a["status"] == "exhausted" for a in attempts),
    }
