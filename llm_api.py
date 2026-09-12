"""Hosted-model access shared by the model runner and the location adapter.

One table of OpenAI-compatible providers, one place that finds the key (an environment
variable, else the Kaggle secret of the same name), one function that builds the client.
A model is described by a small spec dict, edited in the notebook's configuration cell:

    {"provider": "openrouter", "model": "qwen/qwen3-vl-235b-a22b-thinking"}

Optional keys: "api_key" (paste one for a quick test; never commit it), "base_url" and
"api_key_env" (any other OpenAI-compatible endpoint), "token_param" and "temperature"
("max_completion_tokens" and None for OpenAI reasoning models), "request_options" (extra
request fields, e.g. OpenRouter routing under "extra_body"). The key never reaches a
manifest: record public(spec), not the spec.
"""
from __future__ import annotations

import os

PROVIDERS = {
    "openai": {"base_url": "https://api.openai.com/v1", "api_key_env": "OPENAI_API_KEY"},
    "nvidia": {"base_url": "https://integrate.api.nvidia.com/v1", "api_key_env": "NVIDIA_API_KEY"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY"},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "api_key_env": "GEMINI_API_KEY"},
}
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
    if not value and required:
        raise RuntimeError(f"secret {name!r} not found: export it as an environment variable, attach it as a "
                           f"Kaggle secret (Add-ons > Secrets), or put the key in the spec under 'api_key'")
    return value or None


def resolve(spec: dict) -> tuple[str, str]:
    """(base_url, api_key) for a spec: the provider's endpoint and secret, unless the spec overrides them."""
    provider = spec.get("provider")
    row = PROVIDERS.get(provider, {})
    if not row and not spec.get("base_url"):
        raise ValueError(f"unknown provider {provider!r}: use one of {sorted(PROVIDERS)} or give base_url")
    base_url = spec.get("base_url") or row["base_url"]
    api_key = spec.get("api_key")
    if not api_key:
        key_env = spec.get("api_key_env") or row.get("api_key_env")
        if not key_env:
            raise ValueError("spec needs 'api_key', or 'api_key_env' naming the environment variable / Kaggle secret")
        api_key = secret(key_env)
    return base_url, api_key


def connect(base_url: str | None, api_key: str, timeout: float = 600.0, max_retries: int = 2):
    """OpenAI client for one endpoint. The SDK retries connection errors, 408/409/429 and 5xx with backoff."""
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
