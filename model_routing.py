from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class ModelRouting:
    """Validate model roles and resolve their OpenAI-compatible providers."""

    def __init__(self, providers: Mapping[str, Mapping[str, Any]]) -> None:
        self.providers = providers

    def provider(self, provider_name: str) -> Mapping[str, Any]:
        if provider_name not in self.providers:
            raise ValueError(
                f"Unknown provider {provider_name!r}. Add it to PROVIDERS."
            )
        return self.providers[provider_name]

    def provider_base_url(self, provider_name: str) -> str | None:
        return self.provider(provider_name).get("base_url")

    def provider_api_key(self, provider_name: str) -> str:
        api_key = self.provider(provider_name).get("api_key")
        if not api_key:
            raise ValueError(
                f"The API key for provider {provider_name!r} is not configured."
            )
        return str(api_key)

    @staticmethod
    def model_backend(model_config: Mapping[str, Any], usage_name: str) -> str:
        if not isinstance(model_config, Mapping):
            raise ValueError(
                f"{usage_name} must be a model configuration dictionary."
            )
        backend = str(model_config.get("backend", "")).lower()
        if backend not in {"local", "api"}:
            raise ValueError(f"{usage_name}.backend must be 'local' or 'api'.")
        if (
            backend == "local"
            and str(model_config.get("model", "")).lower() != "dentalgpt"
        ):
            raise ValueError(
                f"{usage_name}.model must be 'DentalGPT' for backend='local'."
            )
        return backend

    def api_model(
        self,
        model_config: Mapping[str, Any],
        usage_name: str,
    ) -> tuple[str, str]:
        if self.model_backend(model_config, usage_name) != "api":
            raise ValueError(f"{usage_name} must use backend='api' here.")
        model_name = model_config.get("model")
        provider_name = model_config.get("provider")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(f"{usage_name}.model must be a non-empty string.")
        if not isinstance(provider_name, str) or not provider_name.strip():
            raise ValueError(f"{usage_name}.provider must be a non-empty string.")
        self.provider(provider_name)
        return model_name, provider_name
