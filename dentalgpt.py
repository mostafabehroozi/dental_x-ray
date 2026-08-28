from __future__ import annotations

import base64
import mimetypes
import time
from pathlib import Path

from openai import OpenAI


class DentalExpertModelRunner:
    """Thin multimodal client for the configured dental expert model.

    The implementation currently targets an OpenAI-compatible llama.cpp server, while
    the role and public contract stay independent of the selected model:
        ask(image_path, question) -> dict
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        api_model: str = "dentalgpt",
        model_id: str = "mradermacher/DentalGPT-7B-1026-GGUF",
        max_tokens: int = 768,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 0,
        timeout: float = 600.0,
        cache_prompt: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_model = api_model
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.cache_prompt = cache_prompt
        self.client = OpenAI(
            base_url=f"{self.base_url}/v1",
            api_key="local-llama-cpp",
            timeout=timeout,
            max_retries=1,
        )

    @staticmethod
    def _image_data_uri(image_path: str | Path) -> str:
        path = Path(image_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)

        mime_type, _ = mimetypes.guess_type(path.name)
        if not mime_type or not mime_type.startswith("image/"):
            raise ValueError(
                f"Unsupported image type for {path.name}. Convert the radiograph to PNG/JPEG/WebP first."
            )

        payload = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{payload}"

    def ask(
        self,
        image_path: str,
        question: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> dict:
        image_uri = self._image_data_uri(image_path)
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        temperature = temperature if temperature is not None else self.temperature
        seed = seed if seed is not None else self.seed

        started = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.api_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        # Put the image first so repeated calls share the same large prefix.
                        {"type": "image_url", "image_url": {"url": image_uri}},
                        {"type": "text", "text": question},
                    ],
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=self.top_p,
            seed=seed,
            extra_body={
                # Preserve raw <think>/<answer> output instead of splitting reasoning server-side.
                "reasoning_format": "none",
                # Disabled by default for evaluation isolation; enable only after validating cache behavior.
                "cache_prompt": self.cache_prompt,
            },
        )
        latency = time.perf_counter() - started

        choice = response.choices[0]
        answer = (choice.message.content or "").strip()
        usage = getattr(response, "usage", None)

        result = {
            "raw_answer": answer,
            "latency_seconds": round(latency, 3),
            "finish_reason": choice.finish_reason,
            "truncated": choice.finish_reason == "length",
        }
        if usage is not None:
            result["prompt_tokens"] = usage.prompt_tokens
            result["completion_tokens"] = usage.completion_tokens
        return result


# Backward-compatible import for existing notebooks and downstream callers.
DentalGPTRunner = DentalExpertModelRunner
