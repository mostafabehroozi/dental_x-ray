"""Offline tests for the hosted-model specs: key lookup, client construction, request fields."""
from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import dental_pipeline as dp
import llm_api
import location_adapter as la


class FakeClient:
    """OpenAI-style client that records every request and answers with a fixed text."""

    def __init__(self, text: str = "A. True"):
        self.text, self.requests = text, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request):
        self.requests.append(request)
        message = SimpleNamespace(content=self.text)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=None)


class SpecTests(unittest.TestCase):
    def test_resolve_reads_the_providers_secret(self):
        provider = {"base_url": "https://integrate.api.nvidia.com/v1", "api_key_env": "NVIDIA_API_KEY"}
        with mock.patch.dict(llm_api.PROVIDERS, {"nvidia": provider}, clear=True), \
             mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "nv-key"}):
            self.assertEqual(llm_api.resolve({"provider": "nvidia", "model": "m"}),
                             ("https://integrate.api.nvidia.com/v1", "nv-key"))

    def test_literal_key_and_custom_endpoint(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(llm_api.resolve({"provider": "openai", "model": "m", "api_key": "sk-test",
                                              "base_url": "https://api.openai.com/v1"}),
                             ("https://api.openai.com/v1", "sk-test"))
            self.assertEqual(llm_api.resolve({"base_url": "http://vllm:8000/v1", "api_key": "x"}), ("http://vllm:8000/v1", "x"))
        with mock.patch.dict(os.environ, {"MY_KEY": "k"}):
            self.assertEqual(llm_api.resolve({"base_url": "http://vllm:8000/v1", "api_key_env": "MY_KEY"}),
                             ("http://vllm:8000/v1", "k"))

    def test_configured_provider_owns_its_url_and_key(self):
        providers = {"private": {"base_url": "https://private.example/v1", "api_key": "private-key"}}
        with mock.patch.dict(llm_api.PROVIDERS, {}, clear=True):
            llm_api.configure_providers(providers)
            providers["private"]["api_key"] = "changed-after-configuration"
            self.assertEqual(llm_api.resolve({"provider": "private", "model": "vision-model"}),
                             ("https://private.example/v1", "private-key"))

    def test_errors_are_explicit(self):
        with self.assertRaises(ValueError):
            llm_api.resolve({"provider": "nope", "model": "m"})
        with self.assertRaises(ValueError):
            llm_api.resolve({"base_url": "http://vllm:8000/v1"})
        provider = {"base_url": "https://api.openai.com/v1", "api_key_env": "OPENAI_API_KEY"}
        with mock.patch.dict(llm_api.PROVIDERS, {"openai": provider}, clear=True), \
             mock.patch.dict(os.environ, {}, clear=True), mock.patch.dict("sys.modules", {"kaggle_secrets": None}):
            with self.assertRaises(RuntimeError):
                llm_api.resolve({"provider": "openai", "model": "m"})
            self.assertIsNone(llm_api.secret("OPENAI_API_KEY", required=False))
        with mock.patch.dict(llm_api.PROVIDERS,
                             {"empty": {"base_url": "https://empty.example/v1", "api_key": None}}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "provider 'empty' is empty"):
                llm_api.resolve({"provider": "empty", "model": "m"})

    def test_public_drops_the_key(self):
        spec = {"provider": "openai", "model": "m", "api_key": "sk-test", "request_options": {"a": 1}}
        self.assertEqual(llm_api.public(spec), {"provider": "openai", "model": "m", "request_options": {"a": 1}})

    def test_generation_fields(self):
        self.assertEqual(llm_api.generation_fields("max_tokens", 10, 0.0), {"max_tokens": 10, "temperature": 0.0})
        self.assertEqual(llm_api.generation_fields("max_completion_tokens", 10, None), {"max_completion_tokens": 10})


class FromApiTests(unittest.TestCase):
    def test_runner_from_spec(self):
        client = FakeClient()
        spec = {"provider": "openrouter", "model": "qwen/qwen3-vl", "api_key": "or-key",
                "base_url": "https://openrouter.ai/api/v1",
                "request_options": {"extra_body": {"provider": {"only": ["novita"]}}}}
        runner = dp.VisionRunner.from_api(spec, max_tokens=64, temperature=0.0, client=client)
        reply = runner.ask(b"\x89PNG", "Q?")
        self.assertEqual(reply["text"], "A. True")
        request = client.requests[0]
        self.assertEqual((request["model"], request["max_tokens"], request["temperature"]), ("qwen/qwen3-vl", 64, 0.0))
        self.assertEqual(request["extra_body"], {"provider": {"only": ["novita"]}})
        self.assertFalse(runner.local)
        self.assertNotIn("or-key", str(runner.settings()))

    def test_reasoning_model_spec_overrides_the_arguments(self):
        client = FakeClient()
        spec = {"provider": "openai", "model": "gpt-5", "api_key": "sk",
                "base_url": "https://api.openai.com/v1", "token_param": "max_completion_tokens",
                "temperature": None}
        runner = dp.VisionRunner.from_api(spec, max_tokens=64, temperature=0.0, client=client)
        runner.ask(b"\x89PNG", "Q?")
        request = client.requests[0]
        self.assertEqual(request["max_completion_tokens"], 64)
        self.assertNotIn("temperature", request)
        self.assertNotIn("max_tokens", request)
        with self.assertRaises(ValueError):
            dp.VisionRunner(token_param="max_new_tokens", client=client)

    def test_adapter_from_spec(self):
        client = FakeClient('{"boxes": []}')
        spec = {"provider": "nvidia", "model": "google/gemma-4-31b-it", "api_key": "nv-secret-123",
                "base_url": "https://integrate.api.nvidia.com/v1", "token_param": "max_tokens",
                "temperature": 0.0, "max_output_tokens": 2048, "max_boxes_per_call": 5}
        adapter = la.LLMAdapter.from_api(spec, client=client)
        self.assertEqual(adapter.name, "llm-google-gemma-4-31b-it")
        self.assertEqual((adapter.base_url, adapter.max_output_tokens, adapter.max_boxes_per_call),
                         ("https://integrate.api.nvidia.com/v1", 2048, 5))
        adapter._ask(b"\xff\xd8", "text")
        request = client.requests[0]
        self.assertEqual((request["max_tokens"], request["temperature"]), (2048, 0.0))
        self.assertNotIn("nv-secret-123", str(adapter.settings()))


if __name__ == "__main__":
    unittest.main()
