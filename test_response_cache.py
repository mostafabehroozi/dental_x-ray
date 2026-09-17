"""Offline tests for exact local response reuse."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import dental_pipeline as dp
from response_cache import ResponseCache


class FakeClient:
    def __init__(self):
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request):
        self.requests.append(request)
        message = SimpleNamespace(content="Answer: B. False")
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=2)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=usage)


class ResponseCacheTests(unittest.TestCase):
    def runner(self, root, client, namespace=None, max_tokens=64):
        cache = ResponseCache(root, namespace or {"model": "dentalgpt", "image_max_tokens": 6144})
        return dp.VisionRunner(client=client, max_tokens=max_tokens, response_cache=cache)

    def test_identical_request_is_executed_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeClient()
            first_runner = self.runner(tmp, client)
            second_runner = self.runner(tmp, client)
            first = first_runner.ask(b"same-image", "same question")
            second = second_runner.ask(b"same-image", "same question")

            self.assertEqual(len(client.requests), 1)
            self.assertFalse(first["cache_hit"])
            self.assertTrue(second["cache_hit"])
            self.assertEqual(first["cache_key"], second["cache_key"])
            self.assertEqual((first_runner.requests, first_runner.calls, first_runner.cache_hits), (1, 1, 0))
            self.assertEqual((second_runner.requests, second_runner.calls, second_runner.cache_hits), (1, 0, 1))
            self.assertEqual(first["text"], second["text"])

    def test_cache_is_an_execution_detail_not_a_run_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeClient()
            cached = self.runner(tmp, client)
            uncached = dp.VisionRunner(client=client, max_tokens=64)
            self.assertEqual(cached.settings(), uncached.settings())

    def test_second_experiment_keeps_results_but_makes_no_duplicate_inferences(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp, "image.jpg")
            image.write_bytes(b"synthetic-image")
            client = FakeClient()
            protocol = dp.Protocol(presence_level="overall", counting=False)

            first = dp.analyze_image(self.runner(Path(tmp, "cache"), client), image, protocol=protocol)
            second = dp.analyze_image(self.runner(Path(tmp, "cache"), client), image, protocol=protocol)

            self.assertEqual(len(client.requests), len(dp.CONDITIONS))
            self.assertEqual((first["call_count"], first["inference_call_count"], first["cache_hit_count"]),
                             (14, 14, 0))
            self.assertEqual((second["call_count"], second["inference_call_count"], second["cache_hit_count"]),
                             (14, 0, 14))
            self.assertEqual(first["findings"], second["findings"])

    def test_prompt_image_generation_and_runtime_scope_do_not_cross(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeClient()
            self.runner(tmp, client).ask(b"image-a", "question-a")
            self.runner(tmp, client).ask(b"image-a", "question-b")
            self.runner(tmp, client).ask(b"image-b", "question-a")
            self.runner(tmp, client, max_tokens=32).ask(b"image-a", "question-a")
            self.runner(tmp, client, namespace={"model": "dentalgpt", "image_max_tokens": 4096}).ask(
                b"image-a", "question-a")
            self.assertEqual(len(client.requests), 5)

    def test_malformed_artifact_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = ResponseCache(tmp, {"model": "x"})
            key = cache.key({"request": "x"})
            path = Path(tmp, key[:2], f"{key}.json")
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"schema_version": 1, "key": key}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity mismatch|incomplete reply"):
                cache.get(key)


if __name__ == "__main__":
    unittest.main()
