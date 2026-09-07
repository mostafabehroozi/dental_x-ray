"""Offline tests for visible API pacing and retries; no provider calls."""

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from openai_compat import APICallController, APICallExhaustedError


class APICallControllerTests(unittest.TestCase):
    def test_retries_every_exception_then_returns_success(self):
        calls = []

        def request():
            calls.append(len(calls) + 1)
            if len(calls) < 3:
                raise ValueError(f"temporary-{len(calls)}")
            return "ok"

        controller = APICallController(
            provider="test-provider",
            model="test-model",
            delay_seconds=1,
            max_retries=10,
        )
        output = io.StringIO()
        with patch("openai_compat._APICallPacer.wait") as wait, redirect_stdout(output):
            self.assertEqual(controller.call(request, "test.operation"), "ok")

        self.assertEqual(calls, [1, 2, 3])
        self.assertEqual(wait.call_count, 3)
        wait.assert_called_with(1.0)
        console = output.getvalue()
        self.assertIn("provider=test-provider", console)
        self.assertIn("ValueError: temporary-1", console)
        self.assertIn("API RETRY", console)
        self.assertIn("API SUCCESS", console)

    def test_exhaustion_prints_provider_error_and_raises(self):
        controller = APICallController(
            provider="broken-provider",
            model="broken-model",
            delay_seconds=0,
            max_retries=2,
        )
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(APICallExhaustedError) as raised:
            controller.call(lambda: (_ for _ in ()).throw(ConnectionError("offline")), "create")

        message = str(raised.exception)
        self.assertIn("provider=broken-provider", message)
        self.assertIn("after 3 attempts (2 retries)", message)
        self.assertIn("ConnectionError: offline", message)
        self.assertIn("API FATAL", output.getvalue())

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            APICallController(provider="p", model="m", delay_seconds=-1)
        with self.assertRaises(ValueError):
            APICallController(provider="p", model="m", max_retries=-1)
        with self.assertRaises(TypeError):
            APICallController(provider="p", model="m", max_retries=True)
        with self.assertRaises(TypeError):
            APICallController(provider="p", model="m", log_calls="yes")


if __name__ == "__main__":
    unittest.main()
