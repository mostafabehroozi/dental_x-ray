from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from dentalgpt import LLMVisionAnalysisRunner


class LLMVisionAnalysisRunnerTests(unittest.TestCase):
    def test_api_runner_sends_image_and_keeps_pipeline_result_contract(self) -> None:
        sent_request = {}

        def create_completion(**request):
            sent_request.update(request)
            return completion

        completion = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="<answer>PRESENT</answer>"),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=20, completion_tokens=4),
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create_completion)
            )
        )

        with tempfile.TemporaryDirectory() as tempdir:
            image_path = Path(tempdir) / "xray.png"
            Image.new("RGB", (8, 8), "white").save(image_path)

            with patch("dentalgpt._openai_client", return_value=fake_client) as openai:
                runner = LLMVisionAnalysisRunner(
                    model="vision-model",
                    base_url="https://provider.example/v1",
                    api_key="test-key",
                )
                result = runner.ask(str(image_path), "Analyze this X-ray.")

        openai.assert_called_once_with(
            base_url="https://provider.example/v1",
            api_key="test-key",
            timeout=600.0,
            max_retries=2,
        )
        self.assertEqual(runner.model_id, "vision-model")
        self.assertEqual(result["raw_answer"], "<answer>PRESENT</answer>")
        self.assertEqual(result["finish_reason"], "stop")
        self.assertFalse(result["truncated"])
        self.assertEqual(result["prompt_tokens"], 20)
        self.assertEqual(sent_request["model"], "vision-model")
        content = sent_request["messages"][0]["content"]
        self.assertTrue(content[0]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual(content[1]["text"], "Analyze this X-ray.")


if __name__ == "__main__":
    unittest.main()
