"""Offline contract tests; never call providers or require a GPU."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from prompts import CONDITIONS
from research_prompts import DEFAULT_STRATEGIES
from research_experiments import prepare_suite, run_suite, load_suite, parse_answer, local_preset, question
from evaluation import evaluate_research_suite
from dentalgpt import LLMVisionAnalysisRunner


class FakeRunner:
    def __init__(self, callback=None):
        self.calls = []
        self.callback = callback

    def ask(self, image, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        if self.callback:
            return self.callback(prompt, kwargs, len(self.calls))
        payload = dict.fromkeys(CONDITIONS, 0) if "Counting categories:" in prompt else {"choice": "B", "count": 0}
        return answer(payload)


def answer(value):
    return {"raw_answer": "<answer>" + (json.dumps(value) if isinstance(value, dict) else str(value)) + "</answer>",
            "truncated": False, "prompt_tokens": 10, "completion_tokens": 5}


class SuiteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.image = self.root / "x.png"
        self.image.write_bytes(b"synthetic image for mocked requests")
        self.label = self.root / "x.txt"
        self.label.write_text("0 .5 .5 .1 .1\n0 .6 .5 .1 .1\n")
        self.providers = {"p": {"api_key": "SECRET", "base_url": "https://example.invalid/v1"}}
        self.models = {"a": {"backend": "api", "provider": "p", "model": "vision"}}
        self.experiments = [{"id": "a", "model": "a", "strategies": ["broad_whole", "atomic_whole", "atomic_arch"]}]
        self.images = [{"id": "x", "image_path": str(self.image), "label_path": str(self.label)}]

    def tearDown(self):
        self.tmp.cleanup()

    def plan(self, **kwargs):
        return prepare_suite(self.providers, self.models, DEFAULT_STRATEGIES, self.experiments, self.images, **kwargs)

    def run_fake(self, plan, runner):
        return run_suite(plan, self.providers, self.root / "runs", runner_factory=lambda job: runner)

    def test_three_strategies_calls_resume_report(self):
        plan, runner = self.plan(), FakeRunner()
        self.assertEqual(sum(j["min_calls"] for j in plan["jobs"]), 43)
        directory = self.run_fake(plan, runner)
        self.assertEqual(len(runner.calls), 43)
        self.assertNotIn("SECRET", (Path(directory) / "manifest.json").read_text())
        run_suite(plan, self.providers, self.root, resume_dir=directory, runner_factory=lambda j: self.fail("called on resume"))
        rows = evaluate_research_suite(load_suite(directory), Path(directory) / "evaluation")["summary"]
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["FN"], 1)
        self.assertEqual(rows[0]["TN"], 13)
        self.assertEqual(rows[0]["missed_count"], 2)
        self.assertEqual(rows[0]["count_mae"], 2 / 14)
        self.assertIsNone(rows[0]["estimated_cost"])

    def test_multiple_models_images_and_arch_sum(self):
        self.providers["q"] = {"api_key_env": "TEST_KEY"}
        self.models.update(b={"backend": "api", "provider": "q", "model": "other"},
                           local={"backend": "local", "model": "DentalGPT", "preset": "QUALITY"})
        self.experiments = [{"id": k, "model": k, "strategies": ["atomic_arch"]} for k in self.models]
        self.images.append({**self.images[0], "id": "y"})
        plan = self.plan(local_runtime={"base_url": "http://localhost", "api_model": "dentalgpt"})
        runner = FakeRunner(lambda *args: answer({"choice": "A", "count": 2}))
        results = load_suite(self.run_fake(plan, runner))
        self.assertEqual(len(results), 6)
        self.assertEqual(len({r["id"] for r in results}), 6)
        self.assertEqual(results[0]["prediction_counts_by_condition"][CONDITIONS[0]], 4)
        self.assertEqual(results[0]["region_counts_by_condition"]["maxilla"][CONDITIONS[0]], 2)
        rows = evaluate_research_suite(results)["summary"]
        self.assertTrue(all(r["completed_images"] == 2 for r in rows))

    def test_recovery_order_then_forced_b(self):
        self.experiments[0]["strategies"] = ["broad_whole"]
        runner = FakeRunner(lambda *args: {"raw_answer": "bad", "truncated": False})
        result = load_suite(self.run_fake(self.plan(), runner))[0]
        self.assertEqual(len(runner.calls), 9)
        self.assertEqual([c[1]["temperature"] for c in runner.calls], [0, .2, .4] * 3)
        self.assertEqual([a["template_id"] for a in result["attempts"]], ["broad_1"]*3 + ["broad_2"]*3 + ["broad_3"]*3)
        self.assertTrue(result["checks"][0]["forced_zero"])
        self.assertEqual(sum(result["prediction_counts_by_condition"].values()), 0)
        self.assertIsNone(evaluate_research_suite([result])["summary"][0]["prompt_tokens"])

    def test_truncation_and_successful_fallback(self):
        self.experiments[0]["strategies"] = ["broad_whole"]
        def callback(prompt, kwargs, number):
            result = answer(dict.fromkeys(CONDITIONS, 1))
            result["truncated"] = number <= 3
            return result
        result = load_suite(self.run_fake(self.plan(), FakeRunner(callback)))[0]
        self.assertEqual(len(result["attempts"]), 4)
        self.assertFalse(result["checks"][0]["forced_zero"])
        self.assertEqual(result["attempts"][-1]["template_id"], "broad_2")

    def test_two_step_conditional_counts(self):
        self.experiments[0].update(strategies=["atomic_whole"], atomic_protocol="presence_then_count")
        def callback(prompt, kwargs, number):
            return answer(2 if "How many" in prompt else "A")
        runner = FakeRunner(callback)
        results = load_suite(self.run_fake(self.plan(), runner))
        self.assertEqual(len(runner.calls), 28)
        self.assertTrue(all(c == 2 for c in results[0]["prediction_counts_by_condition"].values()))
        runner = FakeRunner(lambda *args: answer("B"))
        self.run_fake(self.plan(), runner)
        self.assertEqual(len(runner.calls), 14)

    def test_auth_failure_does_not_become_negative(self):
        class Unauthorized(Exception):
            status_code = 401
        def callback(*args):
            raise Unauthorized("sensitive text")
        runner = FakeRunner(callback)
        results = load_suite(self.run_fake(self.plan(), runner))
        self.assertEqual(len(runner.calls), 1)
        self.assertTrue(all(r["status"] == "failed" for r in results))
        rows = evaluate_research_suite(results)["summary"]
        self.assertTrue(all(r["TN"] == 0 and r["coverage"] == 0 for r in rows))
        self.assertNotIn("sensitive text", json.dumps(results))

    def test_resume_changed_input_and_missing_labels(self):
        plan = self.plan()
        directory = self.run_fake(plan, FakeRunner())
        self.image.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Resume mismatch"):
            run_suite(self.plan(), self.providers, self.root, resume_dir=directory)
        self.label.unlink()
        with self.assertRaises(FileNotFoundError):
            self.plan()

    def test_interrupted_resume_and_pending_report(self):
        self.experiments[0]["strategies"] = ["broad_whole"]
        def callback(*args):
            raise KeyboardInterrupt()
        plan = self.plan()
        with self.assertRaises(KeyboardInterrupt):
            self.run_fake(plan, FakeRunner(callback))
        directory = next((self.root / "runs").iterdir())
        self.assertEqual(evaluate_research_suite(load_suite(directory))["summary"][0]["coverage"], 0)
        runner = FakeRunner()
        run_suite(plan, self.providers, self.root, resume_dir=directory, runner_factory=lambda j: runner)
        self.assertEqual(len(runner.calls), 1)
        self.assertEqual(len(load_suite(directory)[0]["attempts"]), 2)

    def test_parser_rejects_ambiguous_values(self):
        for raw in ['<answer>{"choice":"A","count":0}</answer>',
                    '<answer>{"choice":"B","count":true}</answer>',
                    '<answer>{"choice":"B","count":0,"count":1}</answer>',
                    '<answer>{"choice":"B","count":0}</answer><answer>B</answer>']:
            with self.assertRaises(ValueError):
                parse_answer(raw, "combined", CONDITIONS)
        with self.assertRaises(ValueError):
            parse_answer('<answer>{"dental_implant":0}</answer>', "broad", CONDITIONS)

    def test_request_options_and_no_sdk_retries(self):
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")], usage=None)
        with patch("dentalgpt._openai_client") as client:
            client.return_value.chat.completions.create.return_value = response
            runner = LLMVisionAnalysisRunner(model="m", max_retries=0, omit_parameters=("temperature", "top_p"), token_limit_parameter="max_completion_tokens")
            result = runner.ask(str(self.image), "question", temperature=.4)
            request = client.return_value.chat.completions.create.call_args.kwargs
            self.assertNotIn("temperature", request)
            self.assertIn("max_completion_tokens", request)
            self.assertEqual(client.call_args.kwargs["max_retries"], 0)
            self.assertNotIn("messages", result["effective_request_settings"])

    def test_local_preset_validation(self):
        self.assertIsNone(local_preset(self.models, self.experiments))
        self.models.update(l={"backend": "local", "preset": "QUALITY"}, m={"backend": "local", "preset": "FAST"})
        with self.assertRaises(ValueError):
            local_preset(self.models, [{"model": "l"}, {"model": "m"}])

    def test_paper_question_forms_and_regional_adaptations(self):
        self.experiments[0].update(strategies=["atomic_whole", "atomic_arch"], atomic_protocol="presence_then_count")
        whole, arch = self.plan()["jobs"]
        presence = question(whole, "presence", "atomic_1", "impacted_tooth", whole["regions"][0][1])
        self.assertTrue(presence.startswith("Kindly evaluate if the condition 'Impacted tooth' is present in this image.\nA. True\nB. False"))
        self.assertIn("<think>", presence)
        self.assertNotIn("Counting unit", presence)
        self.assertNotIn("boundary", presence)
        self.assertNotIn("Briefly", presence)
        count = question(whole, "count", "atomic_1", "dental_filling", whole["regions"][0][1])
        self.assertTrue(count.startswith("How many visible teeth in the image appear to have dental fillings based on their radiopaque characteristics?"))
        self.assertIn("<think>", count)
        regional = question(arch, "presence", "atomic_1", "impacted_tooth", arch["regions"][0][1])
        self.assertIn("in the maxilla (upper jaw) of this image", regional)
        self.assertNotIn("boundary", regional)
        regional_count = question(arch, "count", "atomic_1", "dental_filling", arch["regions"][0][1])
        self.assertIn("boundary", regional_count)
        for job in (whole, arch):
            for condition in CONDITIONS:
                for stage in ("presence", "count", "combined"):
                    for template in ("atomic_1", "atomic_2", "atomic_3"):
                        rendered = question(job, stage, template, condition, job["regions"][0][1])
                        self.assertNotIn("{", rendered)
                        self.assertIn("<answer>", rendered)

    def test_reasoning_does_not_pollute_count_or_presence_parsing(self):
        self.assertEqual(parse_answer("<think>Initially 8, then 9.</think><answer>10</answer>", "count", CONDITIONS), 10)
        self.assertEqual(parse_answer("<think>Consider A and B.</think><answer>B</answer>", "presence", CONDITIONS), "B")

    def test_rendered_prompt_changes_invalidate_fingerprint(self):
        plan = self.plan()
        with patch("research_experiments.ATOMIC_FINDING_LABELS", {c: "Changed wording" for c in CONDITIONS}):
            changed = self.plan()
        self.assertNotEqual(plan["fingerprint"], changed["fingerprint"])

    def test_notebook_api_setup_and_cell_compilation(self):
        from IPython.core.inputtransformer2 import TransformerManager
        import shutil
        import subprocess
        from model_routing import ModelRouting
        notebook = json.loads(Path(__file__).with_name("main_notebook.ipynb").read_text(encoding="utf8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile(TransformerManager().transform_cell("".join(cell["source"])), cell["id"], "exec")
        env = {"Path": Path, "local_preset": local_preset, "shutil": shutil,
               "subprocess": subprocess, "ModelRouting": ModelRouting,
               "DentalAnalysisPipeline": lambda **kwargs: self.fail("ordinary pipeline constructed")}
        exec("".join(notebook["cells"][3]["source"]), env)
        # The old ordinary analyzer remains local; the selected suite is API-only.
        env.update(RESEARCH_LOCAL_PRESET=None, NEEDS_LOCAL_RUNTIME=False)
        for index in [4, 5, 6, 7, 8, 9, 13]:
            exec("".join(notebook["cells"][index]["source"]), env)
        self.assertIsNone(env["server"])
        self.assertIsNone(env["expert_model_runner"])
        self.assertIsNone(env["pipeline"])

    def test_failed_provider_does_not_block_other_model(self):
        self.models["b"] = {**self.models["a"], "model": "working"}
        self.experiments = [{"id": k, "model": k, "strategies": ["broad_whole"]} for k in self.models]
        def factory(job):
            if job["model_key"] == "a":
                raise ValueError("missing credentials")
            return FakeRunner()
        directory = run_suite(self.plan(), self.providers, self.root / "runs", runner_factory=factory)
        results = load_suite(directory)
        self.assertEqual([r["status"] for r in results], ["failed", "completed"])
        self.assertTrue(all(r["unequal_image_coverage"] for r in evaluate_research_suite(results)["summary"]))

    def test_positive_count_failure_falls_back_and_prices(self):
        self.experiments[0].update(strategies=["atomic_whole"], atomic_protocol="presence_then_count")
        runner = FakeRunner(lambda prompt, *args: answer("A" if "A. True" in prompt else 0))
        result = load_suite(self.run_fake(self.plan(), runner))[0]
        self.assertTrue(all(c["forced_zero"] for c in result["checks"]))
        self.experiments[0].update(strategies=["broad_whole"])
        self.models["a"]["prices_per_million"] = {"input": 2, "output": 4}
        result = load_suite(self.run_fake(self.plan(), FakeRunner()))
        self.assertEqual(evaluate_research_suite(result)["summary"][0]["estimated_cost"], .00004)


if __name__ == "__main__":
    unittest.main()
