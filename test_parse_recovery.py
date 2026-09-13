"""Offline recovery checks through both local and API VisionRunner paths."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import dental_eval as ev
import dental_pipeline as dp
import llm_api

DENTVLM = hasattr(dp, "extract_answer")


class ScriptClient:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        self.requests.append(request)
        question = request["messages"][0]["content"][1]["text"]
        default = ("1" if question.startswith("How many") else "No" if DENTVLM else
                   "Answer: B. False\nCount: 0" if "Finding under review:" in question else "B")
        response = next(self.replies, default)
        if isinstance(response, Exception):
            raise response
        if hasattr(response, "choices"):
            return response
        text, finish = response if isinstance(response, tuple) else (response, "stop")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish)],
            usage=None)


def protocol(retries=1, **kwargs):
    base = {"location": "none"} if DENTVLM else {"presence_level": "overall", "count_level": "overall"}
    return dp.Protocol(parse_retries=retries, **dict(base, **kwargs))


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.image = self.root / "sample.png"
        from PIL import Image
        Image.new("RGB", (32, 32), "white").save(self.image)

    def tearDown(self):
        self.temp.cleanup()

    def run_image(self, replies, *, local=False, retries=1, api_retries=2, **kwargs):
        client = ScriptClient(replies)
        runner = dp.VisionRunner(client=client, local=local, max_tokens=4096, api_call_retries=api_retries)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = dp.analyze_image(runner, self.image, protocol=protocol(retries, **kwargs))
        return result, client, output.getvalue(), runner

    def truth(self, result):
        return {"sample": {"path": str(self.image), "annotated": set(dp.CONDITIONS),
                           "boxes": [{"condition": "dental_implant", "xc": .2, "yc": .2, "w": .1, "h": .1}]}}

    def test_full_failure_printed_and_recovered_for_local_and_api(self):
        raw = "?" * 5001 + "\nEND OF FULL RESPONSE"
        for local in (True, False):
            with self.subTest(local=local):
                replies = [raw, "Yes" if DENTVLM else "A"]
                result, client, log, _ = self.run_image(replies, local=local)
                expected = "yes" if DENTVLM else "A"
                self.assertEqual(result["findings"]["dental_implant"]["presence"], expected)
                self.assertIn(raw, log)
                question = client.requests[0]["messages"][0]["content"][1]["text"]
                self.assertIn(question, log)
                self.assertIn("PARSE WARNING", log)
                self.assertIn("PARSE RECOVERED", log)
                self.assertEqual(result["parse_recovery"]["retry_calls"], 1)
                self.assertEqual(result["parse_recovery"]["recovered_checks"], 1)
                self.assertEqual(result["calls"][0]["text"], raw)
                self.assertIn("parse_recovery", result["calls"][1])
                first, second = client.requests[:2]
                self.assertEqual(first["messages"][0]["content"][0], second["messages"][0]["content"][0])
                self.assertEqual(first["model"], second["model"])
                self.assertEqual(first["temperature"], second["temperature"])
                self.assertNotEqual(first["messages"], second["messages"])

    def test_exhausted_is_neutral_and_keeps_every_attempt(self):
        result, _, log, _ = self.run_image(["???", "???", "???"], retries=2)
        self.assertIsNone(result["findings"]["dental_implant"]["presence"])
        self.assertEqual(result["parse_recovery"]["retry_calls"], 2)
        self.assertEqual(result["parse_recovery"]["unresolved_checks"], 1)
        self.assertEqual(log.count("PROMPT (full):"), 3)
        self.assertIn("PARSE EXHAUSTED", log)
        report = ev.evaluate(self.truth(result), {"sample": result}, evaluate_location=False)
        summary = report["summary"]
        row = next(r for r in report["presence"] if r["condition"] == "dental_implant")
        self.assertEqual([row[k] for k in ("TP", "FP", "TN", "FN")], [0, 0, 0, 0])
        self.assertEqual(row["unparseable"], 1)
        self.assertEqual(summary["expected_finding_checks"], 9 if DENTVLM else 14)
        self.assertEqual(summary["scored_finding_checks"], 8 if DENTVLM else 13)
        self.assertEqual(summary["excluded_unparseable_checks"], 1)
        self.assertTrue(summary["finding_check_invariant_ok"])
        self.assertIsNone(summary["mean_recall_per_image"])
        self.assertIsNone(summary["complete_case_rate"])
        self.assertEqual(report["per_image"][0]["gt_present"], 1)
        self.assertEqual(report["per_image"][0]["gt_present_scored"], 0)
        if DENTVLM:
            self.assertEqual(len(summary["not_assessed"]), 5)

    def test_zero_retries_still_prints_failure(self):
        result, _, log, _ = self.run_image(["???"], retries=0)
        self.assertEqual(result["parse_recovery"]["retry_calls"], 0)
        self.assertIsNone(result["findings"]["dental_implant"]["presence"])
        self.assertIn("PROMPT (full):", log)
        self.assertIn("RESPONSE (full):\n???", log)

    def test_empty_and_truncated_are_retried(self):
        for response, reason in [("", "empty_response"), (("Yes" if DENTVLM else "A", "length"), "truncated_output")]:
            result, _, log, _ = self.run_image([response, "No" if DENTVLM else "B"])
            self.assertEqual(result["findings"]["dental_implant"]["presence"], "no" if DENTVLM else "B")
            self.assertIn(reason, log)
            self.assertEqual(result["parse_recovery"]["recovered_checks"], 1)

    def test_transport_errors_use_visible_api_retries_not_parse_retries(self):
        valid = "Yes" if DENTVLM else "A"
        result, client, log, _ = self.run_image([RuntimeError("transport"), valid], retries=3, api_retries=1)
        self.assertEqual(result["parse_recovery"]["retry_calls"], 0)
        self.assertIn("API RETRY", log)
        self.assertGreaterEqual(len(client.requests), 2)

    def test_invalid_response_envelope_is_retried(self):
        valid = "Yes" if DENTVLM else "A"
        result, _, log, _ = self.run_image([SimpleNamespace(choices=[]), valid], api_retries=1)
        self.assertEqual(result["parse_recovery"]["retry_calls"], 0)
        self.assertIn("API RETRY", log)

    def test_corrupt_resume_artifact_stops(self):
        _, _, _, runner = self.run_image([])
        run_dir = self.root / "corrupt-run"
        with contextlib.redirect_stdout(io.StringIO()):
            dp.run_dataset(runner, {"sample": self.image}, run_dir, protocol=protocol(0))
        saved = run_dir / "results" / "sample.json"
        saved.write_text('{"image_id":"wrong"}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "does not match"):
            dp.run_dataset(runner, {"sample": self.image}, run_dir, protocol=protocol(0))

    def test_retry_setting_changes_manifest_and_resume_is_rejected(self):
        result, client, _, runner = self.run_image([])
        run_dir = self.root / "run"
        with contextlib.redirect_stdout(io.StringIO()):
            dp.run_dataset(runner, {"sample": self.image}, run_dir, protocol=protocol(0))
            previous_calls = len(client.requests)
            dp.run_dataset(runner, {"sample": self.image}, run_dir, protocol=protocol(0))
            self.assertEqual(len(client.requests), previous_calls)
            with self.assertRaisesRegex(ValueError, "different configuration"):
                dp.run_dataset(runner, {"sample": self.image}, run_dir, protocol=protocol(1))
        saved = json.loads((run_dir / "results" / "sample.json").read_text())
        self.assertIn("parse_recovery", saved)
        self.assertIn("parse_recovery", saved["calls"][0])

    def test_invalid_retry_counts_rejected(self):
        for retries in (-1, 1.5, True, "2"):
            with self.subTest(retries=retries), self.assertRaises(ValueError):
                protocol(retries)

    def test_success_never_retried_or_warned(self):
        result, _, log, _ = self.run_image([])
        self.assertEqual(result["parse_recovery"]["retry_calls"], 0)
        self.assertNotIn("PARSE WARNING", log)



    def test_all_unparseable_has_no_accuracy_credit(self):
        result, _, _, _ = self.run_image(["???"] * 100, retries=1)
        summary = ev.evaluate(self.truth(result), {"sample": result}, evaluate_location=False)["summary"]
        self.assertEqual(summary["scored_finding_checks"], 0)
        self.assertEqual(summary["expected_finding_checks"], summary["excluded_unparseable_checks"])
        for metric in ("sensitivity", "specificity", "ppv", "f1", "complete_case_rate",
                       "mean_recall_per_image", "mean_false_alarms_per_image"):
            self.assertIsNone(summary[metric], metric)

    def test_notebook_wires_retry_configuration(self):
        notebook = json.loads(Path(__file__).with_name("main_notebook.ipynb").read_text(encoding="utf-8"))
        code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
        self.assertIn("PARSE_RETRIES = 1", code)
        self.assertIn("parse_retries=PARSE_RETRIES", code)

    @unittest.skipIf(DENTVLM, "DentalGPT combined protocol")
    def test_combined_missing_presence_then_count_recovery_shares_budget(self):
        result, _, log, _ = self.run_image(
            ["???", "Answer: A. True\nCount: 0", "2"],
            retries=2, question_form="combined")
        self.assertEqual(result["findings"]["dental_implant"]["presence"], "A")
        self.assertEqual(result["findings"]["dental_implant"]["count"], 2)
        self.assertEqual(result["parse_recovery"]["retry_calls"], 2)
        self.assertTrue(result["calls"][2]["question"].startswith("How many"))
        self.assertIn("invalid_count_pair", log)
        self.assertEqual(result["calls"][2]["parse_recovery"]["attempt"], 3)

    @unittest.skipIf(DENTVLM, "DentalGPT combined protocol")
    def test_combined_preserves_presence_when_count_recovery_exhausts(self):
        result, _, _, _ = self.run_image(
            ["Answer: A. True\nCount: 0", "???"], question_form="combined")
        self.assertEqual(result["findings"]["dental_implant"]["presence"], "A")
        self.assertIsNone(result["findings"]["dental_implant"]["count"])
        self.assertEqual(result["parse_recovery"]["unresolved_checks"], 1)
        report = ev.evaluate(self.truth(result), {"sample": result}, evaluate_location=False)
        self.assertEqual(report["summary"]["TP"], 1)
        count = next(r for r in report["counts"] if r["condition"] == "dental_implant")
        self.assertEqual(count["count_unparseable"], 1)
        self.assertIsNone(count["mae"])

    @unittest.skipIf(DENTVLM, "DentalGPT regional protocol")
    def test_regional_presence_and_count_retry_preserve_scope(self):
        for region_prompt in ("words", "crop"):
            with self.subTest(region_prompt=region_prompt):
                # All whole-image answers negative; retry presence then count in the first region.
                result, client, _, _ = self.run_image(
                    ["B"] * 14 + ["???", "A", "???", "2"],
                    presence_level="region", count_level="region",
                    region_prompt=region_prompt, local=True)
                finding = result["findings"]["dental_implant"]
                self.assertEqual(finding["regions"]["UR"], "A")
                self.assertEqual(finding["region_counts"]["UR"], 2)
                self.assertEqual(result["parse_recovery"]["retry_calls"], 2)
                a, b = client.requests[14:16]
                self.assertEqual(a["messages"][0]["content"][0], b["messages"][0]["content"][0])
                if region_prompt == "words":
                    self.assertIn("upper right", b["messages"][0]["content"][1]["text"])

    @unittest.skipIf(DENTVLM, "DentalGPT combined regional protocol")
    def test_combined_count_retry_keeps_patient_scope(self):
        result, _, _, _ = self.run_image(
            ["B"] * 14 + ["Answer: A. True\nCount: 0", "2"],
            presence_level="region", count_level="region", question_form="combined")
        self.assertEqual(result["findings"]["dental_implant"]["region_counts"]["UR"], 2)
        retry = result["calls"][15]
        self.assertIn("upper right", retry["question"])
        self.assertEqual(retry["region"], "UR")
        self.assertEqual(retry["parse_recovery"]["value"], ("A", 2))

    @unittest.skipUnless(DENTVLM, "DentVLM crop protocol")
    def test_crop_exhaustion_is_neutral_in_report_as_well(self):
        import report_writer
        n_tasks = len(protocol().tasks())
        result, _, _, _ = self.run_image(
            ["No"] * n_tasks + [("Yes\nunfinished", "length")] * 2, location="crops")
        self.assertIsNone(result["findings"]["dental_implant"]["presence"])
        cells = report_writer._cell_answers(result)
        self.assertIsNone(cells["implant"][dp.CELLS[0]])
        self.assertEqual(result["parse_recovery"]["unresolved_checks"], 1)

    @unittest.skipUnless(DENTVLM, "DentVLM optional count protocol")
    def test_optional_count_recovery(self):
        n_tasks = len(protocol().tasks())
        result, _, _, _ = self.run_image(
            ["Yes"] + ["No"] * (n_tasks - 1) + ["???", "2"], count_question=True)
        self.assertEqual(result["findings"]["dental_implant"]["count"], 2)
        self.assertEqual(result["parse_recovery"]["retry_calls"], 1)
        self.assertEqual(result["calls"][-1]["stage"], "count")


if __name__ == "__main__":
    unittest.main()
