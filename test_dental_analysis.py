"""DentVLM-specific reporting checks; no model service or dataset required."""
import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import dental_analysis as da
import dental_eval as ev
import dental_pipeline as dp


def fixture():
    protocol = dp.Protocol(phrasings=3, extra_tasks=False)
    tasks = {}
    for task in protocol.tasks():
        answers = [{"answer": "no", "regions": [], "truncated": False} for _ in range(3)]
        if task == "fillings":
            answers[1].update(answer="yes", regions=["upper-left"])
            answers[2].update(answer="yes", regions=["upper-left", "lower-right"])
        if task == "prosthetic_bridge":
            for a in answers:
                a.update(answer="yes", regions=["upper-right"])
        decision = dp.vote(answers, protocol.region_vote)
        tasks[task] = {"answers": answers, **decision, "whole_image": decision["presence"]}
    result = {"image_id": "img", "image_sha256": "a" * 64, "protocol": asdict(protocol),
              "location_level": "rationale", "left_is_image_left": True, "tasks": tasks,
              "findings": {c: dp._finding(c, tasks, protocol) for c in dp.CONDITIONS},
              "calls": [], "call_count": 30}
    gt = {"img": {"annotated": set(dp.CONDITIONS), "boxes": [
        {"condition": "dental_filling", "xc": .2, "yc": .2, "w": .05, "h": .05,
         "regions": ["upper-left"], "region_source": "llm"},
        {"condition": "prosthetic_restoration", "xc": .8, "yc": .2, "w": .05, "h": .05,
         "regions": ["upper-right"], "region_source": "llm"}]}}
    return gt, {"img": result}


def call(task, value, attempt=1, stage="presence", cell=None):
    return {"stage": stage, "task": task, "cell": cell,
            "parse_recovery": {"attempt": attempt, "value": value},
            "prompt_tokens": 10, "completion_tokens": 2, "latency_seconds": .5}


def save_run(root, results):
    (root / "results").mkdir(parents=True)
    protocol = next(iter(results.values()))["protocol"]
    (root / "manifest.json").write_text(json.dumps({"hash": "test", "protocol": protocol,
                                                    "runner": {"model": "fake"}}), encoding="utf-8")
    for image_id, result in results.items():
        (root / "results" / (image_id + ".json")).write_text(json.dumps(result), encoding="utf-8")


class AnalysisTests(unittest.TestCase):
    def test_preserves_original_scores_and_input(self):
        gt, results = fixture()
        before = copy.deepcopy((gt, results))
        report = ev.evaluate(gt, results)
        for key, value in ev.evaluate(gt, results, include_analysis=False).items():
            self.assertEqual(report[key], value)
        self.assertEqual((gt, results), before)
        self.assertEqual(report["summary"]["expected_finding_checks"], 9)
        self.assertEqual(len(report["summary"]["not_assessed"]), 5)

    def test_phrasing_recovery_and_union_majority_use_saved_answers(self):
        gt, results = fixture()
        report = ev.evaluate(gt, results)
        transitions = [r for r in report["stage_changes"] if r["condition"] == "dental_filling"]
        self.assertEqual(transitions[0]["transition"], "FN -> TP")
        self.assertEqual(transitions[0]["comparison"], "first_phrasing_to_vote")
        votes = {r["task"]: r for r in report["phrasing_votes"]}
        self.assertEqual(votes["fillings"]["status"], "disagreement")
        modes = {r["region_vote"]: r for r in report["region_vote_comparison"]}
        self.assertEqual(modes["union"]["regions_exact_set_match_rate"], .5)
        self.assertEqual(modes["majority"]["regions_exact_set_match_rate"], 1)
        self.assertEqual(modes["union"]["TP"], modes["majority"]["TP"])
        self.assertEqual(modes["union"]["counts_scored"], 0)
        self.assertIsNone(modes["union"]["counts_mae"])

    def test_crown_bridge_or_and_ties_are_not_invented_errors(self):
        gt, results = fixture()
        report = ev.evaluate(gt, results)
        restoration = [r for r in report["stage_changes"] if r["condition"] == "prosthetic_restoration"]
        self.assertEqual(restoration[0]["transition"], "TP -> TP")
        task = results["img"]["tasks"]["fillings"]
        task["answers"][2]["answer"] = None
        task["whole_image"] = task["presence"] = None
        results["img"]["findings"]["dental_filling"].update(presence=None, whole_image=None, regions=None)
        report = ev.evaluate(gt, results)
        self.assertTrue(any(r["task"] == "fillings" and r["status"] == "tie" for r in report["phrasing_votes"]))
        self.assertTrue(any(r["condition"] == "dental_filling" and r["transition"] == "FN -> unresolved"
                            for r in report["stage_changes"]))

    def test_crops_have_their_own_transitions_and_no_rationale_replay(self):
        gt, results = fixture()
        results["img"]["location_level"] = results["img"]["protocol"]["location"] = "crops"
        results["img"]["findings"]["dental_filling"].update(whole_image="no", presence="yes")
        report = ev.evaluate(gt, results)
        self.assertEqual(report["region_vote_comparison"], [])
        self.assertTrue(any(r["comparison"] == "whole_image_to_crops" and r["transition"] == "FN -> TP"
                            for r in report["stage_changes"]))

    def test_recovery_separates_phrasings_and_respects_composite_truth(self):
        gt, results = fixture()
        results["img"]["calls"] = [call("fillings", None), call("fillings", "no", 2),
            call("fillings", "yes"), call("prosthetic_crown", None), call("prosthetic_crown", "yes", 2),
            call("calculus", "yes"), call("dental_filling", 1, stage="count")]
        rows = {(r["task"], r["status"]): r for r in da.recovery_rows(gt, results, True)}
        self.assertEqual(rows["fillings", "recovered"]["correct"], 0)
        self.assertEqual(rows["fillings", "first_pass"]["correct"], 1)
        self.assertEqual(rows["prosthetic_crown", "recovered"]["correctness_scored"], 0)
        self.assertEqual(rows["calculus", "first_pass"]["correctness_scored"], 0)
        self.assertEqual(rows["dental_filling", "first_pass"]["correct"], 1)
        self.assertEqual(sum(r["calls"] for r in da.call_usage(results)), 7)

    def test_location_toggle_and_legacy_metadata(self):
        gt, results = fixture()
        results["img"]["calls"] = [call("fillings", "yes", stage="crop", cell="upper-left")]
        with patch.object(ev, "gt_regions", side_effect=AssertionError("location evaluated")):
            report = ev.evaluate(gt, results, evaluate_location=False)
        self.assertEqual(report["region_vote_comparison"], [])
        self.assertEqual(report["parse_recovery"][0]["correctness_scored"], 0)
        results["img"].pop("tasks")
        results["img"]["calls"] = [{"stage": "presence"}]
        report = ev.evaluate(gt, results)
        self.assertEqual(report["phrasing_votes"], [])
        self.assertEqual(report["parse_recovery"], [])
        self.assertIsNone(report["call_usage"][0]["prompt_tokens"])

    def test_task_support_and_case_denominators(self):
        gt, results = fixture()
        rows = {(r["situation"], r["group"]): r for r in ev.evaluate(gt, results)["case_breakdown"]}
        self.assertEqual(rows["task_support", "untrained"]["not_assessed_checks"], 5)
        self.assertIsNone(rows["task_support", "untrained"]["coverage"])
        self.assertEqual(rows["instances_of_finding", "1"]["TP"], 2)
        self.assertIsNone(rows["instances_of_finding", "1"]["specificity"])
        self.assertEqual(rows["predicted_named_cells", "2+"]["counts_scored"], 0)

    def test_comparison_distinguishes_newly_asked_from_newly_resolved(self):
        gt, base = fixture()
        variant = copy.deepcopy(base)
        variant["img"]["protocol"]["ask_untrained"] = True
        variant["img"]["findings"]["furcation_lesion"].update(asked=True, presence="no")
        variant["img"]["findings"]["dental_filling"]["presence"] = None
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            save_run(a, base)
            save_run(b, variant)
            report = da.compare_runs(gt, {"base": a, "variant": b}, evaluate_location=False)
            row = report["run_comparison"][1]
            self.assertEqual(row["left_not_assessed"], 1)
            self.assertEqual(row["became_unresolved"], 1)
            self.assertEqual(row["paired_checks"], 8)
            self.assertEqual(row["corrected"], 0)
            self.assertEqual(row["worsened"], 0)
            self.assertEqual(row["coverage"], .9)
            self.assertEqual(row["not_assessed_checks"], 4)

    def test_comparison_rejects_mismatched_inputs(self):
        gt, base = fixture()
        for problem in ("hash", "missing_hash", "protocol", "orientation", "missing"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as tmp:
                a, b = Path(tmp) / "a", Path(tmp) / "b"
                save_run(a, base)
                save_run(b, base)
                p = b / "results" / "img.json"
                modified = copy.deepcopy(base["img"])
                if problem == "hash":
                    modified["image_sha256"] = "b" * 64
                elif problem == "missing_hash":
                    modified.pop("image_sha256")
                elif problem == "orientation":
                    modified["left_is_image_left"] = False
                else:
                    modified["protocol"]["phrasings"] = 1
                p.write_text(json.dumps(modified), encoding="utf-8")
                if problem == "missing":
                    p.unlink()
                with self.assertRaises(ValueError):
                    da.compare_runs(gt, {"a": a, "b": b})

    def test_exports_empty_cleanup_and_notebook_execution(self):
        gt, results = fixture()
        nb = json.loads(Path(__file__).with_name("main_notebook.ipynb").read_text(encoding="utf-8"))
        source = next("".join(c["source"]) for c in nb["cells"] if "# CELL 13 -" in "".join(c["source"]))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_run(root / "a" / "toy", results)
            save_run(root / "b" / "toy", results)
            start = source.index("COMPARE_RUN_DIRS = {")
            end = source.index("\n}", start) + 2
            source = source[:start] + "COMPARE_RUN_DIRS = " + repr({"a": str(root / "a"), "b": str(root / "b")}) + source[end:]
            scope = {"DATASETS": [{"name": "toy"}], "GT": {"toy": gt}, "OUTPUT_DIR": root / "a",
                     "ADAPTED": {}, "EVALUATE_LOCATION": True, "dp": dp, "ev": ev, "Path": Path}
            with redirect_stdout(io.StringIO()), patch("IPython.display.display"):
                exec(compile(source, "<cell13>", "exec"), scope)
            out = root / "a" / "toy" / "evaluation"
            self.assertTrue((out / "run_comparison.csv").exists())
            self.assertTrue((out / "region_vote_comparison.csv").exists())
            ev.evaluate({}, {}, out_dir=out)
            self.assertFalse((out / "region_vote_comparison.csv").exists())
            self.assertFalse((out / "run_comparison.csv").exists())


if __name__ == "__main__":
    unittest.main()
