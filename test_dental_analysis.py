"""Offline diagnostics and fair saved-run comparisons, with no model calls."""
import copy
import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import dental_analysis as da
import dental_eval as ev
import dental_pipeline as dp
import experiments as xp


CONDITION = "dental_filling"
PROTOCOL = {"presence_level": "region", "count_level": "region", "region_scheme": "quadrant",
            "region_prompt": "words", "question_form": "combined", "parse_retries": 1}


def fixture():
    gt, results = {}, {}
    # Recovery, new false alarm, removed false alarm, lost TP, newly resolved, newly unresolved.
    cases = [(True, "B", "A"), (False, "B", "A"), (False, "A", "B"),
             (True, "A", "B"), (True, None, "A"), (True, "A", None)]
    for index, (truth, whole, final) in enumerate(cases):
        image_id = str(index)
        boxes = ([{"condition": CONDITION, "xc": .2, "yc": .2, "w": .1, "h": .1}] if truth else [])
        gt[image_id] = {"annotated": {CONDITION}, "boxes": boxes}
        findings = {c: {"presence": "B", "whole_image": "B", "count": None,
                       "regions": {r: "B" for r in dp.CROPS["quadrant"]}, "region_counts": {}}
                    for c in dp.CONDITIONS}
        findings[CONDITION].update(presence=final, whole_image=whole, count=1 if final == "A" else None,
                                   regions={"UR": final, "UL": "B", "LL": "B", "LR": "B"},
                                   region_counts={"UR": 1} if final == "A" else {})
        results[image_id] = {"image_id": image_id, "image_sha256": str(index) * 64,
                             "protocol": dict(PROTOCOL), "mode": "plain", "findings": findings,
                             "calls": [], "call_count": 0}
    return gt, results


def call(value, attempt=1, stage="presence", region=None):
    return {"stage": stage, "condition": CONDITION, "region": region,
            "parse_recovery": {"attempt": attempt, "value": value},
            "prompt_tokens": 10, "completion_tokens": 2, "latency_seconds": .5}


def save_run(root, results):
    (root / "results").mkdir(parents=True)
    example = next(iter(results.values()))
    manifest = {"hash": "test", "protocol": example["protocol"], "mode": example["mode"],
                "runner": {"model": "fake"}}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for image_id, result in results.items():
        (root / "results" / f"{image_id}.json").write_text(json.dumps(result), encoding="utf-8")


class AnalysisTests(unittest.TestCase):
    def test_transitions_and_existing_scores_are_preserved(self):
        gt, results = fixture()
        untouched = copy.deepcopy((gt, results))
        report = ev.evaluate(gt, results)
        original = ev.evaluate(gt, results, include_analysis=False)
        for key, value in original.items():
            self.assertEqual(report[key], value)
        changes = {r["transition"]: r for r in report["stage_changes"] if r["condition"] == "ALL"}
        self.assertEqual(set(changes), {"FN -> TP", "TN -> FP", "FP -> TN", "TP -> FN",
                                       "unresolved -> TP", "TP -> unresolved"})
        self.assertTrue(all(r["checks"] == 1 for r in changes.values()))
        self.assertEqual(changes["FN -> TP"]["image_ids"], ["0"])
        self.assertEqual((gt, results), untouched)

    def test_recovered_does_not_mean_correct_and_partial_fields_survive(self):
        gt, results = fixture()
        results["0"]["calls"] = [call(None), call("B", 2)]  # wrong but parseable recovery
        results["1"]["calls"] = [call(["B", None]), call(["B", None], 2)]
        rows = da.recovery_rows(gt, results, True)
        rows = {(r["field"], r["status"]): r for r in rows}
        self.assertEqual(rows["presence", "recovered"]["correct"], 0)
        self.assertEqual(rows["presence", "recovered"]["correctness_scored"], 1)
        self.assertEqual(rows["presence", "first_pass"]["correct"], 1)
        self.assertEqual(rows["count", "unresolved"]["correctness_scored"], 0)
        usage = da.call_usage(results)
        self.assertEqual(sum(r["calls"] for r in usage), 4)  # combined fields do not double-count calls
        self.assertEqual(sum(r["prompt_tokens"] for r in usage), 40)

    def test_disabled_location_skips_regional_truth_even_in_recovery(self):
        gt, results = fixture()
        results["0"]["calls"] = [call(["A", 1], stage="region", region="UR")]
        with patch.object(ev, "gt_regions", side_effect=AssertionError("location evaluated")), \
             patch.object(ev, "box_primary_region", side_effect=AssertionError("location evaluated")):
            report = ev.evaluate(gt, results, evaluate_location=False)
        self.assertTrue(all(r["correctness_scored"] == 0 for r in report["parse_recovery"]))
        self.assertFalse(any(r["situation"] == "true_regions" for r in report["case_breakdown"]))

    def test_slices_have_real_denominators_and_honor_annotation_scope(self):
        gt, results = fixture()
        gt["0"]["boxes"].append(dict(gt["0"]["boxes"][0]))
        rows = ev.evaluate(gt, results, evaluate_location=False)["case_breakdown"]
        rows = {(r["situation"], r["group"]): r for r in rows}
        multiple = rows["instances_of_finding", "2+"]
        self.assertEqual(multiple["expected_finding_checks"], 1)
        self.assertEqual(multiple["counts_mae"], 1)
        self.assertIsNone(multiple["specificity"])
        absent = rows["absent_finding_context", "no_annotated_findings"]
        self.assertEqual((absent["FP"], absent["TN"]), (1, 1))
        self.assertIsNone(absent["sensitivity"])
        self.assertEqual(absent["expected_finding_checks"], 2)  # not all 14 unannotated conditions

    def test_empty_legacy_and_missing_usage_are_not_fabricated(self):
        self.assertEqual(ev.evaluate({}, {})["case_breakdown"], [])
        gt, results = fixture()
        for result in results.values():
            for finding in result["findings"].values():
                finding.pop("whole_image")
        results["0"]["calls"] = [{"stage": "presence"}]
        report = ev.evaluate(gt, results)
        self.assertEqual(report["stage_changes"], [])
        self.assertEqual(report["parse_recovery"], [])
        self.assertIsNone(report["call_usage"][0]["prompt_tokens"])
        self.assertEqual(report["call_usage"][0]["attempt"], "unknown")

    def test_location_sources_and_boundary_slices(self):
        gt, results = fixture()
        gt["0"]["boxes"][0].update(regions=["UR", "UL"], region_source="llm")
        gt["3"]["boxes"][0].update(regions=[], region_source="excluded", location_excluded=True)
        rows = ev.evaluate(gt, results)["case_breakdown"]
        groups = {(r["situation"], r["group"]): r for r in rows}
        self.assertEqual(groups["location_truth_source", "llm"]["image_ids"], ["0"])
        self.assertEqual(groups["true_regions", "2+"]["image_ids"], ["0"])
        self.assertEqual(groups["crosses_region_boundary", "yes"]["image_ids"], ["0"])
        self.assertEqual(groups["location_truth_source", "excluded"]["regions_scored"], 0)

    def test_exports_and_stale_table_cleanup(self):
        gt, results = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ev.evaluate(gt, results, out_dir=root)
            with (root / "stage_changes.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertIsInstance(json.loads(rows[0]["image_ids"]), list)
            self.assertTrue((root / "case_breakdown.csv").is_file())
            ev.evaluate({}, {}, out_dir=root)
            self.assertFalse((root / "case_breakdown.csv").exists())
            self.assertFalse((root / "stage_changes.csv").exists())

    def test_comparison_pairs_coverage_and_ignores_unselected_images(self):
        gt, current = fixture()
        baseline = copy.deepcopy(current)
        for result in baseline.values():
            result["findings"][CONDITION]["presence"] = result["findings"][CONDITION]["whole_image"]
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            save_run(a, baseline)
            save_run(b, current)
            report = da.compare_runs(gt, {"baseline": a, "variant": b}, evaluate_location=False)
            row = report["run_comparison"][1]
            self.assertEqual((row["corrected"], row["worsened"]), (2, 2))
            self.assertEqual((row["newly_resolved"], row["newly_unresolved"]), (1, 1))
            self.assertEqual(row["paired_checks"], 4)
            self.assertEqual(row["coverage"], .8333)
            self.assertEqual(row["paired_f1_delta"], 0)
            self.assertIsNone(row["prompt_tokens"])
            selected = da.compare_runs({"0": gt["0"]}, {"baseline": a, "variant": b}, evaluate_location=False)
            self.assertEqual(selected["run_comparison"][1]["corrected"], 1)
            ev.write_report(report, Path(tmp) / "export")
            self.assertTrue((Path(tmp) / "export" / "run_comparison.csv").exists())

    def test_comparison_rejects_missing_mismatched_and_unverifiable_results(self):
        gt, results = fixture()
        for problem in ("missing", "hash", "no_hash", "protocol"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as tmp:
                a, b = Path(tmp) / "a", Path(tmp) / "b"
                modified = copy.deepcopy(results)
                if problem == "missing":
                    modified.pop("0")
                elif problem == "hash":
                    modified["0"]["image_sha256"] = "f" * 64
                elif problem == "no_hash":
                    modified["0"].pop("image_sha256")
                else:
                    modified["1"]["protocol"]["region_prompt"] = "crop"
                save_run(a, results)
                save_run(b, modified)
                with self.assertRaises(ValueError):
                    da.compare_runs(gt, {"a": a, "b": b})

    def test_notebook_cell_ranks_experiments_offline(self):
        gt, results = fixture()
        nb = json.loads(Path(__file__).with_name("main_notebook.ipynb").read_text(encoding="utf-8"))
        source = next("".join(c["source"]) for c in nb["cells"] if "# CELL 10 -" in "".join(c["source"]))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_run(root / "a" / "toy", results)
            save_run(root / "b" / "toy", results)
            configs = xp.build([{"name": "a", "evaluate_location": False},
                                {"name": "b", "evaluate_location": False}], {"output_root": tmp})
            scope = {"EXPERIMENTS": configs, "DATASETS": [{"name": "toy"}], "GT": {"toy": gt}, "ADAPTED": {},
                     "OUTPUT_ROOT": tmp, "xp": xp, "dp": dp, "ev": ev, "da": da, "Path": Path}
            with redirect_stdout(io.StringIO()), patch("IPython.display.display"):
                exec(compile(source, "<evaluation cell>", "exec"), scope)
            self.assertEqual(sorted(scope["REPORTS"]), [("a", "toy"), ("b", "toy")])
            self.assertEqual(len(scope["LEADERBOARD"]), 2)
            self.assertEqual(len(scope["COMPARISONS"]["toy"]["run_comparison"]), 2)
            self.assertTrue((root / "leaderboard.csv").is_file())
            self.assertTrue((root / "comparison" / "toy" / "run_changes.csv").is_file())
            self.assertTrue((root / "a" / "toy" / "evaluation" / "presence.csv").is_file())


if __name__ == "__main__":
    unittest.main()
