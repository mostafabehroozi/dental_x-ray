"""Offline checks for the independent location-scoring switch."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import dental_eval as ev
import experiments as xp
import dental_pipeline as dp


class LocationScoringTests(unittest.TestCase):
    def test_toggle_preserves_finding_and_total_count_scores(self):
        findings = {c: {"asked": True, "presence": "B", "whole_image": "B",
                        "count": 0, "regions": {"UR": "B", "UL": "B", "LL": "B", "LR": "B"},
                        "region_counts": {}} for c in dp.CONDITIONS}
        findings["dental_filling"].update(
            presence="A", whole_image="A", count=1,
            regions={"UR": "A", "UL": "B", "LL": "B", "LR": "B"},
            region_counts={"UR": 1})
        results = {"image": {"findings": findings, "call_count": 71,
                            "location_level": "quadrant",
                            "protocol": {"presence_level": "region", "count_level": "region", "region_scheme": "quadrant", "region_prompt": "words"}}}
        gt = {"image": {"annotated": set(dp.CONDITIONS), "boxes": [
            {"condition": "dental_filling", "xc": 0.2, "yc": 0.25, "w": 0.05, "h": 0.05}]}}
        original = copy.deepcopy(results)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            enabled = ev.evaluate(gt, results, out_dir=out)
            self.assertTrue(enabled["regions"])
            self.assertTrue(enabled["region_presence"] and enabled["summary"]["region_presence"]["TP"] == 1)
            self.assertTrue((out / "regions.csv").exists() and (out / "region_presence.csv").exists())
            with patch.object(ev, "gt_regions", side_effect=AssertionError("Location scoring ran")), \
                 patch.object(ev, "location_truth_summary", side_effect=AssertionError("Truth lookup ran")):
                disabled = ev.evaluate(gt, results, out_dir=out, evaluate_location=False)
            self.assertFalse(disabled["summary"]["evaluate_location"])
            self.assertIsNone(disabled["summary"]["location_truth"])
            self.assertEqual(disabled["regions"], [])
            self.assertNotIn("side_agreement", disabled["summary"])
            self.assertFalse((out / "regions.csv").exists())
            self.assertEqual(disabled["region_counts"], [])
            self.assertFalse((out / "region_counts.csv").exists())
            self.assertEqual(disabled["region_presence"], [])  # presence per region needs the location truth
            self.assertNotIn("region_presence", disabled["summary"])
            self.assertFalse((out / "region_presence.csv").exists())
            for key in ("presence", "whole_image", "counts", "per_image"):
                self.assertEqual(enabled[key], disabled[key])
            self.assertEqual(results, original)
            restored = ev.evaluate(gt, results, out_dir=out, evaluate_location=True)
            self.assertEqual(enabled, restored)
            self.assertTrue((out / "regions.csv").exists())

    def test_notebook_skips_adapter_without_credentials_or_local_runner(self):
        nb = json.loads(Path(__file__).with_name("main_notebook.ipynb").read_text(encoding="utf-8"))
        sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        cell = next(s for s in sources if "# CELL 9 -" in s)
        for truth in ("llm", "geometry"):
            scope = {"EXPERIMENTS": xp.build([{"name": "off", "evaluate_location": False,
                                               "location_truth": truth}]),
                     "DATASETS": [{"name": "toy"}], "MODE_USED": {}, "xp": xp}
            exec(compile(cell, "<location cell>", "exec"), scope)
            self.assertEqual(scope["ADAPTED"], {})
        # Location scoring is a per-experiment knob, and every experiment is evaluated with its own.
        evaluation = next(s for s in sources if "# CELL 10 -" in s)
        self.assertIn('evaluate_location=cfg["evaluate_location"]', evaluation)


if __name__ == "__main__":
    unittest.main()

