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
    def test_toggle_preserves_finding_scores(self):
        findings = {c: {"asked": True, "presence": "no", "whole_image": "no",
                        "regions": [], "region_count": 0} for c in dp.CONDITIONS}
        findings["dental_filling"].update(
            presence="yes", whole_image="yes",
            regions=["upper-left"], region_count=1)
        results = {"image": {"findings": findings, "call_count": 71,
                            "location_level": "regions",
                            "protocol": {"location": "regions"}}}
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
            self.assertEqual(disabled["region_presence"], [])  # presence per cell needs the location truth
            self.assertNotIn("region_presence", disabled["summary"])
            self.assertFalse((out / "region_presence.csv").exists())
            for key in ("presence", "whole_image", "per_image"):
                self.assertEqual(enabled[key], disabled[key])
            self.assertEqual(results, original)
            restored = ev.evaluate(gt, results, out_dir=out, evaluate_location=True)
            self.assertEqual(enabled, restored)
            self.assertTrue((out / "regions.csv").exists())

    def test_notebook_skips_adapter_without_credentials_or_local_runner(self):
        nb = json.loads(Path(__file__).with_name("main_notebook.ipynb").read_text(encoding="utf-8"))
        sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        cell = next(s for s in sources if "# CELL 10 -" in s)
        for truth in ("llm", "geometry"):
            scope = {"EXPERIMENTS": xp.build([{"name": "off", "evaluate_location": False,
                                               "location_truth": truth}]),
                     "DATASETS": [{"name": "toy"}], "xp": xp}
            exec(compile(cell, "<location cell>", "exec"), scope)
            self.assertEqual(scope["ADAPTED"], {})
        # Location scoring is a per-experiment knob, and every experiment is evaluated with its own.
        evaluation = next(s for s in sources if "# CELL 11 -" in s)
        self.assertIn('evaluate_location=cfg["evaluate_location"]', evaluation)


if __name__ == "__main__":
    unittest.main()

