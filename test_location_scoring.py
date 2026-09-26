"""Offline checks for the independent location-scoring switch."""
import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import dental_eval as ev
import experiments as xp
import dental_pipeline as dp
import location_adapter as la
import run_monitor as mon


class LocationScoringTests(unittest.TestCase):
    def test_toggle_preserves_finding_scores(self):
        findings = {c: {"asked": True, "presence": "no", "whole_image": "no",
                        "regions": [], "region_count": 0} for c in ev.CONDITIONS}
        findings["dental_filling"].update(
            presence="yes", whole_image="yes",
            regions=["upper-left"], region_count=1)
        results = {"image": {"findings": findings, "call_count": 71,
                            "location_level": "regions",
                            "protocol": {"location": "regions"}}}
        gt = {"image": {"annotated": set(ev.CONDITIONS), "boxes": [
            {"condition": "dental_filling", "xc": 0.2, "yc": 0.25, "w": 0.05, "h": 0.05}]}}
        original = copy.deepcopy(results)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            enabled = ev.evaluate(gt, results, out_dir=out)
            self.assertTrue(enabled["regions"])
            self.assertTrue(enabled["region_presence"] and enabled["summary"]["region_presence"]["TP"] == 1)
            self.assertTrue((out / "regions.csv").exists() and (out / "region_presence.csv").exists())
            # Location off, counting off: no true box is placed at all.
            with patch.object(ev, "gt_regions", side_effect=AssertionError("Location scoring ran")), \
                 patch.object(ev, "location_truth_summary", side_effect=AssertionError("Truth lookup ran")):
                disabled = ev.evaluate(gt, results, out_dir=out, evaluate_location=False, counting=False)
            self.assertFalse(disabled["summary"]["evaluate_location"])
            self.assertIsNone(disabled["summary"]["location_truth"])
            self.assertEqual(disabled["regions"], [])
            self.assertNotIn("side_agreement", disabled["summary"])
            self.assertFalse((out / "regions.csv").exists())
            self.assertEqual(disabled["region_presence"], [])  # presence per cell needs the location truth
            self.assertNotIn("region_presence", disabled["summary"])
            self.assertFalse((out / "region_presence.csv").exists())
            self.assertEqual(disabled["occupied_regions"], [])
            self.assertNotIn("occupied_regions", disabled["summary"])
            self.assertFalse((out / "occupied_regions.csv").exists())
            for key in ("presence", "whole_image", "per_image"):
                self.assertEqual(enabled[key], disabled[key])
            self.assertEqual(results, original)
            # Location off, counting on: the true boxes are still placed, for the count target only.
            with patch.object(ev, "gt_regions", side_effect=AssertionError("Location scoring ran")):
                counted = ev.evaluate(gt, results, out_dir=out, evaluate_location=False, counting=True)
            self.assertEqual((counted["regions"], counted["region_presence"]), ([], []))
            self.assertEqual(counted["summary"]["location_truth"], {"boxes": 1, "by_source": {"geometry": 1}})
            self.assertEqual(counted["summary"]["occupied_regions"]["exact_rate"], 1.0)
            self.assertTrue((out / "occupied_regions.csv").exists())
            self.assertFalse((out / "regions.csv").exists())
            restored = ev.evaluate(gt, results, out_dir=out, evaluate_location=True)
            self.assertEqual(enabled, restored)
            self.assertTrue((out / "regions.csv").exists())

    def test_notebook_skips_adapter_without_credentials_or_local_runner(self):
        nb = json.loads(Path(__file__).with_name("main_notebook.ipynb").read_text(encoding="utf-8"))
        sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        cell = next(s for s in sources if "# CELL 10 -" in s)
        for truth in ("llm", "geometry"):
            scope = {"EXPERIMENTS": xp.build([{"name": "off", "evaluate_location": False, "counting": False,
                                               "location_truth": truth}]),
                     "DATASETS": [{"name": "toy"}], "xp": xp, "GT": {"toy": {}}}
            exec(compile(cell, "<location cell>", "exec"), scope)
            self.assertEqual(scope["ADAPTED"], {})
        # Counting alone still needs the adapted truth: the cell reaches for the adapter (and, with no
        # provider configured here, records that failure instead of skipping the adaptation silently).
        ledger = mon.Ledger("test")
        scope = {"EXPERIMENTS": xp.build([{"name": "count-only", "evaluate_location": False, "counting": True,
                                           "location_truth": "areas", "output_root": "/tmp/out"}]),
                 "DATASETS": [{"name": "toy"}], "xp": xp, "mon": mon, "LEDGER": ledger, "GT": {"toy": {"image": {"boxes": [{"condition": "carious_lesion"}]}}},
                 "la": la, "ev": ev, "open_runner": lambda cfg: None, "open_parser": lambda cfg: None}
        with redirect_stdout(io.StringIO()):
            exec(compile(cell, "<location cell>", "exec"), scope)
        self.assertEqual([e["scope"] for e in ledger.entries], ["location count-only/toy"])
        # Location scoring and counting are per-experiment knobs, and every experiment is evaluated with its own.
        evaluation = next(s for s in sources if "# CELL 11 -" in s)
        self.assertIn('evaluate_location=cfg["evaluate_location"], counting=cfg["counting"]', evaluation)
        inspection = next(s for s in sources if "# CELL 13 -" in s)
        self.assertIn('dp.dentist_report(result, counting=False)', inspection)


if __name__ == "__main__":
    unittest.main()

