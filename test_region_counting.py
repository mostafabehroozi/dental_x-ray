"""Offline tests for occupied-region counting: the count block and its statuses, the truth target, the
region-question stage, aggregation before counting, the four counting/location switch combinations,
the report, and the configuration gates.

A count here is a number of distinct regions a finding is reported in. Nothing in these tests counts
teeth, boxes, mentions or Yes answers, and nothing is asked of the analyzer that was not asked before.
"""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import dental_analysis as da
import dental_eval as ev
import dental_pipeline as dp
import experiments as xp
import llm_parser as lp
import report_writer as rw
from test_dental_pipeline import FakeRunner, _blank_image

UPPER_RIGHT = "the right posterior region of the upper dentition"
UPPER_ANTERIOR = "the anterior region of the upper dentition"
UPPER_LEFT = "the left posterior region of the upper dentition"
LOWER_RIGHT = "the right posterior region of the lower dentition"
LOWER_LEFT = "the left posterior region of the lower dentition"
BOTH_RIGHT = "the right posterior region of both the upper and lower dentition"

# img1's YOLO labels (class ids of dental_pipeline.CONDITIONS), several boxes per region on purpose.
LABELS = "\n".join([
    "4 0.20 0.20 0.05 0.05", "4 0.25 0.25 0.05 0.05", "4 0.20 0.30 0.05 0.05",  # three caries, upper-left
    "4 0.80 0.80 0.05 0.05", "4 0.85 0.80 0.05 0.05",                            # two caries, lower-right
    "2 0.20 0.20 0.05 0.05",                                                     # one filling, upper-left
    "6 0.80 0.80 0.10 0.10", "6 0.50 0.80 0.10 0.10",                            # impacted: lower-right, lower-anterior
    "8 0.20 0.80 0.05 0.05",                                                     # residual root, lower-left
    "1 0.50 0.20 0.05 0.05", "1 0.20 0.30 0.05 0.05",                            # restorations: upper-anterior, upper-left
]) + "\n"

SCRIPT = {
    # two regions, one of them named twice: the duplicate is one region
    ("presence", "caries", None): f"Yes\nCaries in {UPPER_LEFT}, also in {UPPER_LEFT}, and in {LOWER_RIGHT}.",
    # one descriptor covering both arches: two regions, neither the true one
    ("presence", "fillings", None): f"Yes\nFillings appear in {BOTH_RIGHT}.",
    # the right number of regions with one wrong identity
    ("presence", "impacted_tooth", None): f"Yes\nAn impacted tooth is seen in {LOWER_RIGHT} and {LOWER_LEFT}.",
    # reported without any location
    ("presence", "root_canal_therapy", None): "Yes\nRoot canal filling is present.",
    # unreadable decision
    ("presence", "periodontal_disease", None): "Yes and no.",
    # two tasks of one finding naming an overlapping region set
    ("presence", "prosthetic_crown", None): f"Yes\nA prosthetic crown in {UPPER_ANTERIOR}.",
    ("presence", "prosthetic_bridge", None): f"Yes\nA prosthetic bridge spans {UPPER_ANTERIOR} and {UPPER_LEFT}.",
    # residual root: not scripted, so the fake answers No to a true finding
}


def _box(xc, yc, w=0.05, h=0.05, **extra):
    return {"condition": "carious_lesion", "xc": xc, "yc": yc, "w": w, "h": h, **extra}


# ----------------------------------------------------------------------------
# The count block, the readers behind it, and the truth target
# ----------------------------------------------------------------------------
class CountBlockTests(unittest.TestCase):
    def test_statuses_keep_absence_missing_location_and_unreadable_apart(self):
        self.assertEqual(dp.count_block("no", None), {"region_count": 0, "count_status": "resolved"})
        self.assertEqual(dp.count_block("yes", ["upper-left", "lower-right"]), {"region_count": 2, "count_status": "resolved"})
        self.assertEqual(dp.count_block("yes", []), {"region_count": None, "count_status": "unlocated"})
        self.assertEqual(dp.count_block("yes", None), {"region_count": None, "count_status": "unresolved"})
        self.assertEqual(dp.count_block(None, None), {"region_count": None, "count_status": "unresolved"})
        self.assertEqual(dp.count_block("yes", ["upper-left"], ["lower-left"]), {"region_count": None, "count_status": "partial"})
        self.assertEqual(dp.count_block("no", None, location="none"), {"region_count": None, "count_status": "no_location"})
        for block in (dp.count_block("yes", []), dp.count_block("yes", ["upper-left"], ["lower-left"])):
            self.assertIsNone(block["region_count"], "only a resolved status carries a number")

    def test_finding_count_prefers_the_carried_block_and_derives_for_older_results(self):
        carried = {"asked": True, "presence": "yes", "regions": ["upper-left"], "region_count": None, "count_status": "partial"}
        self.assertEqual(dp.finding_count(carried, "regions"), {"region_count": None, "count_status": "partial"})
        legacy = {"asked": True, "presence": "yes", "regions": ["upper-left"]}
        self.assertEqual(dp.finding_count(legacy, "rationale"), {"region_count": 1, "count_status": "resolved"})
        self.assertEqual(dp.finding_count(legacy, "regions", ["lower-left"])["count_status"], "partial")
        self.assertEqual(dp.finding_count({"asked": False}, "rationale")["count_status"], "not_assessed")

    def test_the_strict_reader_dedups_mentions_and_expands_a_both_arches_descriptor(self):
        text = f"Yes\nCaries in {UPPER_LEFT}, again in {UPPER_LEFT}, and in {BOTH_RIGHT}."
        self.assertEqual(dp.extract_regions(text), ["upper-right", "upper-left", "lower-right"])
        self.assertEqual(dp.count_block("yes", dp.extract_regions(text))["region_count"], 3)
        # The strict reader cannot tell a denial from a report: that is the parser's job, and it is told so.
        self.assertEqual(dp.extract_regions(f"Yes\nCaries in {UPPER_LEFT}; no caries in {LOWER_LEFT}."),
                         ["upper-left", "lower-left"])

    def test_the_parser_dedups_and_its_prompt_carries_the_rules_and_the_code_side_mapping(self):
        regions, error = lp._read_location(json.dumps({"regions": ["lower-left", "lower-left", "upper-anterior"],
                                                       "unresolved": False}))
        self.assertEqual((regions, error), (["upper-anterior", "lower-left"], None))
        user = lp._fill(lp.LOCATION_USER, question_block="", text="t", cell_lines=lp._cell_lines(),
                        descriptor_lines=lp._descriptor_lines(), fdi_lines=lp._fdi_lines(), truncation=lp.COMPLETE_NOTE)
        for rule in ("rule the finding out", "some other finding", "too broad to map", "named twice is one identifier",
                     "never flip it"):
            self.assertIn(rule, user)
        self.assertNotIn("{fdi_lines}", user)
        # FDI quadrant 1 is the patient's upper right, which DentVLM calls its "left": the table the parser
        # is given is built from the same unit_cell mapping the scorer and the adapters use.
        lines = lp._fdi_lines().split("\n")
        for quadrant, line in zip((1, 2, 3, 4), lines):
            self.assertIn(f"quadrant {quadrant} ", line)
            self.assertIn(f"positions 1-3 -> {ev.fdi_cell(quadrant, 1)}", line)
            self.assertIn(f"positions 4-8 -> {ev.fdi_cell(quadrant, 6)}", line)
        self.assertIn("positions 4-8 -> upper-left", lines[0])
        self.assertEqual(lp.PROMPT_VERSION, 2)
        self.assertIn("left_is_image_left", lp.code_only().settings())


class TruthTargetTests(unittest.TestCase):
    def test_boxes_of_one_class_are_deduplicated_per_region(self):
        one_region = [_box(0.20, 0.20), _box(0.25, 0.25), _box(0.20, 0.30)]
        self.assertEqual(ev.truth_count(one_region), 1)
        self.assertEqual(ev.truth_count(one_region + [_box(0.80, 0.80), _box(0.85, 0.80)]), 2)
        self.assertEqual(ev.truth_count([]), 0)

    def test_one_primary_region_per_box_whatever_placed_it(self):
        # A box whole in two overlapping fixed windows: two cells for the location tables, one for the count.
        canine = _box(0.40, 0.20, 0.1, 0.1)
        self.assertEqual(ev.box_regions(canine), {"upper-left", "upper-anterior"})
        self.assertEqual(ev.box_primary_region(canine), "upper-anterior")  # equal shares: cell order
        self.assertEqual(ev.box_primary_region(_box(0.43, 0.20, 0.1, 0.1)), "upper-anterior")  # the larger share
        self.assertEqual(ev.box_primary_region(_box(0.37, 0.20, 0.1, 0.1)), "upper-left")
        # An LLM adapter naming two units: the same share rule decides; one area, FDI: exact.
        two_units = _box(0.37, 0.20, 0.1, 0.1, regions=["upper-anterior", "upper-left"], region_source="llm")
        self.assertEqual(ev.box_primary_region(two_units), "upper-left")
        self.assertEqual(ev.box_primary_region(_box(0.20, 0.20, regions=["upper-anterior"], region_source="areas")),
                         "upper-anterior")
        self.assertEqual(ev.box_primary_region(_box(0.80, 0.80, fdi=(1, 6))), "upper-left")  # patient's right = DentVLM's left
        # A box the adapter excluded has no region, and the finding's truth count is incomplete, not smaller.
        excluded = _box(0.20, 0.20, regions=[], region_source="excluded", location_excluded=True)
        self.assertIsNone(ev.box_primary_region(excluded))
        self.assertIsNone(ev.truth_count([_box(0.80, 0.80), excluded]))
        self.assertEqual(ev.box_primary_region(canine), ev.box_primary_region(dict(canine)))


# ----------------------------------------------------------------------------
# The whole-image protocol end to end: run, evaluate under every switch, report
# ----------------------------------------------------------------------------
@unittest.skipIf(importlib.util.find_spec("PIL") is None, "Pillow not installed")
class RationaleCountingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "images").mkdir()
        (self.root / "labels").mkdir()
        _blank_image(self.root / "images" / "img1.png")
        _blank_image(self.root / "images" / "img2.png", shade=100)
        (self.root / "labels" / "img1.txt").write_text(LABELS)
        (self.root / "labels" / "img2.txt").write_text("")
        self.images = {"img1": self.root / "images" / "img1.png", "img2": self.root / "images" / "img2.png"}
        self.gt = ev.load_yolo(self.root / "images", self.root / "labels")

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, protocol=dp.Protocol(), script=SCRIPT, name="run"):
        runner = FakeRunner(script)
        results = dp.load_results(dp.run_dataset(runner, self.images, self.root / name, protocol=protocol))
        return runner, results

    def test_findings_carry_the_count_and_its_status(self):
        runner, results = self._run()
        f = results["img1"]["findings"]
        counts = {c: (f[c]["region_count"], f[c]["count_status"]) for c in dp.TRAINED}
        self.assertEqual(counts, {
            "carious_lesion": (2, "resolved"), "dental_filling": (2, "resolved"), "impacted_tooth": (2, "resolved"),
            "endodontic_treatment": (None, "unlocated"), "periodontal_bone_loss": (None, "unresolved"),
            "root_fragment": (0, "resolved"), "prosthetic_restoration": (2, "resolved"),
            "dental_implant": (0, "resolved"), "periapical_lesion": (0, "resolved")})
        self.assertEqual(f["carious_lesion"]["regions"], ["upper-left", "lower-right"])  # the repeated region once
        self.assertEqual(f["prosthetic_restoration"]["regions"], ["upper-anterior", "upper-left"])  # two tasks, one set
        self.assertEqual(f["dental_filling"]["regions"], ["upper-right", "lower-right"])  # one descriptor, both arches
        self.assertTrue(all(f[c]["unresolved_regions"] == [] for c in dp.CONDITIONS))
        self.assertTrue(all(f[c]["whole_image_regions"] == f[c]["regions"] for c in dp.CONDITIONS))
        self.assertEqual(f["surgical_device"]["count_status"], "not_assessed")
        # Counting is derived: 13 questions per image, exactly as before, and nothing about it is a protocol knob.
        self.assertEqual((results["img1"]["call_count"], len(runner.log)), (13, 26))
        self.assertNotIn("counting", json.dumps(results["img1"]["protocol"]))
        manifest = json.loads((self.root / "run" / "manifest.json").read_text())
        self.assertEqual(manifest["findings_version"], 2)

    def test_count_metrics_cover_true_negatives_false_alarms_and_missed_findings(self):
        _, results = self._run()
        report = ev.evaluate(self.gt, results, dataset="toy", out_dir=self.root / "eval")
        rows = {r["condition"]: r for r in report["occupied_regions"]}
        self.assertTrue(all(r["stage"] == "presence" and r["target"] == "distinct_primary_regions" for r in rows.values()))
        exact = {c: (r["expected_count_checks"], r["scored_count_checks"], r["exact_rate"], r["mae"]) for c, r in rows.items()}
        self.assertEqual(exact["carious_lesion"], (2, 2, 1.0, 0.0))        # 5 boxes in 2 regions, 2 regions named
        self.assertEqual(exact["impacted_tooth"], (2, 2, 1.0, 0.0))        # the right count with a wrong region
        self.assertEqual(exact["prosthetic_restoration"], (2, 2, 1.0, 0.0))  # crown and bridge deduplicated
        self.assertEqual(exact["dental_implant"], (2, 2, 1.0, 0.0))        # absent, predicted absent: an exact 0
        self.assertEqual((rows["dental_filling"]["overcount_rate"], rows["dental_filling"]["mae"]), (0.5, 0.5))  # false regions
        self.assertEqual((rows["root_fragment"]["undercount_rate"], rows["root_fragment"]["exact_rate"]), (0.5, 0.5))  # missed
        self.assertEqual((rows["endodontic_treatment"]["scored_count_checks"], rows["endodontic_treatment"]["excluded_unlocated"]),
                         (1, 1))  # a false alarm without a region has no count, and is not a zero
        self.assertEqual(rows["periodontal_bone_loss"]["excluded_unresolved"], 1)
        summary = report["summary"]["occupied_regions"]
        self.assertEqual({k: summary[k] for k in ("expected_count_checks", "scored_count_checks", "excluded_count_checks",
                                                  "excluded_unlocated", "excluded_unresolved", "excluded_partial",
                                                  "excluded_truth_incomplete")},
                         {"expected_count_checks": 18, "scored_count_checks": 16, "excluded_count_checks": 2,
                          "excluded_unlocated": 1, "excluded_unresolved": 1, "excluded_partial": 0,
                          "excluded_truth_incomplete": 0})
        self.assertEqual((summary["exact_rate"], summary["mae"], summary["overcount_rate"], summary["undercount_rate"],
                          summary["exact_rate_of_expected"]), (0.875, 0.125, 0.0625, 0.0625, 0.7778))
        self.assertTrue(report["summary"]["counting"])
        self.assertTrue((self.root / "eval" / "occupied_regions.csv").is_file())
        # The spec's example: truth {A, B}, prediction {A, C}: the count is right and the location shows the mistake.
        impacted = next(r for r in report["regions"] if r["condition"] == "impacted_tooth")
        self.assertEqual({k: impacted[k] for k in ("TP", "FP", "FN", "TN")}, {"TP": 1, "FP": 1, "FN": 1, "TN": 3})
        presence = next(r for r in report["presence"] if r["condition"] == "impacted_tooth")
        self.assertEqual(presence["TP"], 1)
        # Diagnostics carry the same numbers, per finding and pooled.
        self.assertEqual(da.metrics(self.gt, report)["count_exact_rate"], 0.875)
        finding = next(r for r in da.finding_rows(self.gt, report) if r["condition"] == "dental_filling")
        self.assertEqual((finding["count_scored_count_checks"], finding["count_overcount_rate"]), (2, 0.5))
        overview = da.compact_views({"toy": self.gt}, {("base", "toy"): report})["experiment_overview"][0]
        self.assertEqual((overview["counting"], overview["count_exact_rate_of_expected"]), (True, 0.7778))
        statuses = {r["group"] for r in report["case_breakdown"] if r["situation"] == "count_status"}
        self.assertEqual(statuses, {"resolved", "unlocated", "unresolved"})

    def test_the_four_switch_combinations(self):
        _, results = self._run()
        out = self.root / "eval"
        both = ev.evaluate(self.gt, results, out_dir=out, evaluate_location=True, counting=True)
        self.assertTrue(both["regions"] and both["region_presence"] and both["occupied_regions"])
        location_only = ev.evaluate(self.gt, results, out_dir=out, evaluate_location=True, counting=False)
        self.assertTrue(location_only["regions"] and location_only["region_presence"])
        self.assertEqual(location_only["occupied_regions"], [])
        self.assertNotIn("occupied_regions", location_only["summary"])
        self.assertFalse(location_only["summary"]["counting"])
        self.assertFalse((out / "occupied_regions.csv").exists(), "re-exporting with counting off removes the table")
        self.assertIsNone(da.metrics(self.gt, location_only)["count_exact_rate"])
        self.assertNotIn("count_status", {r["situation"] for r in location_only["case_breakdown"]})
        with patch.object(ev, "gt_regions", side_effect=AssertionError("location scored")):
            count_only = ev.evaluate(self.gt, results, out_dir=out, evaluate_location=False, counting=True)
        self.assertEqual((count_only["regions"], count_only["region_presence"]), ([], []))
        self.assertEqual(count_only["occupied_regions"], both["occupied_regions"], "the same truth mapping either way")
        self.assertEqual(count_only["summary"]["location_truth"], both["summary"]["location_truth"])
        self.assertTrue((out / "occupied_regions.csv").is_file() and not (out / "regions.csv").exists())
        with patch.object(ev, "gt_regions", side_effect=AssertionError("location scored")), \
             patch.object(ev, "truth_count", side_effect=AssertionError("truth placed")):
            neither = ev.evaluate(self.gt, results, out_dir=out, evaluate_location=False, counting=False)
        self.assertEqual((neither["regions"], neither["region_presence"], neither["occupied_regions"]), ([], [], []))
        self.assertIsNone(neither["summary"]["location_truth"])
        for key in ("presence", "per_image"):
            self.assertEqual(both[key], neither[key])

    def test_aggregation_over_phrasings_happens_before_counting(self):
        class PhrasingRunner(FakeRunner):
            """The three caries wordings answer differently; every other question falls back to the script."""

            def ask(self, image, question):
                reply = super().ask(image, question)
                wordings = dp.questions_for("caries")
                if question in wordings and Path(image).stem == "img1":
                    reply["text"] = [f"Yes\nCaries in {UPPER_LEFT}.", f"Yes\nCaries in {UPPER_LEFT} and {LOWER_RIGHT}.",
                                     "Yes\nCaries is present."][wordings.index(question)]
                return reply

        runner = PhrasingRunner(SCRIPT)
        results = dp.load_results(dp.run_dataset(runner, self.images, self.root / "phrased", protocol=dp.Protocol(phrasings=3)))
        caries = results["img1"]["findings"]["carious_lesion"]
        self.assertEqual((caries["regions"], caries["region_count"]), (["upper-left", "lower-right"], 2))  # never 1 + 2 + 0
        self.assertEqual(len(results["img1"]["tasks"]["caries"]["answers"]), 3)  # the evidence behind the vote stays
        self.assertEqual(results["img1"]["call_count"], 39)
        report = ev.evaluate(self.gt, results, dataset="toy")
        modes = {r["region_vote"]: r for r in report["region_vote_comparison"]}
        self.assertEqual(next(r for r in report["occupied_regions"] if r["condition"] == "carious_lesion")["exact_rate"], 1.0)
        # A majority vote keeps only the region two wordings named: one region, an undercount against the same truth.
        self.assertLess(modes["majority"]["count_exact_rate"], modes["union"]["count_exact_rate"])
        self.assertEqual(modes["union"]["count_exact_rate"], report["summary"]["occupied_regions"]["exact_rate"])

    def test_incomplete_truth_is_never_a_smaller_count(self):
        _, results = self._run()
        truth = json.loads(json.dumps({i: {**e, "annotated": sorted(e["annotated"])} for i, e in self.gt.items()}))
        for entry in truth.values():
            entry["annotated"] = set(entry["annotated"])
        for box in truth["img1"]["boxes"]:
            if box["condition"] == "carious_lesion" and box["xc"] > 0.5:
                box.update(regions=[], region_source="excluded", location_excluded=True)  # the lower-right boxes
        row = next(r for r in ev.evaluate(truth, results, dataset="toy")["occupied_regions"] if r["condition"] == "carious_lesion")
        self.assertEqual((row["scored_count_checks"], row["excluded_truth_incomplete"]), (1, 1))

    def test_the_report_states_regions_never_teeth_and_respects_the_switch(self):
        _, results = self._run()
        result = results["img1"]
        on = rw.structured_findings(result, "DentVLM")
        by = {f["finding"]: f for f in on["findings"]}
        self.assertEqual((by["carious_lesion"]["multiplicity"], by["impacted_tooth"]["multiplicity"]), (2, 2))
        self.assertTrue(by["endodontic_treatment"]["multiplicity"].startswith("not_stated"))
        self.assertEqual((by["periodontal_bone_loss"]["multiplicity"], by["root_fragment"]["multiplicity"],
                          by["surgical_device"]["multiplicity"]), ("not_applicable",) * 3)
        self.assertIn("never lesions or teeth", on["legend"]["multiplicity"])
        prompt = rw.user_prompt(on, "English")
        self.assertIn('never "two lesions" or "two teeth"', prompt)
        self.assertNotIn("{multiplicity_data}", prompt)
        self.assertEqual(rw.verify_report(_good_report(on), on), [])
        off = rw.structured_findings(result, "DentVLM", counting=False)
        self.assertNotIn("multiplicity", json.dumps(off))
        self.assertNotIn("multiplicity", rw.user_prompt(off, "English"))
        self.assertNotIn("{multiplicity_rule}", rw.user_prompt(off, "English"))
        self.assertIn("in 2 region(s): ", dp.dentist_report(result))
        self.assertIn("Dental caries; regions: ", dp.dentist_report(result, counting=False))
        self.assertNotIn("region(s)", dp.dentist_report(result, counting=False))
        self.assertIn("Dental caries; regions: ", rw.fallback_markdown(result, ["x"], counting=False))
        with patch.object(rw.llm_api, "connect", return_value=object()):
            writer = rw.ReportWriter.from_api({"base_url": "http://local/v1", "api_key": "k", "model": "m"}, counting=False)
        self.assertFalse(writer.counting)
        self.assertNotEqual(writer.settings(), rw.ReportWriter(None, "k", "m", client=object()).settings())


def _good_report(structured: dict) -> dict:
    status = {f["finding"]: f["status"] for f in structured["findings"]}
    return {"title": "Report", "headings": {"image": "Image", "findings": "Findings", "impression": "Impression",
                                            "not_assessable": "Not assessable", "limitations": "Limitations"},
            "sections": [{"category": c["key"], "heading": c["label"],
                          "findings": [{"finding": k, "status": status[k], "statement": f"About {k}."} for k in c["findings"]]}
                         for c in structured["categories"]],
            "impression": ["Findings as listed."],
            "not_assessable": [f"{c} could not be assessed." for c in structured["summary"]["unparseable"]],
            "limitations": list(structured["analysis"]["limitations"])}


# ----------------------------------------------------------------------------
# The region-question protocol: partial totals, recovery after a whole-image No, both stages
# ----------------------------------------------------------------------------
@unittest.skipIf(importlib.util.find_spec("PIL") is None, "Pillow not installed")
class RegionalCountingTests(unittest.TestCase):
    SCRIPT = {
        ("presence", "fillings", None): f"Yes\nFillings in {UPPER_LEFT}.",
        ("region", "fillings", "upper-left"): "Yes\nFillings are visible.",
        ("presence", "caries", None): "No\nNo caries is seen.",              # missed on the whole image
        ("region", "caries", "upper-left"): "Yes",
        ("region", "caries", "lower-right"): "Yes",
        ("region", "caries", "lower-left"): "Yes and no.",                    # one region unresolved
        ("presence", "impacted_tooth", None): f"Yes\nAn impacted tooth in {LOWER_RIGHT}.",  # every region then answers No
        ("presence", "implant", None): "Yes and no.",                         # unreadable, then six No regions
        ("region", "root_canal_therapy", "upper-right"): "Yes",              # a false alarm found by a region only
    }

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "images").mkdir()
        (self.root / "labels").mkdir()
        _blank_image(self.root / "images" / "img1.png")
        (self.root / "labels" / "img1.txt").write_text(LABELS)
        self.images = {"img1": self.root / "images" / "img1.png"}
        self.gt = ev.load_yolo(self.root / "images", self.root / "labels")
        self.runner = FakeRunner(self.SCRIPT)
        self.results = dp.load_results(dp.run_dataset(self.runner, self.images, self.root / "run",
                                                      protocol=dp.Protocol(location="regions")))

    def tearDown(self):
        self.tmp.cleanup()

    def test_partial_and_recovered_findings(self):
        f = self.results["img1"]["findings"]
        caries = f["carious_lesion"]
        self.assertEqual((caries["presence"], caries["regions"], caries["unresolved_regions"]),
                         ("yes", ["upper-left", "lower-right"], ["lower-left"]))
        self.assertEqual((caries["region_count"], caries["count_status"]), (None, "partial"))
        self.assertEqual((caries["whole_image"], caries["whole_image_regions"]), ("no", None))
        self.assertEqual((f["dental_filling"]["region_count"], f["dental_filling"]["whole_image_regions"]), (1, ["upper-left"]))
        self.assertEqual((f["impacted_tooth"]["presence"], f["impacted_tooth"]["region_count"],
                          f["impacted_tooth"]["whole_image_regions"]), ("no", 0, ["lower-right"]))
        self.assertEqual((f["dental_implant"]["presence"], f["dental_implant"]["region_count"], f["dental_implant"]["whole_image"]),
                         ("no", 0, None))
        self.assertEqual((f["endodontic_treatment"]["region_count"], f["endodontic_treatment"]["whole_image"]), (1, "no"))
        self.assertEqual((self.results["img1"]["call_count"], len(self.runner.log)), (91, 91))

    def test_both_stages_are_scored_and_a_partial_total_is_excluded(self):
        report = ev.evaluate(self.gt, self.results, dataset="toy", out_dir=self.root / "eval")
        rows = {(r["condition"], r["stage"]): r for r in report["occupied_regions"]}
        self.assertEqual({s for _, s in rows}, {"presence", "whole_image"})
        self.assertEqual((rows["carious_lesion", "presence"]["excluded_partial"], rows["carious_lesion", "presence"]["scored_count_checks"]),
                         (1, 0))
        self.assertEqual((rows["carious_lesion", "whole_image"]["undercount_rate"], rows["carious_lesion", "whole_image"]["mae"]),
                         (1.0, 2.0))  # the rationale stage answered No to five boxes in two regions
        self.assertEqual((rows["endodontic_treatment", "presence"]["overcount_rate"],
                          rows["endodontic_treatment", "whole_image"]["exact_rate"]), (1.0, 1.0))
        self.assertEqual((rows["impacted_tooth", "presence"]["mae"], rows["impacted_tooth", "whole_image"]["mae"]), (2.0, 1.0))
        self.assertEqual((rows["dental_implant", "presence"]["exact_rate"], rows["dental_implant", "whole_image"]["excluded_unresolved"]),
                         (1.0, 1))
        self.assertEqual(report["summary"]["occupied_regions"]["excluded_partial"], 1)  # the authoritative stage only
        # An older artifact without the block: the unresolved region is reconstructed from the saved calls.
        legacy = json.loads(json.dumps(self.results))
        for finding in legacy["img1"]["findings"].values():
            for key in ("region_count", "count_status", "unresolved_regions", "whole_image_regions"):
                finding.pop(key, None)
        old = ev.evaluate(self.gt, legacy, dataset="toy")
        old_rows = {(r["condition"], r["stage"]): r for r in old["occupied_regions"]}
        self.assertEqual(old_rows["carious_lesion", "presence"]["excluded_partial"], 1)
        self.assertNotIn(("carious_lesion", "whole_image"), old_rows)

    def test_the_report_says_at_least(self):
        structured = rw.structured_findings(self.results["img1"], "DentVLM")
        caries = next(f for f in structured["findings"] if f["finding"] == "carious_lesion")
        self.assertTrue(caries["multiplicity"].startswith("at least 2: reported in 2 region(s)"))
        # The report speaks on the patient's side: DentVLM's "lower-left" is the patient's lower right.
        self.assertIn("lower-right-posterior", caries["multiplicity"])
        self.assertEqual(caries["regions"]["lower-right-posterior"], "unparseable")
        self.assertIn("Dental caries; in at least 2 region(s) (1 region(s) could not be read): ",
                      dp.dentist_report(self.results["img1"]))
        self.assertEqual(next(f for f in structured["findings"] if f["finding"] == "dental_implant")["multiplicity"],
                         "not_applicable")


# ----------------------------------------------------------------------------
# Configuration: the switch, its gates, and what it must never touch
# ----------------------------------------------------------------------------
class ConfigurationTests(unittest.TestCase):
    SHARED = {"output_root": "/tmp/out", "backend": "local", "location_truth": "geometry"}

    def test_counting_is_a_boolean_evaluation_switch_outside_the_protocol(self):
        on, off = xp.build([{"name": "on"}, {"name": "off", "counting": False}], self.SHARED)
        self.assertTrue(on["counting"] and not off["counting"])
        self.assertEqual(xp.protocol(on), xp.protocol(off), "counting never changes a question or a manifest")
        self.assertNotIn("counting", json.dumps(dp.run_config(xp.protocol(on), {"model": "fake"})))
        with self.assertRaisesRegex(ValueError, "counting must be True or False"):
            xp.build([{"name": "bad", "counting": "yes"}], self.SHARED)
        with self.assertRaisesRegex(ValueError, "counting needs region evidence"):
            xp.build([{"name": "none", "location": "none"}], self.SHARED)
        xp.build([{"name": "none-off", "location": "none", "counting": False}], self.SHARED)
        for combination in ({"evaluate_location": True, "counting": True}, {"evaluate_location": True, "counting": False},
                            {"evaluate_location": False, "counting": True}, {"evaluate_location": False, "counting": False}):
            cfg, = xp.build([{"name": "c", **combination}], self.SHARED)
            self.assertEqual(xp.uses_location_truth(cfg), combination["evaluate_location"] or combination["counting"])
        self.assertIn("counting", [row for row in xp.table([on, off])][0], "a differing switch shows in the table")

    def test_a_hosted_adapter_is_required_and_built_for_counting_alone(self):
        with self.assertRaisesRegex(ValueError, "adapter"):
            xp.build([{"name": "x", "evaluate_location": False, "counting": True, "location_truth": "areas",
                       "adapter": {"model": None}}], self.SHARED)
        xp.build([{"name": "x", "evaluate_location": False, "counting": False, "location_truth": "areas",
                   "adapter": {"model": None}}], self.SHARED)  # nothing needs the truth: no adapter is checked
        cfg, = xp.build([{"name": "x", "evaluate_location": False, "counting": True, "location_truth": "areas",
                          "adapter": {"base_url": "http://local/v1", "api_key": "k"}}], self.SHARED)
        with patch.object(xp.llm_api, "connect", return_value=object()):
            self.assertEqual(xp.location_adapter(cfg).kind, "areas")
            writer = xp.report_writer({**cfg, "counting": False, "reporter": {**cfg["reporter"], "api_key": "k",
                                                                               "base_url": "http://local/v1"}})
            self.assertFalse(writer.counting)
            self.assertTrue(xp.report_writer({**cfg, "reporter": {**cfg["reporter"], "api_key": "k",
                                                                   "base_url": "http://local/v1"}}).counting)


if __name__ == "__main__":
    unittest.main()
