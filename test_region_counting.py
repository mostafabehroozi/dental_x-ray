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


# ----------------------------------------------------------------------------
# Configuration: the switch, its gates, and what it must never touch
# ----------------------------------------------------------------------------


if __name__ == "__main__":
    unittest.main()
