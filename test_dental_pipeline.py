"""Offline tests: extraction rules, questions, the run loop with a fake model, and evaluation."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import dental_eval as ev
import dental_pipeline as dp

UPPER_LEFT = "the left posterior region of the upper dentition"
LOWER_RIGHT = "the right posterior region of the lower dentition"
LOWER_LEFT = "the left posterior region of the lower dentition"
UPPER_ANTERIOR = "the anterior region of the upper dentition"


class ExtractionTests(unittest.TestCase):
    def test_answer_forms(self):
        cases = {
            "Yes\nThe imaging shows fillings in the left posterior region of the upper dentition.": "yes",
            "No\nNo caries is observed in the image.": "no",
            "Yes.": "yes",
            "yes, there is a filling": "yes",
            "No, there is no evidence of an implant.": "no",
            "\nNo\n": "no",
            "Yes and no.": None,
            "Normal": None,
            "There is nothing to report.": None,
            "": None,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(dp.extract_answer(text), expected)

    def test_regions_from_descriptors(self):
        text = f"Yes\nProsthetic crowns are observed in {UPPER_ANTERIOR} and {LOWER_LEFT}."
        self.assertEqual(dp.extract_regions(text), ["upper-anterior", "lower-left"])
        both = "Yes\nFillings are seen in the right posterior region of both the upper and lower dentition."
        self.assertEqual(dp.extract_regions(both), ["upper-right", "lower-right"])
        self.assertEqual(dp.extract_regions("Yes\nCaries is suspected."), [])
        self.assertEqual(dp.extract_regions("The Left Posterior Region Of The Upper Dentition"), ["upper-left"])
        self.assertEqual(dp.extract_regions("no fillings in the anterior region of the lower dentition"), ["lower-anterior"])

    def test_questions_are_verbatim_and_ordered(self):
        self.assertEqual(dp.questions_for("impacted_tooth")[0],
                         "Based on the imaging, determine whether the patient has an impacted tooth?")
        self.assertEqual(dp.questions_for("fillings")[0], "Based on the imaging analysis, does the patient have fillings?")
        for task, spec in dp.TASKS.items():
            self.assertEqual(len(spec["questions"]), dp.MAX_PHRASINGS, task)
            self.assertEqual(len(set(spec["questions"])), dp.MAX_PHRASINGS, task)
        self.assertEqual(dp.questions_for("furcation_lesion"),
                         ("Based on the imaging, determine whether the patient has furcation involvement?",))
        self.assertEqual(len(dp.CONDITIONS), 14)
        self.assertEqual(len(dp.TASKS), 13)
        self.assertEqual(dp.Protocol().tasks(), (
            "implant", "prosthetic_crown", "prosthetic_bridge", "fillings", "root_canal_therapy", "caries",
            "periodontal_disease", "impacted_tooth", "apical_periodontitis", "residual_root",
            "residual_crown", "insufficient_eruption_space", "calculus"))
        self.assertEqual(len(dp.Protocol(ask_untrained=True).tasks()), 18)
        self.assertEqual(len(dp.Protocol(extra_tasks=False).tasks()), 10)
        self.assertEqual(dp.condition_tasks("surgical_device"), ())
        self.assertEqual(dp.condition_tasks("surgical_device", ask_untrained=True), ("surgical_device",))
        with self.assertRaises(ValueError):
            dp.Protocol(phrasings=4)

    def test_vote(self):
        answers = [{"answer": "yes", "regions": ["upper-left"]}, {"answer": "no", "regions": []},
                   {"answer": "yes", "regions": ["upper-left", "lower-right"]}]
        self.assertEqual(dp.vote(answers, "union"), {"presence": "yes", "regions": ["upper-left", "lower-right"]})
        self.assertEqual(dp.vote(answers, "majority"), {"presence": "yes", "regions": ["upper-left"]})
        tie = [{"answer": "yes", "regions": []}, {"answer": "no", "regions": []}, {"answer": None, "regions": []}]
        self.assertEqual(dp.vote(tie, "union"), {"presence": None, "regions": None})
        self.assertEqual(dp.vote([{"answer": "no", "regions": []}], "union"), {"presence": "no", "regions": None})

    def test_cell_descriptions(self):
        self.assertEqual(dp.describe_cell("upper-left"), "patient's upper right posterior (image left)")
        self.assertEqual(dp.describe_cell("lower-right"), "patient's lower left posterior (image right)")
        self.assertEqual(dp.describe_cell("upper-anterior"), "upper anterior")
        self.assertEqual(dp.describe_cell("upper-left", left_is_image_left=False),
                         "patient's upper left posterior (image right)")
        self.assertEqual(dp.cell_windows(False)["upper-left"], dp.cell_windows(True)["upper-right"])


def _blank_image(path: Path, size=(560, 280), shade: int = 128) -> None:
    """A test radiograph with a mark in every region window."""
    from PIL import Image, ImageDraw

    image = Image.new("L", size, color=shade)
    draw = ImageDraw.Draw(image)
    for index, (left, top, right, bottom) in enumerate(dp.CELL_WINDOWS.values()):
        x, y = int(left * size[0]) + 5 + index * 3, int(top * size[1]) + 5
        draw.rectangle((x, y, x + 12, y + 12), fill=20 + 30 * index)
    image.save(path)


class FakeRunner:
    """Answers from a script keyed by (stage, task, cell); the region asked is read out of the question."""

    def __init__(self, script: dict, scripted_image: str = "img1"):
        self.script, self.log = script, []
        self.scripted_image = scripted_image  # other images always answer No

    def settings(self):
        return {"model": "fake"}

    def ask(self, image, question):
        assert not isinstance(image, (bytes, bytearray)), "no protocol crops the image; the region is words"
        keys = list(dp.TASKS) + list(dp.UNTRAINED_LABELS)
        cell = next((c for c, d in dp.CELL_DESCRIPTORS.items() if question.endswith(f" in {d}?")), None)
        if cell is not None:  # a region question: the same whole image with the region named in it
            stage = "region"
            task = next(t for t in keys if dp.region_question(t, cell) == question)
        else:
            stage = "presence"
            task = next(t for t in keys if question in dp.questions_for(t))
        scripted = Path(image).stem == self.scripted_image
        key = (stage, task, cell)
        self.log.append(key)
        text = "No\nNothing of the kind is seen."
        if scripted:
            text = self.script.get(key, text)
        return {"text": text, "finish_reason": "stop", "truncated": False,
                "prompt_tokens": 100, "completion_tokens": 5, "latency_seconds": 0.0}


SCRIPT = {
    ("presence", "fillings", None): f"Yes\nFillings appear as radiopaque material in {UPPER_LEFT}.",
    ("presence", "impacted_tooth", None): f"Yes\nAn impacted tooth is seen in {LOWER_RIGHT} and {LOWER_LEFT}.",
    ("presence", "caries", None): "Yes\nCaries is suspected.",  # false alarm without a region
    ("presence", "prosthetic_bridge", None): f"Yes\nA prosthetic bridge is seen in {UPPER_ANTERIOR}.",
    ("presence", "residual_crown", None): "Yes\nA residual crown is visible.",
    ("presence", "implant", None): "Yes and no.",  # unparseable
}


class RunAndEvaluateTests(unittest.TestCase):
    def setUp(self):
        if importlib.util.find_spec("PIL") is None:
            self.skipTest("Pillow not installed")
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "images").mkdir()
        (self.root / "labels").mkdir()
        _blank_image(self.root / "images" / "img1.png")
        _blank_image(self.root / "images" / "img2.png", shade=100)
        # img1: two fillings in the upper image-left cell, one impacted tooth in the lower image-right cell.
        (self.root / "labels" / "img1.txt").write_text(
            "2 0.20 0.25 0.05 0.05\n2 0.30 0.30 0.05 0.05\n6 0.80 0.80 0.10 0.10\n")
        # img2: nothing (empty label file).
        (self.root / "labels" / "img2.txt").write_text("")
        self.images = {"img1": self.root / "images" / "img1.png", "img2": self.root / "images" / "img2.png"}

    def tearDown(self):
        self.tmp.cleanup()

    def test_run_then_evaluate(self):
        runner = FakeRunner(SCRIPT)
        out = dp.run_dataset(runner, self.images, self.root / "run")

        results = dp.load_results(out)
        self.assertEqual(set(results), {"img1", "img2"})
        f = results["img1"]["findings"]
        self.assertEqual(f["dental_filling"], {"asked": True, "tasks": ["fillings"], "presence": "yes", "whole_image": "yes",
                                               "whole_image_regions": ["upper-left"], "regions": ["upper-left"],
                                               "unresolved_regions": [], "region_count": 1, "count_status": "resolved"})
        self.assertEqual((f["impacted_tooth"]["regions"], f["impacted_tooth"]["region_count"]), (["lower-right", "lower-left"], 2))
        self.assertEqual(f["prosthetic_restoration"]["tasks"], ["prosthetic_crown", "prosthetic_bridge"])
        self.assertEqual((f["prosthetic_restoration"]["presence"], f["prosthetic_restoration"]["regions"]),
                         ("yes", ["upper-anterior"]))
        # Reported without a region: the count is unavailable, not zero.
        self.assertEqual((f["carious_lesion"]["regions"], f["carious_lesion"]["region_count"], f["carious_lesion"]["count_status"]),
                         ([], None, "unlocated"))
        self.assertEqual((f["dental_implant"]["presence"], f["dental_implant"]["count_status"]), (None, "unresolved"))
        self.assertEqual((f["surgical_device"]["asked"], f["surgical_device"]["count_status"]), (False, "not_assessed"))
        self.assertEqual(f["root_fragment"], {"asked": True, "tasks": ["residual_root"], "presence": "no", "whole_image": "no",
                                             "whole_image_regions": None, "regions": None, "unresolved_regions": [],
                                             "region_count": 0, "count_status": "resolved"})
        self.assertEqual(results["img1"]["tasks"]["residual_crown"]["presence"], "yes")
        self.assertEqual(results["img1"]["call_count"], 13)
        self.assertEqual(results["img2"]["call_count"], 13)
        self.assertTrue((out / "manifest.json").is_file())

        # Resume skips finished images and rejects a different protocol.
        before = len(runner.log)
        dp.run_dataset(runner, {"img1": self.images["img1"]}, out)
        self.assertEqual(len(runner.log), before)
        with self.assertRaises(ValueError):
            dp.run_dataset(runner, {"img1": self.images["img1"]}, out, protocol=dp.Protocol(phrasings=3))

        gt = ev.load_yolo(self.root / "images", self.root / "labels")
        report = ev.evaluate(gt, results, dataset="toy", out_dir=out / "evaluation")
        presence = {r["condition"]: r for r in report["presence"]}
        self.assertNotIn("surgical_device", presence)
        self.assertEqual(report["summary"]["not_assessed"],
                         ["furcation_lesion", "apical_surgery", "root_resorption", "orthodontic_device", "surgical_device"])
        self.assertEqual((presence["dental_filling"]["TP"], presence["dental_filling"]["FN"], presence["dental_filling"]["TN"]), (1, 0, 1))
        self.assertTrue(presence["dental_filling"]["trained_task"])
        self.assertEqual(presence["carious_lesion"]["FP"], 1)
        self.assertEqual((presence["dental_implant"]["unparseable"], presence["dental_implant"]["TN"]), (1, 1))
        self.assertEqual(report["whole_image"], [])  # presence is the whole-image answer here
        regions = {r["condition"]: r for r in report["regions"]}
        self.assertEqual(regions["dental_filling"]["exact_set_match_rate"], 1.0)
        self.assertEqual((regions["impacted_tooth"]["TP"], regions["impacted_tooth"]["FP"]), (1, 1))
        self.assertNotIn("carious_lesion", {r["condition"] for r in report["regions"] if r["n_localized_cases"]})
        # Presence per cell: every cell of every asked image; a named cell is present, an unnamed one absent, and
        # two filling boxes in one cell are one presence. Unasked findings and the unresolved implant image are out.
        rp = {(r["condition"], r["region"]): r for r in report["region_presence"]}
        self.assertEqual(len(rp), 9 * 6)
        self.assertEqual((rp[("dental_filling", "upper-left")]["TP"], rp[("dental_filling", "upper-left")]["positives"],
                          rp[("dental_filling", "upper-left")]["TN"]), (1, 1, 1))
        self.assertEqual((rp[("impacted_tooth", "lower-right")]["TP"], rp[("impacted_tooth", "lower-left")]["FP"]), (1, 1))
        self.assertEqual((rp[("prosthetic_restoration", "upper-anterior")]["FP"], rp[("carious_lesion", "upper-left")]["TN"]), (1, 2))
        self.assertEqual((rp[("dental_implant", "upper-left")]["images"], rp[("dental_implant", "upper-left")]["unparseable"]), (1, 0))
        self.assertEqual({k: report["summary"]["region_presence"][k] for k in ("TP", "FP", "TN", "FN", "unparseable")},
                         {"TP": 2, "FP": 2, "TN": 98, "FN": 0, "unparseable": 0})
        self.assertTrue((out / "evaluation" / "region_presence.csv").is_file())
        self.assertEqual(ev.evaluate(gt, results, dataset="toy", evaluate_location=False)["region_presence"], [])
        summary = report["summary"]
        self.assertEqual(summary["images_scored"], 2)
        self.assertEqual(summary["complete_case_rate"], 1.0)
        self.assertEqual(summary["mean_false_alarms_per_image"], 1.0)
        self.assertTrue((out / "evaluation" / "presence.csv").is_file())
        # The count question is gone: no counts table, and a counts.csv left by an older
        # version is removed on re-export rather than sitting next to fresh metrics.
        self.assertNotIn("counts", report)
        (out / "evaluation" / "counts.csv").write_text("condition,mae\ndental_filling,3.0\n")
        ev.evaluate(gt, results, dataset="toy", out_dir=out / "evaluation")
        self.assertFalse((out / "evaluation" / "counts.csv").exists())
        self.assertEqual(ev.side_agreement(gt, results), {"sides_named": 3, "agree": 2, "agreement_rate": 0.6667,
                                                          "left_is_image_left": True})
        text = dp.dentist_report(results["img1"])
        self.assertIn("Dental filling; in 1 region(s): patient's upper right posterior (image left)", text)
        self.assertIn("Dental caries; region not stated", text)
        self.assertIn("Also present (no benchmark class): Residual Crown", text)
        self.assertIn("Not assessable (unparseable answer): Dental implant", text)
        self.assertIn("Not assessed by this model: Furcation involvement", text)

    def test_region_location_mode(self):
        """Every region is asked every task, whatever the whole image answered; the whole image is kept separately."""
        script = dict(SCRIPT)
        script[("region", "fillings", "upper-left")] = "Yes\nFillings are visible."
        script[("region", "impacted_tooth", "lower-right")] = "Yes"
        script[("region", "root_canal_therapy", "upper-right")] = "Yes"  # missed on the whole image, found by region
        script[("region", "residual_root", "upper-anterior")] = "Yes and no."  # one unparseable region
        runner = FakeRunner(script)
        out = dp.run_dataset(runner, self.images, self.root / "run_regions", protocol=dp.Protocol(location="regions"))
        results = dp.load_results(out)
        f = results["img1"]["findings"]
        self.assertEqual((f["dental_filling"]["presence"], f["dental_filling"]["regions"]), ("yes", ["upper-left"]))
        self.assertEqual(f["impacted_tooth"]["regions"], ["lower-right"])
        self.assertEqual((f["endodontic_treatment"]["presence"], f["endodontic_treatment"]["whole_image"],
                          f["endodontic_treatment"]["regions"]), ("yes", "no", ["upper-right"]))
        # Caries and the bridge were yes on the whole image only: no region answers yes, so they are absent.
        self.assertEqual((f["carious_lesion"]["presence"], f["carious_lesion"]["whole_image"]), ("no", "yes"))
        self.assertEqual((f["prosthetic_restoration"]["presence"], f["prosthetic_restoration"]["regions"]), ("no", None))
        self.assertEqual(results["img1"]["tasks"]["implant"]["presence"], "no")  # unparseable on the whole image only
        self.assertIsNone(results["img1"]["findings"]["root_fragment"]["presence"])  # no yes, one unparseable region
        self.assertEqual((dp.cell_answers(results["img1"])["fillings"]["upper-left"],
                          dp.cell_answers(results["img1"])["residual_root"]["upper-anterior"]), ("yes", None))
        # 13 whole-image calls + 13 tasks x 6 regions, for every image, and no image was ever cropped.
        self.assertEqual((results["img1"]["call_count"], results["img2"]["call_count"]), (91, 91))
        region_calls = [c for c in results["img1"]["calls"] if c["stage"] == "region"]
        self.assertEqual(len(region_calls), 78)
        self.assertTrue(all(c["question"].endswith(f" in {dp.CELL_DESCRIPTORS[c['cell']]}?") for c in region_calls))
        self.assertFalse((out / "crops").exists())

        gt = ev.load_yolo(self.root / "images", self.root / "labels")
        report = ev.evaluate(gt, results, dataset="toy", out_dir=out / "evaluation")
        presence = {r["condition"]: r for r in report["presence"]}
        whole = {r["condition"]: r for r in report["whole_image"]}
        self.assertEqual((presence["carious_lesion"]["FP"], whole["carious_lesion"]["FP"]), (0, 1))
        self.assertEqual((presence["endodontic_treatment"]["FP"], whole["endodontic_treatment"]["TN"]), (1, 2))
        self.assertEqual((presence["dental_implant"]["unparseable"], whole["dental_implant"]["unparseable"]), (0, 1))
        self.assertEqual(report["summary"]["whole_image"]["FP"], 2)  # caries and the bridge
        self.assertTrue((out / "evaluation" / "whole_image.csv").is_file())
        # Presence per cell reads each region's own answer: the implant's six No regions count although the
        # whole image was unparseable, and only the one unparseable residual-root region is excluded.
        rp = {(r["condition"], r["region"]): r for r in report["region_presence"]}
        self.assertEqual((rp[("dental_filling", "upper-left")]["TP"], rp[("endodontic_treatment", "upper-right")]["FP"]), (1, 1))
        self.assertEqual((rp[("dental_implant", "upper-left")]["TN"], rp[("dental_implant", "upper-left")]["images"]), (2, 2))
        self.assertEqual((rp[("root_fragment", "upper-anterior")]["unparseable"], rp[("root_fragment", "upper-left")]["TN"]), (1, 2))
        self.assertEqual({k: report["summary"]["region_presence"][k] for k in ("TP", "FP", "TN", "FN", "unparseable")},
                         {"TP": 2, "FP": 1, "TN": 104, "FN": 0, "unparseable": 1})


class PredictedCellsTests(unittest.TestCase):
    def test_rationale_and_fallback(self):
        finding = {"asked": True, "tasks": ["fillings"], "presence": "yes", "regions": ["upper-left"]}
        result = {"location_level": "rationale", "findings": {"dental_filling": finding}, "calls": []}
        self.assertEqual(ev.predicted_cells(result, "dental_filling"), {c: c == "upper-left" for c in dp.CELLS})
        result["findings"]["dental_filling"] = {**finding, "presence": "no", "regions": None}
        self.assertEqual(set(ev.predicted_cells(result, "dental_filling").values()), {False})
        result["findings"]["dental_filling"] = {**finding, "presence": None, "regions": None}
        self.assertIsNone(ev.predicted_cells(result, "dental_filling"))
        result["findings"]["dental_filling"] = {**finding, "asked": False, "tasks": []}
        self.assertIsNone(ev.predicted_cells(result, "dental_filling"))
        self.assertIsNone(ev.predicted_cells({**result, "location_level": "none", "findings": {"dental_filling": finding}},
                                             "dental_filling"))

    def test_region_answers_merge_the_tasks_cell_by_cell(self):
        def call(task, cell, text):
            return {"stage": "region", "task": task, "cell": cell, "text": text, "parse_recovery": {"value": dp.extract_answer(text)}}

        calls = [call("prosthetic_crown", c, "Yes" if c == "upper-anterior" else "No") for c in dp.CELLS]
        calls += [call("prosthetic_bridge", c, "Yes and no." if c == "lower-left" else "No") for c in dp.CELLS]
        finding = {"asked": True, "tasks": ["prosthetic_crown", "prosthetic_bridge"], "presence": "yes", "regions": ["upper-anterior"]}
        result = {"location_level": "regions", "findings": {"prosthetic_restoration": finding}, "calls": calls}
        expected = {c: True if c == "upper-anterior" else None if c == "lower-left" else False for c in dp.CELLS}
        self.assertEqual(ev.predicted_cells(result, "prosthetic_restoration"), expected)
        # A result saved without its calls falls back to the finding's cell set.
        self.assertEqual(ev.predicted_cells({**result, "calls": []}, "prosthetic_restoration"),
                         {c: c == "upper-anterior" for c in dp.CELLS})


class GeometryAndDentexTests(unittest.TestCase):
    def test_box_regions_follow_cell_windows(self):
        self.assertEqual(ev.box_regions({"xc": 0.2, "yc": 0.2, "w": 0.1, "h": 0.1}), {"upper-left"})
        self.assertEqual(ev.box_regions({"xc": 0.8, "yc": 0.8, "w": 0.1, "h": 0.1}), {"lower-right"})
        self.assertEqual(ev.box_regions({"xc": 0.5, "yc": 0.2, "w": 0.1, "h": 0.1}), {"upper-anterior"})
        canine = {"xc": 0.4, "yc": 0.2, "w": 0.1, "h": 0.1}  # on the canine line: whole in both windows
        self.assertEqual(ev.box_regions(canine), {"upper-left", "upper-anterior"})
        self.assertTrue(ev.straddling(canine))
        occlusal = {"xc": 0.2, "yc": 0.5, "w": 0.05, "h": 0.05}
        self.assertEqual(ev.box_regions(occlusal), {"upper-left", "lower-left"})
        self.assertEqual(ev.gt_regions([canine, occlusal]), {"upper-left", "upper-anterior", "lower-left"})
        self.assertEqual(ev.gt_regions([]), set())

    def test_fdi_mapping_follows_table_s6(self):
        self.assertEqual(ev.fdi_cell(1, 6), "upper-left")   # patient's upper right = DentVLM's "left"
        self.assertEqual(ev.fdi_cell(2, 1), "upper-anterior")
        self.assertEqual(ev.fdi_cell(3, 7), "lower-right")
        self.assertEqual(ev.fdi_cell(4, 4), "lower-left")
        self.assertEqual(ev.fdi_cell(8, 5), "lower-left")   # primary dentition
        self.assertEqual(ev.fdi_cell(1, 6, left_is_image_left=False), "upper-right")
        # FDI wins over geometry when present.
        self.assertEqual(ev.box_regions({"xc": 0.8, "yc": 0.8, "w": 0.1, "h": 0.1, "fdi": (1, 6)}), {"upper-left"})

    def test_load_dentex(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = {
                "images": [{"id": 1, "file_name": "a.png", "width": 1000, "height": 500}],
                "categories_1": [{"id": 0, "name": "1"}, {"id": 1, "name": "2"}, {"id": 2, "name": "3"}, {"id": 3, "name": "4"}],
                "categories_2": [{"id": i, "name": str(i + 1)} for i in range(8)],
                "categories_3": [{"id": 0, "name": "Impacted"}, {"id": 1, "name": "Caries"},
                                 {"id": 2, "name": "Periapical Lesion"}, {"id": 3, "name": "Deep Caries"}],
                "annotations": [
                    {"image_id": 1, "category_id_1": 0, "category_id_2": 5, "category_id_3": 1, "bbox": [100, 50, 50, 50]},
                    {"image_id": 1, "category_id_1": 2, "category_id_2": 0, "category_id_3": 3, "bbox": [600, 300, 50, 50]},
                ],
            }
            path = Path(tmp) / "ann.json"
            path.write_text(json.dumps(payload))
            gt = ev.load_dentex(tmp, path)
            self.assertEqual(set(gt), {"a"})
            self.assertEqual(gt["a"]["annotated"], {"carious_lesion", "periapical_lesion", "impacted_tooth"})
            boxes = gt["a"]["boxes"]
            self.assertEqual([b["condition"] for b in boxes], ["carious_lesion", "carious_lesion"])
            self.assertAlmostEqual(boxes[0]["xc"], 0.125)
            self.assertEqual([b["fdi"] for b in boxes], [(1, 6), (3, 1)])
            self.assertEqual([ev.box_regions(b) for b in boxes], [{"upper-left"}, {"lower-anterior"}])


if __name__ == "__main__":
    unittest.main()
