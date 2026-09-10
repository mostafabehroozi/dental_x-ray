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

    def test_count_forms(self):
        cases = {
            "The panoramic X-ray demonstrates 10 teeth with visible dental fillings.": 10,
            "Three teeth show root canal treatment.": 3,
            "There are 2 teeth (#16, #26) with fillings.": 2,
            "A total of three teeth have fillings, on 16, 26 and 36.": 3,
            "Only tooth 36 shows a periapical lesion.": None,
            "No teeth have fillings.": 0,
            "It is hard to say.": None,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(dp.extract_count(text), expected)

    def test_cell_descriptions(self):
        self.assertEqual(dp.describe_cell("upper-left"), "patient's upper right posterior (image left)")
        self.assertEqual(dp.describe_cell("lower-right"), "patient's lower left posterior (image right)")
        self.assertEqual(dp.describe_cell("upper-anterior"), "upper anterior")
        self.assertEqual(dp.describe_cell("upper-left", left_is_image_left=False),
                         "patient's upper left posterior (image right)")
        self.assertEqual(dp.cell_windows(False)["upper-left"], dp.cell_windows(True)["upper-right"])


def _blank_image(path: Path, size=(560, 280)) -> None:
    """A test radiograph whose regions differ, so crops are distinguishable bytes."""
    from PIL import Image, ImageDraw

    image = Image.new("L", size, color=128)
    draw = ImageDraw.Draw(image)
    for index, (left, top, right, bottom) in enumerate(dp.CELL_WINDOWS.values()):
        x, y = int(left * size[0]) + 5 + index * 3, int(top * size[1]) + 5
        draw.rectangle((x, y, x + 12, y + 12), fill=20 + 30 * index)
    image.save(path)


class FakeRunner:
    """Answers from a script keyed by (stage, task, cell); crops are recognised by bytes."""

    def __init__(self, script: dict, cell_of: dict | None = None, scripted_image: str = "img1"):
        self.script, self.cell_of, self.log = script, cell_of or {}, []
        self.scripted_image = scripted_image  # other images always answer No / 0

    def settings(self):
        return {"model": "fake"}

    def ask(self, image, question):
        task = next((t for t in list(dp.TASKS) + list(dp.UNTRAINED_LABELS) if question in dp.questions_for(t)), None)
        stage = "presence"
        if task is None:
            stage = "count"
            task = next(c for c, q in dp.COUNT_QUESTIONS.items() if q == question)
        cell = self.cell_of.get(image) if isinstance(image, bytes) else None
        if cell:
            stage = "crop"
        key = (stage, task, cell)
        self.log.append(key)
        text = "0" if stage == "count" else "No\nNothing of the kind is seen."
        if cell or Path(image).stem == self.scripted_image:
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
        _blank_image(self.root / "images" / "img2.png")
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
        self.assertEqual(f["dental_filling"], {"asked": True, "tasks": ["fillings"], "presence": "yes",
                                               "regions": ["upper-left"], "region_count": 1, "count": None})
        self.assertEqual(f["impacted_tooth"]["regions"], ["lower-right", "lower-left"])
        self.assertEqual(f["prosthetic_restoration"]["tasks"], ["prosthetic_crown", "prosthetic_bridge"])
        self.assertEqual((f["prosthetic_restoration"]["presence"], f["prosthetic_restoration"]["regions"]),
                         ("yes", ["upper-anterior"]))
        self.assertEqual(f["carious_lesion"]["regions"], [])
        self.assertIsNone(f["dental_implant"]["presence"])
        self.assertEqual(f["surgical_device"]["asked"], False)
        self.assertEqual(f["root_fragment"], {"asked": True, "tasks": ["residual_root"], "presence": "no",
                                             "regions": None, "region_count": None, "count": None})
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
        self.assertEqual(report["counts"], [])
        regions = {r["condition"]: r for r in report["regions"]}
        self.assertEqual(regions["dental_filling"]["exact_set_match_rate"], 1.0)
        self.assertEqual((regions["impacted_tooth"]["TP"], regions["impacted_tooth"]["FP"]), (1, 1))
        self.assertNotIn("carious_lesion", {r["condition"] for r in report["regions"] if r["n_localized_cases"]})
        summary = report["summary"]
        self.assertEqual(summary["images_scored"], 2)
        self.assertEqual(summary["complete_case_rate"], 1.0)
        self.assertEqual(summary["mean_false_alarms_per_image"], 1.0)
        self.assertTrue((out / "evaluation" / "presence.csv").is_file())
        self.assertFalse((out / "evaluation" / "counts.csv").exists())
        self.assertEqual(ev.side_agreement(gt, results), {"sides_named": 3, "agree": 2, "agreement_rate": 0.6667,
                                                          "left_is_image_left": True})
        text = dp.dentist_report(results["img1"])
        self.assertIn("Dental filling; in 1 region(s): patient's upper right posterior (image left)", text)
        self.assertIn("Dental caries; region not stated", text)
        self.assertIn("Also present (no benchmark class): Residual Crown", text)
        self.assertIn("Not assessable (unparseable answer): Dental implant", text)
        self.assertIn("Not assessed by this model: Furcation involvement", text)

    def test_count_question_is_optional(self):
        script = dict(SCRIPT)
        script[("count", "dental_filling", None)] = "The image shows 2 teeth with fillings."
        runner = FakeRunner(script)
        out = dp.run_dataset(runner, self.images, self.root / "run_counts", protocol=dp.Protocol(count_question=True))
        results = dp.load_results(out)
        self.assertEqual(results["img1"]["findings"]["dental_filling"]["count"], 2)
        self.assertEqual(results["img1"]["findings"]["impacted_tooth"]["count"], 0)  # the fake model's default reply
        # 13 presence calls + one count call per positive countable finding
        # (filling, impacted tooth, caries, prosthetic restoration).
        self.assertEqual(results["img1"]["call_count"], 17)
        self.assertEqual(results["img2"]["call_count"], 13)
        gt = ev.load_yolo(self.root / "images", self.root / "labels")
        report = ev.evaluate(gt, results, dataset="toy")
        counts = {r["condition"]: r for r in report["counts"]}
        self.assertEqual(counts["dental_filling"]["exact_rate"], 1.0)
        self.assertEqual(counts["impacted_tooth"]["mae"], 1.0)

    def test_crop_location_mode(self):
        cell_of = {png: cell for cell, png in dp.make_crops(self.images["img1"]).items()}
        self.assertEqual(len(cell_of), 6)
        script = dict(SCRIPT)
        script[("crop", "fillings", "upper-left")] = "Yes\nFillings are visible."
        script[("crop", "impacted_tooth", "lower-right")] = "Yes"
        runner = FakeRunner(script, cell_of)
        out = dp.run_dataset(runner, self.images, self.root / "run_crops", protocol=dp.Protocol(location="crops"))
        results = dp.load_results(out)
        f = results["img1"]["findings"]
        self.assertEqual(f["dental_filling"]["regions"], ["upper-left"])
        self.assertEqual(f["impacted_tooth"]["regions"], ["lower-right"])
        self.assertEqual(f["prosthetic_restoration"]["regions"], [])
        # 13 presence calls + 6 crops x 5 positive tasks (fillings, impacted, caries, bridge, residual crown).
        self.assertEqual(results["img1"]["call_count"], 43)
        self.assertTrue((out / "crops" / "img1_upper-left.png").is_file())


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
