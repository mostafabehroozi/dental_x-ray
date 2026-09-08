"""Offline tests: extraction rules, crops, the run loop with a fake model, and evaluation."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import dental_eval as ev
import dental_pipeline as dp


class ExtractionTests(unittest.TestCase):
    def test_choice_forms(self):
        cases = {
            "<think>looking...</think><answer>A</answer>": "A",
            "<think>x</think>\n<answer>B. False</answer>": "B",
            "A. True": "A",
            "(B)": "B",
            "**A**": "A",
            "The answer is B.": "B",
            "Answer: A": "A",
            "A\nThe radiograph shows fillings.": "A",
            "True": "A",
            "No, there is no impacted tooth.": "B",
            "Yes, it is present.": "A",
            "A distinct lesion is visible, so the answer is true.": "A",
            "A. True\nB. False\n\nThe correct option is B.": "B",
            "A. True\nB. False\n\nA. True": "A",
            "The answer is A because a radiopaque crown is visible.": "A",
            "B is correct.": "B",
            "<answer>false</answer>": "B",
            "<think>I see a lesion, so the answer is A. True. But wait": None,
            "I cannot tell.": None,
            "": None,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(dp.extract_choice(text), expected)

    def test_last_answer_block_wins(self):
        text = "<answer>A</answer> wait, on review <think>...</think><answer>B</answer>"
        self.assertEqual(dp.extract_choice(text), "B")

    def test_count_forms(self):
        cases = {
            "<answer>In summary, the panoramic X-ray demonstrates 10 teeth with visible dental fillings.</answer>": 10,
            "<think>maybe 9... no, 10</think><answer>10</answer>": 10,
            "Three teeth show root canal treatment.": 3,
            "There are 2 teeth (#16, #26) with fillings.": 2,
            "Teeth 16 and 26 have fillings, so 2 teeth.": 2,
            "<answer>Two teeth: 16 and 26</answer>": 2,
            "<answer>Count: 3. Teeth: 16, 26, 36.</answer>": 3,
            "<answer>A total of three teeth have fillings, on 16, 26 and 36.</answer>": 3,
            "<answer>Two teeth have fillings, no others.</answer>": 2,
            "<answer>Only tooth 36 shows a periapical lesion.</answer>": None,
            "<answer>12. No others.</answer>": 12,
            "No teeth have fillings.": 0,
            "None visible.": 0,
            "It is hard to say.": None,
            "<think>counting 3, maybe 4": None,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(dp.extract_count(text), expected)

    def test_prompts_are_paper_shaped(self):
        self.assertEqual(
            dp.presence_question("impacted_tooth"),
            "Kindly evaluate if the condition 'Impacted tooth' is present in this image.\nA. True\nB. False",
        )
        self.assertTrue(dp.presence_question("impacted_tooth", "tagged").endswith(dp.THINK_SUFFIX))
        self.assertIn("radiopaque characteristics", dp.count_question("dental_filling"))
        self.assertTrue(set(dp.COUNT_QUESTIONS) <= set(dp.CONDITIONS))
        self.assertEqual(len(dp.CONDITIONS), 14)

    def test_truncated_reply_is_unparseable(self):
        cut = {"text": "<think>looking at the molars, A. True seems", "truncated": True}
        self.assertIsNone(dp.graded(cut, dp.extract_choice))
        done = {"text": "<think>x</think><answer>B</answer> and more text", "truncated": True}
        self.assertEqual(dp.graded(done, dp.extract_choice), "B")


def _blank_image(path: Path, size=(560, 280)) -> None:
    """A test radiograph whose four quadrants differ, so crops are distinguishable bytes."""
    from PIL import Image, ImageDraw

    image = Image.new("L", size, color=128)
    draw = ImageDraw.Draw(image)
    for index, (x, y) in enumerate([(0, 0), (size[0] // 2, 0), (size[0] // 2, size[1] // 2), (0, size[1] // 2)]):
        draw.rectangle((x + 10, y + 10, x + 40, y + 40), fill=30 + 50 * index)
    image.save(path)


class FakeRunner:
    """Answers from a script keyed by (stage, condition, region); crops are recognised by bytes."""

    def __init__(self, script: dict, region_of: dict | None = None, scripted_image: str = "img1"):
        self.script, self.region_of, self.log = script, region_of or {}, []
        self.scripted_image = scripted_image  # other images always answer B / 0

    def settings(self):
        return {"model": "fake"}

    def ask(self, image, question):
        first = question.split("\n")[0]
        if first in dp.COUNT_QUESTIONS.values():
            stage = "count"
            condition = next(c for c, q in dp.COUNT_QUESTIONS.items() if q == first)
        else:
            stage = "presence"
            condition = next(c for c in dp.CONDITIONS if f"'{dp.LABELS[c]}'" in first)
        region = self.region_of.get(image) if isinstance(image, bytes) else None
        if region:
            stage = "region"
        key = (stage, condition, region)
        self.log.append(key)
        text = "0" if stage == "count" else "B"
        if region or Path(image).stem == self.scripted_image:
            text = self.script.get(key, text)
        return {"text": text, "finish_reason": "stop", "truncated": False,
                "prompt_tokens": 100, "completion_tokens": 5, "latency_seconds": 0.0}


class RunAndEvaluateTests(unittest.TestCase):
    def setUp(self):
        try:
            import PIL  # noqa: F401
        except ImportError:
            self.skipTest("Pillow not installed")
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "images").mkdir()
        (self.root / "labels").mkdir()
        _blank_image(self.root / "images" / "img1.png")
        _blank_image(self.root / "images" / "img2.png")
        # img1: two fillings in the patient's upper-right (image left), one impacted tooth lower-left.
        (self.root / "labels" / "img1.txt").write_text(
            "2 0.20 0.25 0.05 0.05\n2 0.30 0.30 0.05 0.05\n6 0.80 0.80 0.10 0.10\n")
        # img2: nothing (empty label file).
        (self.root / "labels" / "img2.txt").write_text("")

    def tearDown(self):
        self.tmp.cleanup()

    def test_crops_have_expected_geometry(self):
        crops = dp.make_crops(self.root / "images" / "img1.png", "quadrant", self.root / "crops")
        from PIL import Image
        import io

        self.assertEqual(list(crops), ["UR", "UL", "LL", "LR"])
        with Image.open(io.BytesIO(crops["UR"])) as ur:
            self.assertEqual(ur.size, (round(0.55 * 560), round(0.60 * 280)))
        self.assertTrue((self.root / "crops" / "img1_LL.png").is_file())

    def test_run_then_evaluate(self):
        script = {
            ("presence", "dental_filling", None): "<think>..</think><answer>A</answer>",
            ("count", "dental_filling", None): "<answer>The image shows 2 teeth with fillings.</answer>",
            ("presence", "impacted_tooth", None): "A. True",
            ("count", "impacted_tooth", None): "1",
            ("presence", "carious_lesion", None): "A",  # false alarm, count unparseable
            ("count", "carious_lesion", None): "unclear",
            ("presence", "surgical_device", None): "???",  # unparseable presence
        }
        region_of = {png: region for region, png in dp.make_crops(self.root / "images" / "img1.png", "quadrant").items()}
        script.update({
            ("region", "dental_filling", "UR"): "A",
            ("region", "impacted_tooth", "LL"): "<answer>A</answer>",
            ("region", "impacted_tooth", "LR"): "A. True",
        })
        runner = FakeRunner(script, region_of)
        out = dp.run_dataset(runner, {"img1": self.root / "images" / "img1.png", "img2": self.root / "images" / "img2.png"},
                             self.root / "run", mode="tagged", location="quadrant")

        results = dp.load_results(out)
        self.assertEqual(set(results), {"img1", "img2"})
        f = results["img1"]["findings"]
        self.assertEqual(f["dental_filling"], {"presence": "A", "count": 2, "regions": {"UR": "A", "UL": "B", "LL": "B", "LR": "B"}})
        self.assertEqual([r for r, a in f["impacted_tooth"]["regions"].items() if a == "A"], ["LL", "LR"])
        self.assertEqual(f["carious_lesion"]["count"], None)
        self.assertEqual(set(f["carious_lesion"]["regions"].values()), {"B"})
        self.assertIsNone(f["surgical_device"]["presence"])
        self.assertEqual(f["dental_implant"], {"presence": "B", "count": None, "regions": None})
        # 14 presence + 3 counts + 4 crops x 3 positives = 29 calls for img1; 14 for img2.
        self.assertEqual(results["img1"]["call_count"], 29)
        self.assertEqual(results["img2"]["call_count"], 14)
        self.assertTrue((out / "manifest.json").is_file())

        # Resume skips finished images and rejects a different configuration.
        before = len(runner.log)
        dp.run_dataset(runner, {"img1": self.root / "images" / "img1.png"}, out, mode="tagged", location="quadrant")
        self.assertEqual(len(runner.log), before)
        with self.assertRaises(ValueError):
            dp.run_dataset(runner, {"img1": self.root / "images" / "img1.png"}, out, mode="plain", location="quadrant")

        gt = ev.load_yolo(self.root / "images", self.root / "labels")
        report = ev.evaluate(gt, results, dataset="toy", out_dir=out / "evaluation")
        presence = {r["condition"]: r for r in report["presence"]}
        self.assertEqual((presence["dental_filling"]["TP"], presence["dental_filling"]["FN"]), (1, 0))
        self.assertEqual(presence["dental_filling"]["TN"], 1)
        self.assertEqual(presence["carious_lesion"]["FP"], 1)
        self.assertEqual(presence["surgical_device"]["unparseable"], 1)
        self.assertEqual(presence["surgical_device"]["TN"], 1)
        counts = {r["condition"]: r for r in report["counts"]}
        self.assertEqual(counts["dental_filling"]["exact_rate"], 1.0)
        self.assertEqual(counts["impacted_tooth"]["mae"], 0.0)
        regions = {r["condition"]: r for r in report["regions"]}
        self.assertEqual(regions["dental_filling"]["exact_set_match_rate"], 1.0)
        # The impacted tooth box (0.8, 0.8) lies only in the LL crop window; the LR answer is a false positive.
        self.assertEqual((regions["impacted_tooth"]["TP"], regions["impacted_tooth"]["FP"]), (1, 1))
        self.assertEqual(regions["carious_lesion"]["n_localized_cases"] if "carious_lesion" in regions else 0, 0)
        summary = report["summary"]
        self.assertEqual(summary["images_scored"], 2)
        self.assertEqual(summary["complete_case_rate"], 1.0)
        self.assertEqual(summary["mean_false_alarms_per_image"], 0.5)
        self.assertTrue((out / "evaluation" / "presence.csv").is_file())
        self.assertIn("Dental filling; count 2; location UR", dp.dentist_report(results["img1"]))


class GeometryAndDentexTests(unittest.TestCase):
    def test_box_regions_follow_crop_windows(self):
        self.assertEqual(ev.box_regions({"xc": 0.2, "yc": 0.2, "w": 0.1, "h": 0.1}, "quadrant"), {"UR"})
        self.assertEqual(ev.box_regions({"xc": 0.8, "yc": 0.8, "w": 0.1, "h": 0.1}, "quadrant"), {"LL"})
        midline = {"xc": 0.5, "yc": 0.2, "w": 0.2, "h": 0.1}
        self.assertEqual(ev.box_regions(midline, "quadrant"), {"UR", "UL"})
        self.assertTrue(ev.straddling(midline, "quadrant"))
        # A crown-level filling at y=0.55 is fully inside both the upper and lower crop windows.
        occlusal = {"xc": 0.2, "yc": 0.55, "w": 0.05, "h": 0.05}
        self.assertEqual(ev.box_regions(occlusal, "quadrant"), {"UR", "LR"})
        self.assertEqual(ev.gt_regions([midline, occlusal], "arch"), {"upper", "lower"})
        self.assertEqual(ev.gt_regions([], "arch"), set())

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
            self.assertEqual([ev.box_regions(b, "quadrant") for b in boxes], [{"UR"}, {"LL"}])


if __name__ == "__main__":
    unittest.main()
