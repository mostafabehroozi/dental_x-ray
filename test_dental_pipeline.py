"""Offline tests: extraction rules, question wording, crops, the run loop with a fake model at both
levels and in both region prompts, and evaluation."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import dental_eval as ev
import dental_pipeline as dp

# The whole-image count wording as it was before the region templates existed; must not drift.
WHOLE_IMAGE_COUNTS = {
    "dental_implant": "How many dental implants are visualized in the panoramic radiograph?",
    "prosthetic_restoration": "How many teeth in the image have a dental crown or bridge?",
    "dental_filling": "How many visible teeth in the image appear to have dental fillings based on their radiopaque characteristics?",
    "endodontic_treatment": "How many teeth in the image have root canal treatment?",
    "carious_lesion": "How many teeth in the image are suspected to have caries?",
    "impacted_tooth": "How many impacted teeth are visualized in the panoramic radiograph?",
    "periapical_lesion": "How many teeth in the image show signs of a periapical lesion?",
    "root_fragment": "How many residual roots are visualized in the panoramic radiograph?",
    "root_resorption": "How many teeth in the image show root resorption?",
}


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

    def test_truncated_reply_is_unparseable(self):
        cut = {"text": "<think>looking at the molars, A. True seems", "truncated": True}
        self.assertIsNone(dp.graded(cut, dp.extract_choice))
        done = {"text": "<think>x</think><answer>B</answer> and more text", "truncated": True}
        self.assertEqual(dp.graded(done, dp.extract_choice), "B")


class QuestionTests(unittest.TestCase):
    def test_presence_is_figure_7(self):
        self.assertEqual(
            dp.presence_question("impacted_tooth"),
            "Kindly evaluate if the condition 'Impacted tooth' is present in this image.\nA. True\nB. False",
        )
        self.assertTrue(dp.presence_question("impacted_tooth", "tagged").endswith(dp.THINK_SUFFIX))
        self.assertEqual(len(dp.CONDITIONS), 14)

    def test_whole_image_counts_unchanged(self):
        self.assertEqual(dp.COUNT_QUESTIONS, WHOLE_IMAGE_COUNTS)
        for condition, wording in WHOLE_IMAGE_COUNTS.items():
            self.assertEqual(dp.count_question(condition), wording)
        self.assertTrue(set(dp.COUNT_TEMPLATES) <= set(dp.CONDITIONS))

    def test_region_wording(self):
        self.assertEqual(
            dp.presence_question("carious_lesion", region="UR"),
            "Kindly evaluate if the condition 'Dental caries' is present in the upper right quadrant of this image.\nA. True\nB. False",
        )
        self.assertEqual(
            dp.presence_question("periodontal_bone_loss", region="lower", scheme="arch"),
            "Kindly evaluate if the condition 'Periodontal disease' is present in the lower jaw of this image.\nA. True\nB. False",
        )
        self.assertEqual(dp.count_question("endodontic_treatment", region="LL"),
                         "How many teeth in the lower left quadrant have root canal treatment?")
        self.assertEqual(dp.count_question("dental_implant", region="upper", scheme="arch"),
                         "How many dental implants are visualized in the upper jaw of the panoramic radiograph?")
        self.assertEqual(dp.count_question("dental_filling", region="UL", mode="tagged"),
                         "How many visible teeth in the upper left quadrant appear to have dental fillings based on "
                         f"their radiopaque characteristics?\n\n{dp.THINK_SUFFIX}")
        self.assertEqual([dp.region_phrase(r) for r in dp.CROPS["quadrant"]],
                         ["the upper right quadrant", "the upper left quadrant", "the lower left quadrant", "the lower right quadrant"])
        # With the words read as image sides, the image-left window UR gets the "left" words.
        self.assertEqual(dp.region_phrase("UR", patient_side=False), "the upper left quadrant")
        self.assertEqual(dp.region_phrase("upper", "arch", patient_side=False), "the upper jaw")
        with self.assertRaises(ValueError):
            dp.region_phrase("UR", "arch")

    def test_protocol(self):
        default = dp.Protocol()
        self.assertEqual((default.presence_level, default.count_level, default.region_scheme, default.region_prompt),
                         ("region", "region", "quadrant", "words"))
        self.assertEqual(default.regions, ("UR", "UL", "LL", "LR"))
        self.assertEqual(dp.Protocol(region_scheme="arch", count_level="overall").regions, ("upper", "lower"))
        flat = dp.Protocol(presence_level="overall", count_level="overall")
        self.assertFalse(flat.uses_regions)
        self.assertEqual(flat.regions, ())
        for bad in ({"presence_level": "crop"}, {"count_level": "none"}, {"region_scheme": "sextant"}, {"region_prompt": "json"}):
            with self.assertRaises(ValueError):
                dp.Protocol(**bad)
        # The manifest hash follows every knob and the region words.
        hashes = {dp.run_config("plain", p, {"model": "x"})["hash"] for p in (
            default, flat, dp.Protocol(region_prompt="crop"), dp.Protocol(count_level="overall"),
            dp.Protocol(region_scheme="arch"), dp.Protocol(presence_level="overall"))}
        self.assertEqual(len(hashes), 6)
        self.assertNotEqual(dp.run_config("plain", default, {"model": "x"})["hash"],
                            dp.run_config("tagged", default, {"model": "x"})["hash"])


def _blank_image(path: Path, size=(560, 280)) -> None:
    """A test radiograph whose four quadrants differ, so crops are distinguishable bytes."""
    from PIL import Image, ImageDraw

    image = Image.new("L", size, color=128)
    draw = ImageDraw.Draw(image)
    for index, (x, y) in enumerate([(0, 0), (size[0] // 2, 0), (size[0] // 2, size[1] // 2), (0, size[1] // 2)]):
        draw.rectangle((x + 10, y + 10, x + 40, y + 40), fill=30 + 50 * index)
    image.save(path)


class FakeRunner:
    """Answers from a script keyed by (stage, condition, region).

    stage is "presence" (whole image), "region" (region presence), "count" (whole-image count) or
    "region_count". Word-based region questions are recognised by their text, crops by their bytes
    (region_of maps crop bytes to the region name). Unscripted answers are B and 0; images other
    than scripted_image always answer B / 0 on the whole image.
    """

    def __init__(self, script: dict, region_of: dict | None = None, scripted_image: str = "img1"):
        self.script, self.region_of, self.log = script, region_of or {}, []
        self.scripted_image = scripted_image
        self.lookup = {}
        for condition in dp.CONDITIONS:
            self.lookup[dp.presence_question(condition)] = ("presence", condition, None)
            if condition in dp.COUNTABLE:
                self.lookup[dp.count_question(condition)] = ("count", condition, None)
            for scheme in dp.REGION_SCHEMES:
                for region in dp.CROPS[scheme]:
                    self.lookup[dp.presence_question(condition, region=region, scheme=scheme)] = ("region", condition, region)
                    if condition in dp.COUNTABLE:
                        self.lookup[dp.count_question(condition, region=region, scheme=scheme)] = ("region_count", condition, region)

    def settings(self):
        return {"model": "fake"}

    def ask(self, image, question):
        stage, condition, region = self.lookup[question.replace(f"\n\n{dp.THINK_SUFFIX}", "")]
        if isinstance(image, bytes):  # a crop carries the whole-image question; the region is the picture
            region = self.region_of[image]
            stage = "region" if stage == "presence" else "region_count"
        key = (stage, condition, region)
        self.log.append(key)
        text = "0" if stage in ("count", "region_count") else "B"
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
        # img1: two fillings in the patient's upper-right (image left), one root canal treatment in the
        # upper-left (image right), one impacted tooth lower-left.
        (self.root / "labels" / "img1.txt").write_text(
            "2 0.20 0.25 0.05 0.05\n2 0.30 0.30 0.05 0.05\n3 0.70 0.30 0.05 0.05\n6 0.80 0.80 0.10 0.10\n")
        # img2: nothing (empty label file).
        (self.root / "labels" / "img2.txt").write_text("")
        self.images = {"img1": self.root / "images" / "img1.png", "img2": self.root / "images" / "img2.png"}

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, script, protocol, region_of=None, mode="tagged", out="run"):
        runner = FakeRunner(script, region_of)
        run_dir = dp.run_dataset(runner, self.images, self.root / out, mode=mode, protocol=protocol)
        return runner, dp.load_results(run_dir)

    def test_crops_have_expected_geometry(self):
        crops = dp.make_crops(self.root / "images" / "img1.png", "quadrant", self.root / "crops")
        from PIL import Image
        import io

        self.assertEqual(list(crops), ["UR", "UL", "LL", "LR"])
        with Image.open(io.BytesIO(crops["UR"])) as ur:
            self.assertEqual(ur.size, (round(0.55 * 560), round(0.60 * 280)))
        self.assertTrue((self.root / "crops" / "img1_LL.png").is_file())

    def test_region_presence_and_region_counts_in_words(self):
        script = {
            ("presence", "dental_filling", None): "<think>..</think><answer>A</answer>",
            ("region", "dental_filling", "UR"): "A",
            ("region_count", "dental_filling", "UR"): "<answer>The quadrant shows 2 teeth with fillings.</answer>",
            ("presence", "endodontic_treatment", None): "A",  # true, but no region answers A
            ("presence", "impacted_tooth", None): "A. True",
            ("region", "impacted_tooth", "LL"): "<answer>A</answer>",
            ("region", "impacted_tooth", "LR"): "A. True",  # false region
            ("region_count", "impacted_tooth", "LL"): "1",
            ("region_count", "impacted_tooth", "LR"): "0",
            ("presence", "carious_lesion", None): "A",  # false alarm, no region answers A
            ("presence", "surgical_device", None): "???",  # unparseable presence
        }
        runner, results = self._run(script, dp.Protocol())  # region / region / quadrant / words
        f = results["img1"]["findings"]
        self.assertEqual(f["dental_filling"], {"presence": "A", "count": 2, "regions": {"UR": "A", "UL": "B", "LL": "B", "LR": "B"},
                                               "region_counts": {"UR": 2}})
        self.assertEqual(f["impacted_tooth"]["region_counts"], {"LL": 1, "LR": 0})
        self.assertEqual(f["impacted_tooth"]["count"], 1)
        self.assertEqual(f["endodontic_treatment"], {"presence": "A", "count": None, "region_counts": {},
                                                     "regions": {"UR": "B", "UL": "B", "LL": "B", "LR": "B"}})
        self.assertIsNone(f["surgical_device"]["presence"])
        self.assertEqual(f["dental_implant"], {"presence": "B", "count": None, "regions": None, "region_counts": None})
        # 14 presence + 4 positives x 4 regions + 3 region counts (UR filling, LL and LR impacted) = 33; img2: 14.
        self.assertEqual(results["img1"]["call_count"], 33)
        self.assertEqual(results["img2"]["call_count"], 14)
        self.assertEqual(results["img1"]["region_scheme"], "quadrant")
        # Every call sent the whole image: no crop was made.
        self.assertFalse((self.root / "run" / "crops").exists())
        self.assertIn("in the upper right quadrant of this image", next(
            c["question"] for c in results["img1"]["calls"] if c["stage"] == "region" and c["region"] == "UR"))

        gt = ev.load_yolo(self.root / "images", self.root / "labels")
        report = ev.evaluate(gt, results, dataset="toy", out_dir=self.root / "run" / "evaluation")
        presence = {r["condition"]: r for r in report["presence"]}
        self.assertEqual((presence["dental_filling"]["TP"], presence["dental_filling"]["TN"]), (1, 1))
        self.assertEqual(presence["carious_lesion"]["FP"], 1)
        self.assertEqual(presence["surgical_device"]["unparseable"], 1)
        counts = {r["condition"]: r for r in report["counts"]}
        self.assertEqual((counts["dental_filling"]["exact_rate"], counts["impacted_tooth"]["mae"]), (1.0, 0.0))
        self.assertEqual((counts["endodontic_treatment"]["count_unasked"], counts["endodontic_treatment"]["count_unparseable"]), (1, 0))
        region_counts = {(r["condition"], r["region"]): r for r in report["region_counts"]}
        self.assertEqual(region_counts[("dental_filling", "UR")]["exact_rate"], 1.0)
        self.assertEqual((region_counts[("dental_filling", "UL")]["n_scored"], region_counts[("dental_filling", "UL")]["strict_mae"]), (0, 0.0))
        self.assertEqual(region_counts[("impacted_tooth", "LR")]["exact_rate"], 1.0)  # 0 predicted, 0 true
        self.assertEqual(region_counts[("endodontic_treatment", "UL")]["strict_mae"], 1.0)  # UL never counted, 1 true box
        regions = {r["condition"]: r for r in report["regions"]}
        self.assertEqual(regions["dental_filling"]["exact_set_match_rate"], 1.0)
        self.assertEqual((regions["impacted_tooth"]["TP"], regions["impacted_tooth"]["FP"]), (1, 1))
        self.assertEqual((regions["endodontic_treatment"]["FN"], regions["endodontic_treatment"]["unlocalized_rate"]), (1, 1.0))
        self.assertEqual(regions["dental_filling"]["from_counts"], 0)
        self.assertEqual(regions["dental_filling"]["pred_all_regions_rate"], 0.0)
        summary = report["summary"]
        self.assertEqual(summary["protocol"], {"presence_level": "region", "count_level": "region",
                                               "region_scheme": "quadrant", "region_prompt": "words"})
        # UR (image left) holds the fillings, LL (image right) the impacted tooth, LR does not: 2 of 3 sides agree.
        self.assertEqual(summary["side_agreement"], {"sides_named": 3, "agree": 2, "agreement_rate": 0.6667,
                                                     "quadrant_words_are_patient_side": True})
        self.assertEqual(summary["complete_case_rate"], 1.0)
        self.assertTrue((self.root / "run" / "evaluation" / "region_counts.csv").is_file())
        text = dp.dentist_report(results["img1"])
        self.assertIn("Dental filling; count 2 (UR 2); location UR", text)
        self.assertIn("Impacted tooth; count 1 (LL 1, LR 0); location LL, LR", text)
        self.assertIn("Root canal treatment; location not resolved", text)

        # Resume skips finished images and rejects a different protocol.
        before = len(runner.log)
        dp.run_dataset(runner, {"img1": self.images["img1"]}, self.root / "run", mode="tagged", protocol=dp.Protocol())
        self.assertEqual(len(runner.log), before)
        with self.assertRaises(ValueError):
            dp.run_dataset(runner, {"img1": self.images["img1"]}, self.root / "run", mode="tagged",
                           protocol=dp.Protocol(count_level="overall"))

    def test_overall_presence_with_region_counts(self):
        script = {
            ("presence", "dental_filling", None): "A",
            ("region_count", "dental_filling", "UR"): "2 teeth",
            ("presence", "impacted_tooth", None): "A",
            ("region_count", "impacted_tooth", "LL"): "one",
            ("presence", "periodontal_bone_loss", None): "A",  # presence-only finding: no counts, no regions
        }
        _, results = self._run(script, dp.Protocol(presence_level="overall", count_level="region"))
        f = results["img1"]["findings"]
        self.assertIsNone(f["dental_filling"]["regions"])
        self.assertEqual(f["dental_filling"]["region_counts"], {"UR": 2, "UL": 0, "LL": 0, "LR": 0})
        self.assertEqual((f["dental_filling"]["count"], f["impacted_tooth"]["count"]), (2, 1))
        self.assertEqual(f["periodontal_bone_loss"], {"presence": "A", "count": None, "regions": None, "region_counts": None})
        self.assertEqual(results["img1"]["call_count"], 14 + 2 * 4)

        gt = ev.load_yolo(self.root / "images", self.root / "labels")
        report = ev.evaluate(gt, results, dataset="toy")
        regions = {r["condition"]: r for r in report["regions"]}
        self.assertEqual((regions["dental_filling"]["from_counts"], regions["dental_filling"]["exact_set_match_rate"]), (1, 1.0))
        self.assertEqual((regions["impacted_tooth"]["TP"], regions["impacted_tooth"]["FP"]), (1, 0))
        self.assertEqual(regions["periodontal_bone_loss"]["n_localized_cases"], 0)
        self.assertEqual(report["summary"]["side_agreement"]["agreement_rate"], 1.0)
        self.assertIn("Dental filling; count 2 (UR 2, UL 0, LL 0, LR 0); location UR", dp.dentist_report(results["img1"]))

    def test_crop_prompt_with_whole_image_counts(self):
        """The previous behaviour: presence on quadrant crops, one whole-image count per positive."""
        region_of = {png: region for region, png in dp.make_crops(self.images["img1"], "quadrant").items()}
        script = {
            ("presence", "dental_filling", None): "<think>..</think><answer>A</answer>",
            ("count", "dental_filling", None): "<answer>The image shows 2 teeth with fillings.</answer>",
            ("region", "dental_filling", "UR"): "A",
            ("presence", "impacted_tooth", None): "A. True",
            ("count", "impacted_tooth", None): "1",
            ("region", "impacted_tooth", "LL"): "<answer>A</answer>",
            ("region", "impacted_tooth", "LR"): "A. True",
            ("presence", "carious_lesion", None): "A",
            ("count", "carious_lesion", None): "unclear",
        }
        protocol = dp.Protocol(presence_level="region", count_level="overall", region_prompt="crop")
        runner, results = self._run(script, protocol, region_of)
        f = results["img1"]["findings"]
        self.assertEqual(f["dental_filling"], {"presence": "A", "count": 2, "regions": {"UR": "A", "UL": "B", "LL": "B", "LR": "B"},
                                               "region_counts": None})
        self.assertEqual(f["carious_lesion"]["count"], None)
        # 14 presence + 4 crops x 3 positives + 3 counts = 29; img2: 14.
        self.assertEqual((results["img1"]["call_count"], results["img2"]["call_count"]), (29, 14))
        self.assertTrue((self.root / "run" / "crops" / "img1_LL.png").is_file())
        self.assertIn(("region", "impacted_tooth", "LL"), runner.log)

        gt = ev.load_yolo(self.root / "images", self.root / "labels")
        report = ev.evaluate(gt, results, dataset="toy")
        counts = {r["condition"]: r for r in report["counts"]}
        self.assertEqual((counts["dental_filling"]["exact_rate"], counts["impacted_tooth"]["mae"]), (1.0, 0.0))
        self.assertEqual(report["region_counts"], [])
        regions = {r["condition"]: r for r in report["regions"]}
        self.assertEqual((regions["impacted_tooth"]["TP"], regions["impacted_tooth"]["FP"]), (1, 1))
        self.assertEqual(report["summary"]["mean_false_alarms_per_image"], 0.5)
        self.assertIn("Dental filling; count 2; location UR", dp.dentist_report(results["img1"]))

    def test_arch_scheme_and_crop_region_counts(self):
        script = {("presence", "dental_filling", None): "A", ("region", "dental_filling", "upper"): "A",
                  ("region_count", "dental_filling", "upper"): "2"}
        _, results = self._run(script, dp.Protocol(region_scheme="arch"))
        f = results["img1"]["findings"]["dental_filling"]
        self.assertEqual((f["regions"], f["region_counts"], f["count"]), ({"upper": "A", "lower": "B"}, {"upper": 2}, 2))
        self.assertEqual(results["img1"]["call_count"], 14 + 2 + 1)
        report = ev.evaluate(ev.load_yolo(self.root / "images", self.root / "labels"), results, dataset="toy")
        regions = {r["condition"]: r for r in report["regions"]}
        self.assertEqual((regions["dental_filling"]["level"], regions["dental_filling"]["TP"]), ("arch", 1))
        self.assertNotIn("side_agreement", report["summary"])

        # Region counts by crops: the whole-image count question on each crop that answered A.
        region_of = {png: region for region, png in dp.make_crops(self.images["img1"], "arch").items()}
        _, results = self._run(script, dp.Protocol(region_scheme="arch", region_prompt="crop"), region_of, out="crop_run")
        f = results["img1"]["findings"]["dental_filling"]
        self.assertEqual((f["regions"], f["region_counts"], f["count"]), ({"upper": "A", "lower": "B"}, {"upper": 2}, 2))
        question = next(c["question"] for c in results["img1"]["calls"] if c["stage"] == "region_count")
        self.assertEqual(question.split("\n")[0], WHOLE_IMAGE_COUNTS["dental_filling"])

    def test_unresolved_mode_is_rejected_before_anything_is_written(self):
        out = self.root / "run_auto"
        with self.assertRaises(ValueError):
            dp.run_dataset(FakeRunner({}), self.images, out, mode="auto", protocol=dp.Protocol())
        self.assertFalse((out / "manifest.json").exists())

    def test_results_saved_before_the_levels_still_evaluate(self):
        blank = {c: {"presence": "B", "count": None, "regions": None} for c in dp.CONDITIONS}
        img1 = {c: dict(v) for c, v in blank.items()}
        img1["dental_filling"] = {"presence": "A", "count": 2, "regions": {"UR": "A", "UL": "B", "LL": "B", "LR": "B"}}
        results = {"img1": {"location_level": "quadrant", "findings": img1, "call_count": 19},
                   "img2": {"location_level": "quadrant", "findings": blank, "call_count": 14}}
        report = ev.evaluate(ev.load_yolo(self.root / "images", self.root / "labels"), results, dataset="old")
        self.assertEqual(report["summary"]["protocol"]["region_prompt"], "crop")
        self.assertEqual({r["condition"]: r["TP"] for r in report["regions"]}["dental_filling"], 1)
        self.assertEqual(report["region_counts"], [])


class GeometryAndDentexTests(unittest.TestCase):
    def test_box_regions_follow_crop_windows(self):
        self.assertEqual(ev.box_regions({"xc": 0.2, "yc": 0.2, "w": 0.1, "h": 0.1}, "quadrant"), {"UR"})
        self.assertEqual(ev.box_regions({"xc": 0.8, "yc": 0.8, "w": 0.1, "h": 0.1}, "quadrant"), {"LL"})
        midline = {"xc": 0.5, "yc": 0.2, "w": 0.2, "h": 0.1}
        self.assertEqual(ev.box_regions(midline, "quadrant"), {"UR", "UL"})
        self.assertTrue(ev.straddling(midline, "quadrant"))
        self.assertEqual(ev.box_primary_region(midline, "quadrant"), "UR")  # counted once, first window in order
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
