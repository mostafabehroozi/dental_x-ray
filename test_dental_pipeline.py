"""Offline tests: extraction rules, question wording, crops, the run loop with a fake model at both
levels, in both region prompts (every region asked about every finding) and in both question forms
(separate presence and count questions, or the combined presence-and-count question), and evaluation."""
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

    def test_combined_forms(self):
        cases = {
            "Answer: A. True\nCount: 3": ("A", 3),
            "Answer: B. False\nCount: 0": ("B", 0),
            "**Answer:** A. True\n**Count:** 2": ("A", 2),
            "Answer: A. True, Count: 4": ("A", 4),
            "Answer: A. True; Count: 2 teeth": ("A", 2),
            "Answer\uff1aB. False\nCount\uff1a0": ("B", 0),
            "<think>hmm</think>\nAnswer: A. True\nCount: 1": ("A", 1),
            "Two molars carry amalgam. Answer: A. True\nCount: two": ("A", 2),
            "I see fillings on teeth 16 and 26.\nAnswer: A. True\nCount: 2\n\nNote: tooth 36 has a crown.": ("A", 2),
            "A. True\nB. False\n\nAnswer: B. False\nCount: 0": ("B", 0),
            "A. True\n3": ("A", 3),
            "B. False. No implants are visible.": ("B", 0),
            "B. False": ("B", None),
            "Answer: A. True\nCount: N/A": ("A", None),
            "The count is 3.": (None, 3),
            "": (None, None),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(dp.extract_combined(text), expected)
        # The letter stands; a count that contradicts it is unparseable; without a letter nothing is kept.
        self.assertEqual(dp.consistent_pair("A", 0), ("A", None))
        self.assertEqual(dp.consistent_pair("B", 3), ("B", None))
        self.assertEqual(dp.consistent_pair(None, 3), (None, None))
        self.assertEqual(dp.consistent_pair("B", None), ("B", None))
        self.assertEqual(dp.consistent_pair("A", 2), ("A", 2))
        cut = {"text": "<think>counting the molars, Answer: A. True Count: 2", "truncated": True}
        self.assertEqual(dp.graded_pair(cut), (None, None))
        self.assertEqual(dp.graded_pair({"text": "Answer: A. True\nCount: 2", "truncated": False}), ("A", 2))
        self.assertEqual(dp.graded_pair({"text": "Answer: B. False\nCount: 5", "truncated": False}), ("B", None))


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

    def test_combined_wording(self):
        whole = dp.combined_question("dental_filling")
        # The Figure 7 question and the whole-image count question are inside, verbatim.
        self.assertIn(dp.presence_question("dental_filling"), whole)
        self.assertIn(WHOLE_IMAGE_COUNTS["dental_filling"], whole)
        self.assertIn("Finding under review: Dental filling. " + dp.DEFINITIONS["dental_filling"][0], whole)
        self.assertIn(dp.DEFINITIONS["dental_filling"][1] + " If the answer is B, the count is 0.", whole)
        self.assertIn("\nScope: the whole radiograph.\n", whole)
        self.assertTrue(whole.startswith(dp.COMBINED_CONTEXT))
        self.assertTrue(whole.endswith('Answer: <exactly "A. True" or "B. False">\n'
                                       'Count: <a whole number written in digits; 0 when the answer is B>'))
        words = dp.combined_question("endodontic_treatment", region="LL")
        self.assertIn(dp.presence_question("endodontic_treatment", region="LL"), words)
        self.assertIn("also answer: How many teeth in the lower left quadrant have root canal treatment? ", words)
        self.assertIn("Scope: only the patient's lower left quadrant (FDI quadrant 3): the lower teeth on the RIGHT half of "
                      "the image as displayed, from the lower left central incisor back to the lower left third molar; "
                      "ignore every tooth outside it.", words)
        arch = dp.combined_question("dental_implant", region="upper", scheme="arch")
        self.assertIn("present in the upper jaw of this image.", arch)
        self.assertIn("How many dental implants are visualized in the upper jaw of the panoramic radiograph? ", arch)
        self.assertIn("Scope: only the upper jaw (maxilla)", arch)
        crop = dp.combined_question("dental_filling", crop=True)
        self.assertIn(dp.presence_question("dental_filling"), crop)  # the whole-image wording goes with the crop
        self.assertIn(WHOLE_IMAGE_COUNTS["dental_filling"], crop)
        self.assertIn("Scope: this image, which is a cropped region of the radiograph", crop)
        self.assertTrue(dp.combined_question("dental_filling", mode="tagged").endswith(dp.THINK_SUFFIX))
        # Quadrant words are the patient's whatever the flag says: the scope note already names the image half.
        dp.QUADRANT_WORDS_ARE_PATIENT_SIDE = False
        try:
            self.assertIn("present in the upper right quadrant of this image", dp.combined_question("dental_filling", region="UR"))
            self.assertIn("in the upper right quadrant appear to have", dp.combined_question("dental_filling", region="UR"))
        finally:
            dp.QUADRANT_WORDS_ARE_PATIENT_SIDE = True
        for bad in ({"condition": "periodontal_bone_loss"}, {"condition": "dental_filling", "region": "UR", "crop": True}):
            with self.assertRaises(ValueError):
                dp.combined_question(**bad)
        self.assertEqual(set(dp.DEFINITIONS), set(dp.COUNTABLE))
        self.assertEqual(set(dp.SCOPE_NOTES), set(dp.CROPS["quadrant"]) | set(dp.CROPS["arch"]))

    def test_protocol(self):
        default = dp.Protocol()
        self.assertEqual((default.presence_level, default.count_level, default.region_scheme, default.region_prompt,
                          default.question_form), ("region", "region", "quadrant", "words", "separate"))
        self.assertEqual(default.regions, ("UR", "UL", "LL", "LR"))
        self.assertEqual(dp.Protocol(region_scheme="arch", count_level="overall").regions, ("upper", "lower"))
        flat = dp.Protocol(presence_level="overall", count_level="overall")
        self.assertFalse(flat.uses_regions)
        self.assertEqual(flat.regions, ())
        for bad in ({"presence_level": "crop"}, {"count_level": "none"}, {"region_scheme": "sextant"}, {"region_prompt": "json"},
                    {"question_form": "joint"}):
            with self.assertRaises(ValueError):
                dp.Protocol(**bad)
        # The manifest hash follows every knob and the region words.
        hashes = {dp.run_config("plain", p, {"model": "x"})["hash"] for p in (
            default, flat, dp.Protocol(region_prompt="crop"), dp.Protocol(count_level="overall"),
            dp.Protocol(region_scheme="arch"), dp.Protocol(presence_level="overall"))}
        self.assertEqual(len(hashes), 6)
        self.assertNotEqual(dp.run_config("plain", default, {"model": "x"})["hash"],
                            dp.run_config("tagged", default, {"model": "x"})["hash"])
        # The combined form is part of the hash and its wording is in the manifest.
        combined = dp.run_config("plain", dp.Protocol(question_form="combined"), {"model": "x"})
        self.assertNotEqual(combined["hash"], dp.run_config("plain", default, {"model": "x"})["hash"])
        self.assertEqual(combined["protocol"]["question_form"], "combined")
        self.assertEqual((combined["combined"]["question"], combined["combined"]["definitions"]), (dp.COMBINED_QUESTION, dp.DEFINITIONS))
        self.assertEqual(list(combined["combined"]["scope_notes"]), ["UR", "UL", "LL", "LR"])
        self.assertIsNone(dp.run_config("plain", dp.Protocol(question_form="combined", region_prompt="crop"), {"model": "x"})["combined"]["scope_notes"])
        self.assertIsNone(dp.run_config("plain", default, {"model": "x"})["combined"])


def _blank_image(path: Path, size=(560, 280), shade: int = 128) -> None:
    """A test radiograph whose four quadrants differ, so crops are distinguishable bytes."""
    from PIL import Image, ImageDraw

    image = Image.new("L", size, color=shade)
    draw = ImageDraw.Draw(image)
    for index, (x, y) in enumerate([(0, 0), (size[0] // 2, 0), (size[0] // 2, size[1] // 2), (0, size[1] // 2)]):
        draw.rectangle((x + 10, y + 10, x + 40, y + 40), fill=30 + 50 * index)
    image.save(path)


class FakeRunner:
    """Answers from a script keyed by (stage, condition, region).

    stage is "presence" (whole image), "region" (region presence), "count" (whole-image count) or
    "region_count". A combined presence-and-count question uses the key of the presence question of
    its scope ("presence" or "region", whatever stage the pipeline records for it) and a scripted text
    such as "Answer: A. True\\nCount: 2". Word-based region questions are recognised by their text, crops
    by their bytes (region_of maps the scripted image's crop bytes to the region name). Unscripted
    answers are B and 0 ("Answer: B. False / Count: 0" for a combined question); images other than
    scripted_image always answer B / 0.
    """

    def __init__(self, script: dict, region_of: dict | None = None, scripted_image: str = "img1"):
        self.script, self.region_of, self.log = script, region_of or {}, []
        self.scripted_image = scripted_image
        self.lookup = {}
        for condition in dp.CONDITIONS:
            self.lookup[dp.presence_question(condition)] = ("presence", condition, None, False)
            if condition in dp.COUNTABLE:
                self.lookup[dp.count_question(condition)] = ("count", condition, None, False)
                self.lookup[dp.combined_question(condition)] = ("presence", condition, None, True)
                self.lookup[dp.combined_question(condition, crop=True)] = ("presence", condition, None, True)
            for scheme in dp.REGION_SCHEMES:
                for region in dp.CROPS[scheme]:
                    self.lookup[dp.presence_question(condition, region=region, scheme=scheme)] = ("region", condition, region, False)
                    if condition in dp.COUNTABLE:
                        self.lookup[dp.count_question(condition, region=region, scheme=scheme)] = ("region_count", condition, region, False)
                        self.lookup[dp.combined_question(condition, region=region, scheme=scheme)] = ("region", condition, region, True)

    def settings(self):
        return {"model": "fake"}

    def ask(self, image, question):
        stage, condition, region, combined = self.lookup[question.replace(f"\n\n{dp.THINK_SUFFIX}", "")]
        if isinstance(image, bytes):  # a crop carries the whole-image question; the region is the picture
            region = self.region_of.get(image)
            stage = "region" if stage == "presence" else "region_count"
            scripted = region is not None
        else:
            scripted = Path(image).stem == self.scripted_image
        key = (stage, condition, region)
        self.log.append(key)
        text = "Answer: B. False\nCount: 0" if combined else "0" if stage in ("count", "region_count") else "B"
        if scripted:
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
        _blank_image(self.root / "images" / "img2.png", shade=100)  # different bytes, so its crops are not img1's
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
            # Root canal treatment is missed on the whole image (default B) and recovered in UL.
            ("region", "endodontic_treatment", "UL"): "A",
            ("region_count", "endodontic_treatment", "UL"): "1",
            ("presence", "impacted_tooth", None): "A. True",
            ("region", "impacted_tooth", "LL"): "<answer>A</answer>",
            ("region", "impacted_tooth", "LR"): "A. True",  # false region
            ("region_count", "impacted_tooth", "LL"): "1",
            ("region_count", "impacted_tooth", "LR"): "0",
            ("presence", "carious_lesion", None): "A",  # whole-image false alarm; no region answers A
            ("presence", "surgical_device", None): "???",  # unparseable on the whole image; every region answers B
            ("region", "dental_implant", "UR"): "???",  # one unparseable region and no A: unparseable finding
        }
        runner, results = self._run(script, dp.Protocol())  # region / region / quadrant / words
        f = results["img1"]["findings"]
        self.assertEqual(f["dental_filling"], {"presence": "A", "whole_image": "A", "count": 2,
                                               "regions": {"UR": "A", "UL": "B", "LL": "B", "LR": "B"}, "region_counts": {"UR": 2}})
        self.assertEqual(f["endodontic_treatment"], {"presence": "A", "whole_image": "B", "count": 1,
                                                     "regions": {"UR": "B", "UL": "A", "LL": "B", "LR": "B"}, "region_counts": {"UL": 1}})
        self.assertEqual(f["impacted_tooth"]["region_counts"], {"LL": 1, "LR": 0})
        self.assertEqual(f["impacted_tooth"]["count"], 1)
        self.assertEqual(f["carious_lesion"], {"presence": "B", "whole_image": "A", "count": None,
                                               "regions": {"UR": "B", "UL": "B", "LL": "B", "LR": "B"}, "region_counts": {}})
        self.assertEqual((f["surgical_device"]["presence"], f["surgical_device"]["whole_image"]), ("B", None))
        self.assertEqual((f["dental_implant"]["presence"], f["dental_implant"]["regions"]["UR"]), (None, None))
        self.assertEqual(f["periodontal_bone_loss"], {"presence": "B", "whole_image": "B", "count": None,
                                                      "regions": {"UR": "B", "UL": "B", "LL": "B", "LR": "B"}, "region_counts": None})
        # 14 whole image + 14 findings x 4 regions + 4 region counts (UR filling, UL root canal, LL and LR
        # impacted) = 74; img2: 70. Every region is asked about every finding, whatever the whole image said.
        self.assertEqual(results["img1"]["call_count"], 74)
        self.assertEqual(results["img2"]["call_count"], 70)
        self.assertEqual(results["img1"]["region_scheme"], "quadrant")
        # Every call sent the whole image: no crop was made.
        self.assertFalse((self.root / "run" / "crops").exists())
        self.assertIn("in the upper right quadrant of this image", next(
            c["question"] for c in results["img1"]["calls"] if c["stage"] == "region" and c["region"] == "UR"))
        # The count follows its region's presence question at once (region-major order).
        stages = [(c["stage"], c["region"]) for c in results["img1"]["calls"]]
        self.assertEqual(stages.index(("region_count", "UR")), stages.index(("region", "UR")) + 3)

        gt = ev.load_yolo(self.root / "images", self.root / "labels")
        report = ev.evaluate(gt, results, dataset="toy", out_dir=self.root / "run" / "evaluation")
        presence = {r["condition"]: r for r in report["presence"]}
        self.assertEqual((presence["dental_filling"]["TP"], presence["dental_filling"]["TN"]), (1, 1))
        self.assertEqual((presence["endodontic_treatment"]["TP"], presence["endodontic_treatment"]["FN"]), (1, 0))
        self.assertEqual((presence["carious_lesion"]["FP"], presence["carious_lesion"]["TN"]), (0, 2))
        self.assertEqual((presence["surgical_device"]["unparseable"], presence["dental_implant"]["unparseable"]), (0, 1))
        # The whole-image answers are scored on their own: the root canal miss and the caries false alarm show there.
        whole = {r["condition"]: r for r in report["whole_image"]}
        self.assertEqual((whole["endodontic_treatment"]["TP"], whole["endodontic_treatment"]["FN"]), (0, 1))
        self.assertEqual((whole["carious_lesion"]["FP"], whole["surgical_device"]["unparseable"]), (1, 1))
        self.assertEqual(list(whole["dental_filling"]), list(presence["dental_filling"]))
        counts = {r["condition"]: r for r in report["counts"]}
        self.assertEqual((counts["dental_filling"]["exact_rate"], counts["impacted_tooth"]["mae"]), (1.0, 0.0))
        self.assertEqual((counts["endodontic_treatment"]["exact_rate"], counts["endodontic_treatment"]["count_unparseable"]), (1.0, 0))
        region_counts = {(r["condition"], r["region"]): r for r in report["region_counts"]}
        self.assertEqual(region_counts[("dental_filling", "UR")]["exact_rate"], 1.0)
        self.assertEqual((region_counts[("dental_filling", "UL")]["n_scored"], region_counts[("dental_filling", "UL")]["strict_mae"]), (0, 0.0))
        self.assertEqual(region_counts[("impacted_tooth", "LR")]["exact_rate"], 1.0)  # 0 predicted, 0 true
        self.assertEqual(region_counts[("endodontic_treatment", "UL")]["exact_rate"], 1.0)
        regions = {r["condition"]: r for r in report["regions"]}
        self.assertEqual(regions["dental_filling"]["exact_set_match_rate"], 1.0)
        self.assertEqual((regions["impacted_tooth"]["TP"], regions["impacted_tooth"]["FP"]), (1, 1))
        self.assertEqual((regions["endodontic_treatment"]["TP"], regions["endodontic_treatment"]["unlocalized_rate"]), (1, 0.0))
        self.assertEqual(regions["dental_filling"]["from_counts"], 0)
        self.assertEqual(regions["dental_filling"]["pred_all_regions_rate"], 0.0)
        summary = report["summary"]
        self.assertEqual(summary["protocol"], {"presence_level": "region", "count_level": "region",
                                               "region_scheme": "quadrant", "region_prompt": "words",
                                               "question_form": "separate"})
        self.assertEqual((summary["sensitivity"], summary["whole_image"]["sensitivity"]), (1.0, 0.6667))
        # UR (image left) holds the fillings, UL and LL (image right) the root canal and the impacted tooth,
        # LR does not: 3 of 4 sides agree.
        self.assertEqual(summary["side_agreement"], {"sides_named": 4, "agree": 3, "agreement_rate": 0.75,
                                                     "quadrant_words_are_patient_side": True})
        self.assertEqual(summary["complete_case_rate"], 1.0)
        for name in ("region_counts", "whole_image"):
            self.assertTrue((self.root / "run" / "evaluation" / f"{name}.csv").is_file())
        text = dp.dentist_report(results["img1"])
        self.assertIn("Dental filling; count 2 (UR 2); location UR", text)
        self.assertIn("Root canal treatment; count 1 (UL 1); location UL", text)
        self.assertIn("Impacted tooth; count 1 (LL 1, LR 0); location LL, LR", text)
        self.assertIn("Dental caries", text.split("Not seen: ")[1])

        # Resume skips finished images and rejects a different protocol.
        before = len(runner.log)
        dp.run_dataset(runner, {"img1": self.images["img1"]}, self.root / "run", mode="tagged", protocol=dp.Protocol())
        self.assertEqual(len(runner.log), before)
        with self.assertRaises(ValueError):
            dp.run_dataset(runner, {"img1": self.images["img1"]}, self.root / "run", mode="tagged",
                           protocol=dp.Protocol(count_level="overall"))

    def test_overall_presence_with_region_counts(self):
        """Whole-image presence; every countable finding is counted in every region, whatever the whole image said."""
        script = {
            ("presence", "dental_filling", None): "A",
            ("region_count", "dental_filling", "UR"): "2 teeth",
            ("presence", "impacted_tooth", None): "A",
            ("region_count", "impacted_tooth", "LL"): "one",
            ("region_count", "endodontic_treatment", "UL"): "1",  # counted although missed on the whole image
            ("presence", "periodontal_bone_loss", None): "A",  # presence-only finding: no counts, no regions
        }
        _, results = self._run(script, dp.Protocol(presence_level="overall", count_level="region"))
        f = results["img1"]["findings"]
        self.assertIsNone(f["dental_filling"]["regions"])
        self.assertEqual(f["dental_filling"]["region_counts"], {"UR": 2, "UL": 0, "LL": 0, "LR": 0})
        self.assertEqual((f["dental_filling"]["count"], f["impacted_tooth"]["count"]), (2, 1))
        self.assertEqual((f["endodontic_treatment"]["presence"], f["endodontic_treatment"]["region_counts"]["UL"]), ("B", 1))
        self.assertEqual(f["periodontal_bone_loss"], {"presence": "A", "whole_image": "A", "count": None,
                                                      "regions": None, "region_counts": None})
        # 14 whole image + 9 countable findings x 4 regions.
        self.assertEqual((results["img1"]["call_count"], results["img2"]["call_count"]), (50, 50))

        gt = ev.load_yolo(self.root / "images", self.root / "labels")
        report = ev.evaluate(gt, results, dataset="toy")
        self.assertEqual(report["whole_image"], [])  # presence is the whole-image answer: nothing separate to score
        presence = {r["condition"]: r for r in report["presence"]}
        self.assertEqual(presence["endodontic_treatment"]["FN"], 1)
        regions = {r["condition"]: r for r in report["regions"]}
        self.assertEqual((regions["dental_filling"]["from_counts"], regions["dental_filling"]["exact_set_match_rate"]), (1, 1.0))
        self.assertEqual((regions["impacted_tooth"]["TP"], regions["impacted_tooth"]["FP"]), (1, 0))
        self.assertEqual(regions["periodontal_bone_loss"]["n_localized_cases"], 0)
        self.assertEqual(report["summary"]["side_agreement"]["agreement_rate"], 1.0)
        self.assertIn("Dental filling; count 2 (UR 2, UL 0, LL 0, LR 0); location UR", dp.dentist_report(results["img1"]))

    def test_crop_prompt_with_whole_image_counts(self):
        """Presence on quadrant crops for every finding, one whole-image count per positive."""
        region_of = {png: region for region, png in dp.make_crops(self.images["img1"], "quadrant").items()}
        script = {
            ("presence", "dental_filling", None): "<think>..</think><answer>A</answer>",
            ("count", "dental_filling", None): "<answer>The image shows 2 teeth with fillings.</answer>",
            ("region", "dental_filling", "UR"): "A",
            ("presence", "impacted_tooth", None): "A. True",
            ("count", "impacted_tooth", None): "1",
            ("region", "impacted_tooth", "LL"): "<answer>A</answer>",
            ("region", "impacted_tooth", "LR"): "A. True",
            ("presence", "carious_lesion", None): "A",  # whole image only: no crop answers A, so it is not counted
            ("region", "endodontic_treatment", "UL"): "A",  # recovered on the UL crop
            ("count", "endodontic_treatment", None): "unclear",
        }
        protocol = dp.Protocol(presence_level="region", count_level="overall", region_prompt="crop")
        runner, results = self._run(script, protocol, region_of)
        f = results["img1"]["findings"]
        self.assertEqual(f["dental_filling"], {"presence": "A", "whole_image": "A", "count": 2,
                                               "regions": {"UR": "A", "UL": "B", "LL": "B", "LR": "B"}, "region_counts": None})
        self.assertEqual((f["carious_lesion"]["presence"], f["carious_lesion"]["count"]), ("B", None))
        self.assertEqual((f["endodontic_treatment"]["presence"], f["endodontic_treatment"]["count"]), ("A", None))
        # 14 whole image + 4 crops x 14 findings + 3 counts (filling, impacted, root canal) = 73; img2: 70.
        self.assertEqual((results["img1"]["call_count"], results["img2"]["call_count"]), (73, 70))
        self.assertTrue((self.root / "run" / "crops" / "img1_LL.png").is_file())
        self.assertIn(("region", "impacted_tooth", "LL"), runner.log)

        gt = ev.load_yolo(self.root / "images", self.root / "labels")
        report = ev.evaluate(gt, results, dataset="toy")
        counts = {r["condition"]: r for r in report["counts"]}
        self.assertEqual((counts["dental_filling"]["exact_rate"], counts["impacted_tooth"]["mae"]), (1.0, 0.0))
        self.assertEqual(counts["endodontic_treatment"]["count_unparseable"], 1)
        self.assertEqual(report["region_counts"], [])
        regions = {r["condition"]: r for r in report["regions"]}
        self.assertEqual((regions["impacted_tooth"]["TP"], regions["impacted_tooth"]["FP"]), (1, 1))
        whole = {r["condition"]: r for r in report["whole_image"]}
        self.assertEqual((whole["carious_lesion"]["FP"], whole["endodontic_treatment"]["FN"]), (1, 1))
        self.assertEqual((report["summary"]["mean_false_alarms_per_image"], report["summary"]["whole_image"]["FP"]), (0.0, 1))
        self.assertIn("Dental filling; count 2; location UR", dp.dentist_report(results["img1"]))

    def test_arch_scheme_and_crop_region_counts(self):
        # The whole image answers B for fillings (the fake's default); the upper jaw recovers them.
        script = {("region", "dental_filling", "upper"): "A", ("region_count", "dental_filling", "upper"): "2"}
        _, results = self._run(script, dp.Protocol(region_scheme="arch"))
        f = results["img1"]["findings"]["dental_filling"]
        self.assertEqual((f["presence"], f["whole_image"]), ("A", "B"))
        self.assertEqual((f["regions"], f["region_counts"], f["count"]), ({"upper": "A", "lower": "B"}, {"upper": 2}, 2))
        self.assertEqual(results["img1"]["call_count"], 14 + 2 * 14 + 1)
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
        self.assertEqual((report["summary"]["protocol"]["region_prompt"], report["summary"]["protocol"]["question_form"]),
                         ("crop", "separate"))
        self.assertEqual({r["condition"]: r["TP"] for r in report["regions"]}["dental_filling"], 1)
        self.assertEqual((report["region_counts"], report["whole_image"]), ([], []))


class CombinedFormTests(RunAndEvaluateTests):
    """The combined presence-and-count question in every protocol; the results keep the separate form's fields."""

    def test_region_presence_and_region_counts_in_words(self):
        script = {
            ("presence", "dental_filling", None): "A",  # the whole image is presence-only when counts are regional
            ("region", "dental_filling", "UR"): "Answer: A. True\nCount: 2",
            ("region", "endodontic_treatment", "UL"): "**Answer:** A. True\n**Count:** 1",  # missed on the whole image (default B)
            ("presence", "impacted_tooth", None): "A. True",
            ("region", "impacted_tooth", "LL"): "Answer: A. True\nCount: 1",
            ("region", "impacted_tooth", "LR"): "Answer: A. True\nCount: 0",  # A with 0: the region stands, its count is unparseable
            ("region", "carious_lesion", "UR"): "Answer: B. False\nCount: 3",  # B with 3: the B stands, nothing is counted
            ("region", "dental_implant", "UR"): "Count: 2",  # no letter: nothing is kept
            ("region", "root_fragment", "LR"): "B. False",  # a B without a count line is harmless
            ("region", "periodontal_bone_loss", "LL"): "A",  # presence-only finding: the bare regional question
        }
        runner, results = self._run(script, dp.Protocol(question_form="combined"), mode="plain")
        f = results["img1"]["findings"]
        self.assertEqual(f["dental_filling"], {"presence": "A", "whole_image": "A", "count": 2,
                                               "regions": {"UR": "A", "UL": "B", "LL": "B", "LR": "B"}, "region_counts": {"UR": 2}})
        self.assertEqual(f["endodontic_treatment"], {"presence": "A", "whole_image": "B", "count": 1,
                                                     "regions": {"UR": "B", "UL": "A", "LL": "B", "LR": "B"}, "region_counts": {"UL": 1}})
        self.assertEqual((f["impacted_tooth"]["regions"]["LR"], f["impacted_tooth"]["region_counts"], f["impacted_tooth"]["count"]),
                         ("A", {"LL": 1, "LR": None}, None))
        self.assertEqual((f["carious_lesion"]["presence"], f["carious_lesion"]["region_counts"]), ("B", {}))
        self.assertEqual((f["dental_implant"]["presence"], f["dental_implant"]["regions"]["UR"], f["dental_implant"]["region_counts"]),
                         (None, None, {}))
        self.assertEqual((f["root_fragment"]["presence"], f["root_fragment"]["region_counts"]), ("B", {}))
        self.assertEqual(f["periodontal_bone_loss"], {"presence": "A", "whole_image": "B", "count": None,
                                                      "regions": {"UR": "B", "UL": "B", "LL": "A", "LR": "B"}, "region_counts": None})
        # 14 whole-image questions + 4 regions x 14 questions and never a separate count call: 70 for every image.
        self.assertEqual((results["img1"]["call_count"], results["img2"]["call_count"]), (70, 70))
        calls = results["img1"]["calls"]
        self.assertEqual({c["stage"] for c in calls}, {"presence", "region"})
        self.assertTrue(all(c["question"] == dp.presence_question(c["condition"]) for c in calls if c["stage"] == "presence"))
        ur_filling = next(c for c in calls if c["stage"] == "region" and c["region"] == "UR" and c["condition"] == "dental_filling")
        self.assertEqual(ur_filling["question"], dp.combined_question("dental_filling", region="UR"))
        self.assertEqual(ur_filling["parsed"], {"choice": "A", "count": 2, "consistent": True})
        lr_impacted = next(c for c in calls if c["stage"] == "region" and c["region"] == "LR" and c["condition"] == "impacted_tooth")
        self.assertEqual(lr_impacted["parsed"], {"choice": "A", "count": 0, "consistent": False})
        ur_implant = next(c for c in calls if c["stage"] == "region" and c["region"] == "UR" and c["condition"] == "dental_implant")
        self.assertEqual(ur_implant["parsed"], {"choice": None, "count": 2, "consistent": None})
        ur_bone = next(c for c in calls if c["stage"] == "region" and c["region"] == "UR" and c["condition"] == "periodontal_bone_loss")
        self.assertEqual(ur_bone["question"], dp.presence_question("periodontal_bone_loss", region="UR"))
        self.assertNotIn("parsed", ur_bone)
        self.assertFalse((self.root / "run" / "crops").exists())

        gt = ev.load_yolo(self.root / "images", self.root / "labels")
        report = ev.evaluate(gt, results, dataset="toy", out_dir=self.root / "run" / "evaluation")
        self.assertEqual(report["summary"]["protocol"]["question_form"], "combined")
        presence = {r["condition"]: r for r in report["presence"]}
        self.assertEqual((presence["dental_filling"]["TP"], presence["endodontic_treatment"]["TP"], presence["dental_implant"]["unparseable"]),
                         (1, 1, 1))
        whole = {r["condition"]: r for r in report["whole_image"]}
        self.assertEqual(whole["endodontic_treatment"]["FN"], 1)
        counts = {r["condition"]: r for r in report["counts"]}
        self.assertEqual((counts["dental_filling"]["exact_rate"], counts["impacted_tooth"]["count_unparseable"]), (1.0, 1))
        region_counts = {(r["condition"], r["region"]): r for r in report["region_counts"]}
        self.assertEqual((region_counts[("impacted_tooth", "LR")]["count_unparseable"], region_counts[("dental_filling", "UR")]["exact_rate"]), (1, 1.0))
        regions = {r["condition"]: r for r in report["regions"]}
        self.assertEqual((regions["dental_filling"]["exact_set_match_rate"], regions["impacted_tooth"]["FP"]), (1.0, 1))
        self.assertIn("Dental filling; count 2 (UR 2); location UR", dp.dentist_report(results["img1"]))
        self.assertIn("Impacted tooth; count incomplete (LL 1); location LL, LR", dp.dentist_report(results["img1"]))
        # Resume rejects the other form.
        with self.assertRaises(ValueError):
            dp.run_dataset(runner, {"img1": self.images["img1"]}, self.root / "run", mode="plain", protocol=dp.Protocol())

    def test_whole_image_only(self):
        script = {
            ("presence", "dental_filling", None): "Answer: A. True\nCount: 2",
            ("presence", "impacted_tooth", None): "Answer: A. True\nCount: 0",  # present, count unparseable, and no second question
            ("presence", "carious_lesion", None): "Answer: B. False\nCount: 2",  # absent
            ("presence", "endodontic_treatment", None): "The count is 1.",  # no letter: unparseable
            ("presence", "periodontal_bone_loss", None): "A",  # presence-only finding: the bare question
        }
        protocol = dp.Protocol(presence_level="overall", count_level="overall", question_form="combined")
        _, results = self._run(script, protocol, mode="plain")
        f = results["img1"]["findings"]
        self.assertEqual(f["dental_filling"], {"presence": "A", "whole_image": "A", "count": 2, "regions": None, "region_counts": None})
        self.assertEqual((f["impacted_tooth"]["presence"], f["impacted_tooth"]["count"]), ("A", None))
        self.assertEqual((f["carious_lesion"]["presence"], f["carious_lesion"]["count"]), ("B", None))
        self.assertEqual((f["endodontic_treatment"]["presence"], f["endodontic_treatment"]["count"]), (None, None))
        self.assertEqual(f["periodontal_bone_loss"]["presence"], "A")
        # Exactly 14 calls per image: nine combined questions and five bare ones.
        self.assertEqual((results["img1"]["call_count"], results["img2"]["call_count"]), (14, 14))
        calls = results["img1"]["calls"]
        self.assertEqual({c["stage"] for c in calls}, {"presence"})
        self.assertEqual(next(c["question"] for c in calls if c["condition"] == "dental_filling"), dp.combined_question("dental_filling"))
        self.assertEqual(next(c["question"] for c in calls if c["condition"] == "periodontal_bone_loss"),
                         dp.presence_question("periodontal_bone_loss"))
        report = ev.evaluate(ev.load_yolo(self.root / "images", self.root / "labels"), results, dataset="toy")
        counts = {r["condition"]: r for r in report["counts"]}
        self.assertEqual((counts["dental_filling"]["exact_rate"], counts["impacted_tooth"]["count_unparseable"]), (1.0, 1))
        self.assertEqual((report["summary"]["mean_calls_per_image"], report["whole_image"], report["region_counts"]), (14.0, [], []))
        self.assertIn("Dental filling; count 2", dp.dentist_report(results["img1"]))

    def test_region_presence_with_whole_image_counts(self):
        script = {
            ("presence", "dental_filling", None): "Answer: A. True\nCount: 2",  # counted on the whole image: no follow-up
            ("region", "dental_filling", "UR"): "A",
            ("presence", "endodontic_treatment", None): "Answer: B. False\nCount: 0",  # missed on the whole image ...
            ("region", "endodontic_treatment", "UL"): "A",  # ... recovered in UL: the Figure 9 whole-image count is asked
            ("count", "endodontic_treatment", None): "1",
            ("presence", "impacted_tooth", None): "Answer: A. True\nCount: 0",  # A with 0 stays unparseable: no second question
            ("region", "impacted_tooth", "LL"): "A",
            ("region", "root_fragment", "LR"): "A",  # whole image unscripted (B): recovered, follow-up count unparseable
            ("count", "root_fragment", None): "unclear",
        }
        protocol = dp.Protocol(presence_level="region", count_level="overall", question_form="combined")
        _, results = self._run(script, protocol, mode="plain")
        f = results["img1"]["findings"]
        self.assertEqual(f["dental_filling"], {"presence": "A", "whole_image": "A", "count": 2,
                                               "regions": {"UR": "A", "UL": "B", "LL": "B", "LR": "B"}, "region_counts": None})
        endo = f["endodontic_treatment"]
        self.assertEqual((endo["whole_image"], endo["presence"], endo["count"]), ("B", "A", 1))
        self.assertEqual((f["impacted_tooth"]["presence"], f["impacted_tooth"]["count"]), ("A", None))
        self.assertEqual((f["root_fragment"]["presence"], f["root_fragment"]["count"]), ("A", None))
        # 14 whole image (9 combined + 5 bare) + 4 x 14 bare regional + 2 follow-up counts = 72; img2: 70.
        self.assertEqual((results["img1"]["call_count"], results["img2"]["call_count"]), (72, 70))
        calls = results["img1"]["calls"]
        self.assertEqual([(c["condition"], c["question"]) for c in calls if c["stage"] == "count"],
                         [("endodontic_treatment", dp.count_question("endodontic_treatment")),
                          ("root_fragment", dp.count_question("root_fragment"))])
        self.assertTrue(all(c["question"] == dp.presence_question(c["condition"], region=c["region"]) for c in calls if c["stage"] == "region"))
        self.assertEqual(next(c["parsed"] for c in calls if c["stage"] == "presence" and c["condition"] == "endodontic_treatment"),
                         {"choice": "B", "count": 0, "consistent": True})
        report = ev.evaluate(ev.load_yolo(self.root / "images", self.root / "labels"), results, dataset="toy")
        counts = {r["condition"]: r for r in report["counts"]}
        self.assertEqual((counts["endodontic_treatment"]["exact_rate"], counts["impacted_tooth"]["count_unparseable"]), (1.0, 1))
        self.assertEqual({r["condition"]: r["FN"] for r in report["whole_image"]}["endodontic_treatment"], 1)

    def test_overall_presence_with_crop_region_counts(self):
        region_of = {png: region for region, png in dp.make_crops(self.images["img1"], "quadrant").items()}
        script = {
            ("presence", "dental_filling", None): "A",
            ("region", "dental_filling", "UR"): "Answer: A. True\nCount: 2",
            ("region", "impacted_tooth", "LL"): "Answer: A. True\nCount: 1",
            ("region", "impacted_tooth", "LR"): "Answer: B. False\nCount: 1",  # B with 1: that region's count is unparseable
            ("region", "endodontic_treatment", "UL"): "Answer: A. True\nCount: 1",  # counted although missed on the whole image
        }
        protocol = dp.Protocol(presence_level="overall", count_level="region", region_prompt="crop", question_form="combined")
        _, results = self._run(script, protocol, region_of, mode="plain")
        f = results["img1"]["findings"]
        self.assertEqual(f["dental_filling"], {"presence": "A", "whole_image": "A", "count": 2, "regions": None,
                                               "region_counts": {"UR": 2, "UL": 0, "LL": 0, "LR": 0}})
        self.assertEqual((f["impacted_tooth"]["region_counts"], f["impacted_tooth"]["count"]), ({"UR": 0, "UL": 0, "LL": 1, "LR": None}, None))
        self.assertEqual((f["endodontic_treatment"]["presence"], f["endodontic_treatment"]["region_counts"]["UL"], f["endodontic_treatment"]["count"]),
                         ("B", 1, 1))
        self.assertEqual(f["periodontal_bone_loss"]["region_counts"], None)
        # 14 bare whole-image questions + 4 crops x 9 combined questions = 50; the whole-image wording goes with every crop.
        self.assertEqual((results["img1"]["call_count"], results["img2"]["call_count"]), (50, 50))
        crop_calls = [c for c in results["img1"]["calls"] if c["stage"] == "region_count"]
        self.assertEqual(len(crop_calls), 36)
        self.assertTrue(all(c["question"] == dp.combined_question(c["condition"], crop=True) for c in crop_calls))
        self.assertEqual(crop_calls[0]["parsed"], {"choice": "B", "count": 0, "consistent": True})
        self.assertTrue((self.root / "run" / "crops" / "img1_LL.png").is_file())
        report = ev.evaluate(ev.load_yolo(self.root / "images", self.root / "labels"), results, dataset="toy")
        regions = {r["condition"]: r for r in report["regions"]}
        self.assertEqual((regions["dental_filling"]["from_counts"], regions["dental_filling"]["exact_set_match_rate"]), (1, 1.0))
        region_counts = {(r["condition"], r["region"]): r for r in report["region_counts"]}
        self.assertEqual((region_counts[("dental_filling", "UL")]["exact_rate"], region_counts[("dental_filling", "UL")]["n_scored"]), (1.0, 1))
        self.assertIn("Dental filling; count 2 (UR 2, UL 0, LL 0, LR 0); location UR", dp.dentist_report(results["img1"]))


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
