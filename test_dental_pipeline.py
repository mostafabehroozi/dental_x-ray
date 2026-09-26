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
        self.assertEqual(len(dp.TASKS), 12)
        self.assertEqual(dp.Protocol().tasks(), tuple(dp.TASKS))
        with self.assertRaises(KeyError):
            dp.questions_for("periodontal_disease")

    def test_vote(self):
        answers = [{"answer": "yes", "regions": ["upper-left"]}, {"answer": "no", "regions": []},
                   {"answer": "yes", "regions": ["upper-left", "lower-right"]}]
        self.assertEqual(dp.vote(answers, "union"), {"presence": "yes", "regions": ["upper-left", "lower-right"]})
        self.assertEqual(dp.vote(answers, "majority"), {"presence": "yes", "regions": ["upper-left"]})
        tie = [{"answer": "yes", "regions": []}, {"answer": "no", "regions": []}, {"answer": None, "regions": []}]
        self.assertEqual(dp.vote(tie, "union"), {"presence": None, "regions": None})
        self.assertEqual(dp.vote([{"answer": "no", "regions": []}], "union"), {"presence": "no", "regions": None})

    def test_cell_descriptions(self):
        self.assertEqual(dp.describe_cell("upper-left", left_is_image_left=True), "patient's upper right posterior (image left)")
        self.assertEqual(dp.describe_cell("lower-right", left_is_image_left=True), "patient's lower left posterior (image right)")
        self.assertEqual(dp.describe_cell("upper-anterior"), "upper anterior region")
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
