"""Offline tests for the location adapter: drawing, JSON parsing, fake LLM and fake DentVLM adapters,
resume, and the hand-off into evaluation."""
from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import dental_eval as ev
import dental_pipeline as dp
import location_adapter as la
from test_dental_pipeline import SCRIPT, FakeRunner, _blank_image

UPPER_RIGHT = "the right posterior region of the upper dentition"
LOWER_LEFT = "the left posterior region of the lower dentition"


class UnitTests(unittest.TestCase):
    def test_units_map_onto_cells_like_fdi(self):
        self.assertEqual(dp.fdi_unit(1, 6), "Q1-posterior")
        self.assertEqual(dp.fdi_unit(2, 1), "Q2-anterior")
        self.assertEqual(dp.fdi_unit(8, 5), "Q4-posterior")  # primary dentition folds onto 1-4
        self.assertEqual(dp.unit_cell("Q1-posterior"), "upper-left")   # patient's right = image left
        self.assertEqual(dp.unit_cell("Q2-posterior"), "upper-right")
        self.assertEqual(dp.unit_cell("Q3-anterior"), "lower-anterior")
        self.assertEqual(dp.unit_cell("Q4-posterior"), "lower-left")
        self.assertEqual(dp.unit_cell("Q1-posterior", left_is_image_left=False), "upper-right")
        self.assertEqual(dp.units_to_cells(["Q4-anterior", "Q3-anterior", "Q2-posterior"]), ["upper-right", "lower-anterior"])
        for quadrant in (1, 2, 3, 4):
            for tooth in range(1, 9):
                self.assertEqual(ev.fdi_cell(quadrant, tooth), dp.unit_cell(dp.fdi_unit(quadrant, tooth)))
        with self.assertRaises(ValueError):
            dp.unit_cell("Q5-anterior")

    def test_parse_units(self):
        text = 'Sure:\n```json\n{"boxes": [{"id": 1, "units": ["Q1-posterior", "bogus"], "teeth": [16, "17"]},' \
               ' {"id": 2, "units": [], "teeth": []}, {"id": 9, "units": ["Q2-anterior"]}, "junk"]}\n```'
        parsed = la.parse_units(text, 3)
        self.assertEqual(parsed, {1: {"units": ["Q1-posterior"], "teeth": [16, 17]}, 2: {"units": [], "teeth": []}})
        self.assertEqual(la.parse_units("no json here", 3), {})
        self.assertEqual(la.parse_units('{"boxes": "nope"}', 3), {})


class FakeClient:
    """OpenAI-style client whose replies come from a queue; records every request."""

    def __init__(self, replies: list[str]):
        self.replies, self.requests = list(replies), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request):
        self.requests.append(request)
        text = self.replies.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")])


def _reply(entries: list[tuple]) -> str:
    return json.dumps({"boxes": [{"id": i, "units": units, "teeth": teeth} for i, units, teeth in entries]})


@unittest.skipIf(importlib.util.find_spec("PIL") is None, "Pillow not installed")
class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "images").mkdir()
        (self.root / "labels").mkdir()
        _blank_image(self.root / "images" / "img1.png")
        _blank_image(self.root / "images" / "img2.png")
        # img1: two fillings in the upper image-left cell (geometry), one impacted tooth lower image-right.
        (self.root / "labels" / "img1.txt").write_text(
            "2 0.20 0.25 0.05 0.05\n2 0.30 0.30 0.05 0.05\n6 0.80 0.80 0.10 0.10\n")
        (self.root / "labels" / "img2.txt").write_text("")
        self.gt = ev.load_yolo(self.root / "images", self.root / "labels")
        self.images = {"img1": self.root / "images" / "img1.png", "img2": self.root / "images" / "img2.png"}

    def tearDown(self):
        self.tmp.cleanup()

    def test_draw_and_spotlight(self):
        from PIL import Image

        boxes = self.gt["img1"]["boxes"]
        jpeg, width, height, pixels = la.draw_boxes(self.images["img1"], boxes, max_side=280)
        self.assertEqual((width, height), (280, 140))  # 560x280 scaled to the longest side
        with Image.open(io.BytesIO(jpeg)) as drawn:
            self.assertEqual((drawn.format, drawn.size), ("JPEG", (280, 140)))
        self.assertEqual(pixels[0], [49, 32, 63, 38])
        self.assertEqual(len(pixels), 3)
        jpeg, width, height, _ = la.draw_boxes(self.images["img1"], boxes[:1], numbered=False, corner_labels=False, color="red")
        self.assertEqual((width, height), (560, 280))

        png = la.spotlight(self.images["img1"], boxes[2], margin=0.05)
        with Image.open(io.BytesIO(png)) as spot:
            self.assertEqual(spot.size, (560, 280))
            self.assertEqual(spot.getpixel((10, 10)), (0, 0, 0))          # outside the box: black
            self.assertEqual(spot.getpixel((448, 224)), (128, 128, 128))  # inside: the radiograph

    def test_llm_adapter_chunks_retries_and_falls_back(self):
        client = FakeClient([
            _reply([(1, ["Q2-posterior"], [26]), (2, ["Q1-posterior", "Q1-anterior"], [13, 14])]),  # boxes 1-2
            "garbage",                                                # box 3, first try: no JSON -> retry
            _reply([(1, [], [])]),                                    # box 3, second try: unplaceable
        ])
        adapter = la.LLMAdapter(base_url=None, api_key="x", model="fake/model-1", max_boxes_per_call=2, client=client)
        self.assertEqual(adapter.name, "llm-fake-model-1")
        rows = adapter.adapt(self.images["img1"], self.gt["img1"]["boxes"], "img1", self.root / "drawn")
        self.assertEqual({k: v for k, v in rows[0].items() if k != "raw"},
                         {"regions": ["upper-right"], "units": ["Q2-posterior"], "teeth": [26], "source": "llm"})
        self.assertIn('"Q2-posterior"', rows[0]["raw"])
        self.assertEqual(rows[1]["regions"], ["upper-anterior", "upper-left"])
        self.assertEqual((rows[2]["regions"], rows[2]["units"], rows[2]["source"]), (None, [], None))
        self.assertEqual(len(client.requests), 3)
        self.assertEqual(client.requests[0]["max_tokens"], 4096)
        self.assertNotIn("temperature", client.requests[0])
        self.assertEqual(client.requests[0]["messages"][0]["role"], "system")
        user_text = client.requests[0]["messages"][1]["content"][0]["text"]
        self.assertIn("1. Dental filling - [", user_text)
        self.assertIn("2. Dental filling - [", user_text)
        self.assertNotIn("3.", user_text.split("TASK")[0])
        self.assertTrue((self.root / "drawn" / "img1_1.jpg").is_file())
        self.assertTrue((self.root / "drawn" / "img1_2.jpg").is_file())

        reasoning = la.LLMAdapter(None, "x", "gpt-5", token_param="max_completion_tokens", temperature=0.0,
                                  request_options={"reasoning_effort": "low"}, client=FakeClient([_reply([(1, ["Q3-anterior"], [])])]))
        reasoning.adapt(self.images["img1"], self.gt["img1"]["boxes"][:1])
        request = reasoning.client.requests[0]
        self.assertEqual((request["max_completion_tokens"], request["temperature"], request["reasoning_effort"]), (4096, 0.0, "low"))
        with self.assertRaises(ValueError):
            la.LLMAdapter(None, "x", "m", token_param="max_new_tokens", client=client)

    def test_adapt_dataset_then_evaluate(self):
        client = FakeClient([_reply([(1, ["Q2-posterior"], [26]), (2, ["Q2-posterior"], [27]), (3, ["Q4-posterior"], [46])])])
        adapter = la.LLMAdapter(None, "x", "fake", client=client)
        out = self.root / "truth"
        adapted = la.adapt_dataset(adapter, self.gt, out)
        self.assertEqual(set(adapted), {"img1", "img2"})
        self.assertEqual(adapted["img2"]["boxes"], [])
        records = adapted["img1"]["boxes"]
        self.assertEqual([r["regions"] for r in records], [["upper-right"], ["upper-right"], ["lower-left"]])
        self.assertEqual([r["geometry"] for r in records], [["upper-left"], ["upper-left"], ["lower-right"]])
        self.assertEqual({r["source"] for r in records}, {"llm"})
        self.assertEqual(la.summarize(adapted), {"images": 2, "boxes": 3, "by_source": {"llm": 3},
                                                 "agreement_with_geometry": 0.0, "multi_region_boxes": 0})
        self.assertTrue((out / "manifest.json").is_file())
        self.assertTrue((out / "drawn" / "img1.jpg").is_file())
        # Resume: no new calls; a different adapter configuration is refused.
        la.adapt_dataset(adapter, self.gt, out)
        self.assertEqual(len(client.requests), 1)
        with self.assertRaises(ValueError):
            la.adapt_dataset(la.LLMAdapter(None, "x", "other", client=client), self.gt, out)
        self.assertEqual(la.load_adapted(out)["img1"]["boxes"][0]["units"], ["Q2-posterior"])

        # The fake DentVLM names upper-left for the fillings: a region hit under geometry, a miss under the adapter.
        results = dp.load_results(dp.run_dataset(FakeRunner(SCRIPT), self.images, self.root / "run"))
        geometry = {r["condition"]: r for r in ev.evaluate(self.gt, results, dataset="toy")["regions"]}
        self.assertEqual((geometry["dental_filling"]["TP"], geometry["dental_filling"]["FN"]), (1, 0))
        truth = ev.apply_adapted(self.gt, adapted)
        self.assertEqual(truth["img1"]["boxes"][0]["regions"], ["upper-right"])
        self.assertEqual(truth["img1"]["boxes"][0]["region_source"], "llm")
        self.assertNotIn("regions", self.gt["img1"]["boxes"][0])  # the original truth is untouched
        report = ev.evaluate(truth, results, dataset="toy")
        adapted_rows = {r["condition"]: r for r in report["regions"]}
        self.assertEqual((adapted_rows["dental_filling"]["TP"], adapted_rows["dental_filling"]["FP"],
                          adapted_rows["dental_filling"]["FN"]), (0, 1, 1))
        self.assertEqual(report["summary"]["location_truth"], {"boxes": 3, "by_source": {"llm": 3}})
        self.assertEqual(ev.evaluate(self.gt, results, dataset="toy")["summary"]["location_truth"],
                         {"boxes": 3, "by_source": {"geometry": 3}})
        with self.assertRaises(ValueError):
            ev.apply_adapted(self.gt, {"img1": {"boxes": records[:1]}})

    def test_truth_agreement_on_fdi_boxes(self):
        gt = {"a": {"path": str(self.images["img1"]), "annotated": set(dp.CONDITIONS), "boxes": [
            {"condition": "carious_lesion", "xc": 0.2, "yc": 0.2, "w": 0.1, "h": 0.1, "fdi": (1, 6)},   # geometry agrees
            {"condition": "carious_lesion", "xc": 0.30, "yc": 0.2, "w": 0.05, "h": 0.1, "fdi": (1, 3)},  # a canine left of the fixed anterior window
        ]}}
        adapted = {"a": {"boxes": [
            {"regions": ["upper-left"], "units": ["Q1-posterior"], "source": "llm", "geometry": ["upper-left"]},
            {"regions": ["upper-anterior"], "units": ["Q1-anterior"], "source": "llm", "geometry": ["upper-left"]},
        ]}}
        self.assertEqual(ev.truth_agreement(gt, adapted), {"boxes_with_fdi": 2, "adapter_exact": 1.0, "adapter_contains": 1.0,
                                                           "geometry_exact": 0.5, "geometry_contains": 0.5})
        self.assertEqual(ev.truth_agreement(self.gt, {}), {"boxes_with_fdi": 0, "adapter_exact": None, "adapter_contains": None,
                                                           "geometry_exact": None, "geometry_contains": None})

    def test_fdm_adapter_spotlight(self):
        class SpotRunner:
            def __init__(self):
                self.log = []

            def settings(self):
                return {"model": "fake"}

            def ask(self, image, question):
                self.log.append(question)
                if question == dp.questions_for("fillings")[0]:
                    text = f"Yes\nA filling is seen in {UPPER_RIGHT}."
                elif question == dp.questions_for("impacted_tooth")[0]:
                    text = "No\nNo impacted tooth is visible."
                else:
                    text = "Yes\nSomething is seen."  # positive without a descriptor
                return {"text": text, "finish_reason": "stop", "truncated": False}

        runner = SpotRunner()
        adapter = la.FdmAdapter(runner, margin=0.05)
        boxes = self.gt["img1"]["boxes"] + [{"condition": "surgical_device", "xc": 0.5, "yc": 0.8, "w": 0.1, "h": 0.1},
                                             {"condition": "prosthetic_restoration", "xc": 0.5, "yc": 0.2, "w": 0.1, "h": 0.1}]
        rows = adapter.adapt(self.images["img1"], boxes, "img1", self.root / "spots")
        self.assertEqual([r["source"] for r in rows], ["fdm", "fdm", None, None, None])
        self.assertEqual(rows[0]["regions"], ["upper-right"])
        self.assertIsNone(rows[2]["regions"])                 # answered No -> geometry later
        self.assertIsNone(rows[3]["raw"])                     # no DentVLM task: no call
        self.assertIn("[prosthetic_bridge]", rows[4]["raw"])  # crown then bridge, no descriptor either time
        self.assertEqual(len(runner.log), 5)
        self.assertTrue((self.root / "spots" / "img1_1.png").is_file())
        self.assertFalse((self.root / "spots" / "img1_4.png").exists())

        gt = {"img1": self.gt["img1"]}
        adapted = la.adapt_dataset(adapter, gt, self.root / "truth_fdm")
        self.assertEqual([r["source"] for r in adapted["img1"]["boxes"]], ["fdm", "fdm", "geometry"])
        self.assertEqual(adapted["img1"]["boxes"][2]["regions"], ["lower-right"])
        truth = ev.apply_adapted(gt, adapted)
        self.assertEqual([b["regions"] for b in truth["img1"]["boxes"]], [["upper-right"], ["upper-right"], ["lower-right"]])
        self.assertEqual(la.summarize(adapted)["by_source"], {"fdm": 2, "geometry": 1})


if __name__ == "__main__":
    unittest.main()
