"""Offline tests for the location adapter: drawing, JSON parsing, option extraction, fake LLM and fake
DentalGPT adapters, resume, and the hand-off into evaluation."""
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
from test_dental_pipeline import FakeRunner, _blank_image


class UnitTests(unittest.TestCase):
    def test_units_map_onto_windows(self):
        self.assertEqual(dp.fdi_unit(1, 6), "Q1-posterior")
        self.assertEqual(dp.fdi_unit(2, 1), "Q2-anterior")
        self.assertEqual(dp.fdi_unit(8, 5), "Q4-posterior")  # primary dentition folds onto 1-4
        self.assertEqual([dp.unit_region(u) for u in dp.UNITS], ["UR", "UR", "UL", "UL", "LL", "LL", "LR", "LR"])
        self.assertEqual(dp.unit_region("Q3-anterior", "arch"), "lower")
        self.assertEqual(dp.units_to_regions(["Q4-anterior", "Q3-anterior", "Q2-posterior"]), ["UL", "LL", "LR"])
        self.assertEqual(dp.units_to_regions(["Q4-anterior", "Q3-anterior"], "arch"), ["lower"])
        self.assertEqual(dp.quadrants_to_regions(["LR", "UR"]), ["UR", "LR"])
        self.assertEqual(dp.quadrants_to_regions(["LR", "UR"], "arch"), ["upper", "lower"])
        self.assertEqual(ev.fdi_quadrant(1), "UR")
        self.assertEqual(ev.fdi_quadrant(7), "LL")
        with self.assertRaises(ValueError):
            dp.unit_region("Q5-anterior")
        # FDI wins over geometry when present, at both levels.
        box = {"xc": 0.8, "yc": 0.8, "w": 0.1, "h": 0.1, "fdi": (1, 6)}
        self.assertEqual((ev.box_regions(box, "quadrant"), ev.box_regions(box, "arch")), ({"UR"}, {"upper"}))
        self.assertEqual(ev.box_source(box), "fdi")

    def test_parse_units(self):
        text = 'Sure:\n```json\n{"boxes": [{"id": 1, "units": ["Q1-posterior", "bogus"], "teeth": [16, "17"]},' \
               ' {"id": 2, "units": [], "teeth": []}, {"id": 9, "units": ["Q2-anterior"]}, "junk"]}\n```'
        parsed = la.parse_units(text, 3)
        self.assertEqual(parsed, {1: {"units": ["Q1-posterior"], "teeth": [16, 17]}, 2: {"units": [], "teeth": []}})
        self.assertEqual(la.parse_units("no json here", 3), {})
        self.assertEqual(la.parse_units('{"boxes": "nope"}', 3), {})

    def test_extract_option(self):
        cases = {
            "<think>...</think><answer>A</answer>": "A",
            "<answer>C. Both jaws</answer>": "C",
            "B. Lower jaw": "B",
            "A. Upper jaw\nB. Lower jaw\nC. Both jaws\n\nThe answer is B.": "B",
            "The red box is in the lower jaw.": "B",
            "The marked region lies in the mandible.": "B",
            "It spans both jaws.": "C",
            "Answer: A": "A",
            "The upper jaw, unless it is the lower jaw.": None,  # two options named, no letter
            "<think>I think A. Upper jaw but": None,
            "": None,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(la.extract_option(text, la.JAW_OPTIONS), expected)
        self.assertEqual(la.extract_option("The box is on the left side of the image.", la.SIDE_OPTIONS), "A")
        self.assertEqual(la.extract_option("It crosses the midline.", la.SIDE_OPTIONS), "C")


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
        _blank_image(self.root / "images" / "img2.png", shade=100)
        # img1: two fillings in the UR window (image left), one impacted tooth in the LL window.
        (self.root / "labels" / "img1.txt").write_text(
            "2 0.20 0.25 0.05 0.05\n2 0.30 0.30 0.05 0.05\n6 0.80 0.80 0.10 0.10\n")
        (self.root / "labels" / "img2.txt").write_text("")
        self.gt = ev.load_yolo(self.root / "images", self.root / "labels")
        self.images = {"img1": self.root / "images" / "img1.png", "img2": self.root / "images" / "img2.png"}

    def tearDown(self):
        self.tmp.cleanup()

    def test_draw_boxes(self):
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
        with Image.open(io.BytesIO(jpeg)) as drawn:
            r, g, b = drawn.getpixel((round(0.175 * 560) + 1, round(0.25 * 280)))  # on the left edge of box 1
            self.assertTrue(r > 200 and g < 80 and b < 80)

    def test_llm_adapter_chunks_retries_and_falls_back(self):
        client = FakeClient([
            _reply([(1, ["Q2-posterior"], [26]), (2, ["Q1-posterior", "Q1-anterior"], [13, 14])]),  # boxes 1-2
            "garbage",                                                # box 3, first try: no JSON -> retry
            _reply([(1, [], [])]),                                    # box 3, second try: unplaceable
        ])
        adapter = la.LLMAdapter(base_url=None, api_key="x", model="fake/model-1", max_boxes_per_call=2, client=client)
        self.assertEqual(adapter.name, "llm-fake-model-1")
        rows = adapter.adapt(self.images["img1"], self.gt["img1"]["boxes"], "img1", self.root / "drawn")
        self.assertEqual({k: v for k, v in rows[0].items() if k not in ("raw", "attempts", "fallback_reason")},
                         {"regions": ["UL"], "units": ["Q2-posterior"], "teeth": [26], "source": "llm"})
        self.assertIn('"Q2-posterior"', rows[0]["raw"])
        self.assertEqual(rows[1]["regions"], ["UR"])
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

    def test_location_parse_failure_policy_and_attempt_audit(self):
        adapter = la.LLMAdapter(None, "x", "fake", parse_retries=0, failure_policy="exclude",
                                client=FakeClient(["not json"]))
        row = adapter.adapt(self.images["img1"], self.gt["img1"]["boxes"][:1], "img1")[0]
        self.assertEqual((row["source"], row["regions"], row["fallback_reason"]),
                         ("excluded", [], "invalid_json"))
        self.assertEqual(len(row["attempts"]), 1)
        with self.assertRaisesRegex(ValueError, "remained unparseable"):
            la.LLMAdapter(None, "x", "fake", parse_retries=0, failure_policy="error",
                          client=FakeClient(["not json"])).adapt(
                              self.images["img1"], self.gt["img1"]["boxes"][:1], "img1")

    def test_adapt_dataset_then_evaluate(self):
        client = FakeClient([_reply([(1, ["Q2-posterior"], [26]), (2, ["Q2-posterior"], [27]), (3, ["Q4-posterior"], [46])])])
        adapter = la.LLMAdapter(None, "x", "fake", client=client)
        out = self.root / "truth"
        adapted = la.adapt_dataset(adapter, self.gt, out)
        self.assertEqual(set(adapted), {"img1", "img2"})
        self.assertEqual(adapted["img2"]["boxes"], [])
        records = adapted["img1"]["boxes"]
        self.assertEqual([r["regions"] for r in records], [["UL"], ["UL"], ["LR"]])
        self.assertEqual([r["geometry"] for r in records], [["UR"], ["UR"], ["LL"]])
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

        # The fake DentalGPT answers True on the UR crop for fillings: a hit under geometry, a miss under the adapter.
        region_of = {png: region for region, png in dp.make_crops(self.images["img1"], "quadrant").items()}
        script = {("presence", "dental_filling", None): "A", ("count", "dental_filling", None): "2",
                  ("region", "dental_filling", "UR"): "A"}
        results = dp.load_results(dp.run_dataset(
            FakeRunner(script, region_of), self.images, self.root / "run",
            protocol=dp.Protocol(presence_level="region", count_level="overall", region_prompt="crop")))
        geometry = {r["condition"]: r for r in ev.evaluate(self.gt, results, dataset="toy")["regions"]}
        self.assertEqual((geometry["dental_filling"]["TP"], geometry["dental_filling"]["FN"]), (1, 0))
        truth = ev.apply_adapted(self.gt, adapted)
        self.assertEqual(truth["img1"]["boxes"][0]["regions"], ["UL"])
        self.assertEqual(truth["img1"]["boxes"][0]["region_source"], "llm")
        self.assertNotIn("regions", self.gt["img1"]["boxes"][0])  # the original truth is untouched
        report = ev.evaluate(truth, results, dataset="toy")
        adapted_rows = {r["condition"]: r for r in report["regions"]}
        self.assertEqual((adapted_rows["dental_filling"]["TP"], adapted_rows["dental_filling"]["FP"],
                          adapted_rows["dental_filling"]["FN"]), (0, 1, 1))
        self.assertEqual(report["summary"]["location_truth"], {"boxes": 3, "by_source": {"llm": 3}})
        self.assertEqual(ev.evaluate(self.gt, results, dataset="toy")["summary"]["location_truth"],
                         {"boxes": 3, "by_source": {"geometry": 3}})
        # Arch level derives from the quadrants.
        self.assertEqual(ev.box_regions(truth["img1"]["boxes"][2], "arch"), {"lower"})
        with self.assertRaises(ValueError):
            ev.apply_adapted(self.gt, {"img1": {"boxes": records[:1]}})
        misaligned = json.loads(json.dumps(adapted))
        misaligned["img1"]["boxes"][0]["condition"] = "wrong"
        with self.assertRaisesRegex(ValueError, "order/content"):
            ev.apply_adapted(self.gt, misaligned)

    def test_truth_agreement_on_fdi_boxes(self):
        gt = {"a": {"path": str(self.images["img1"]), "annotated": set(dp.CONDITIONS), "boxes": [
            {"condition": "carious_lesion", "xc": 0.2, "yc": 0.2, "w": 0.1, "h": 0.1, "fdi": (1, 6)},   # geometry agrees
            {"condition": "carious_lesion", "xc": 0.58, "yc": 0.2, "w": 0.04, "h": 0.1, "fdi": (1, 1)},  # a rotated patient: 11 right of centre
        ]}}
        adapted = {"a": {"boxes": [
            {"regions": ["UR"], "units": ["Q1-posterior"], "source": "llm", "geometry": ["UR"]},
            {"regions": ["UR"], "units": ["Q1-anterior"], "source": "llm", "geometry": ["UL"]},
        ]}}
        self.assertEqual(ev.truth_agreement(gt, adapted), {"boxes_with_fdi": 2, "adapter_exact": 1.0, "adapter_contains": 1.0,
                                                           "geometry_exact": 0.5, "geometry_contains": 0.5})
        self.assertEqual(ev.truth_agreement(self.gt, {}), {"boxes_with_fdi": 0, "adapter_exact": None, "adapter_contains": None,
                                                           "geometry_exact": None, "geometry_contains": None})

    def test_fdm_adapter_mcq(self):
        class McqRunner:
            """Scripted per box order: (jaw answer, side answer)."""

            def __init__(self, script):
                self.script, self.log, self.calls = script, [], 0

            def settings(self):
                return {"model": "fake"}

            def ask(self, image, question):
                self.log.append(question)
                box_index = self.calls // 2
                answer = self.script[box_index][self.calls % 2]
                self.calls += 1
                return {"text": answer, "finish_reason": "stop", "truncated": False}

        runner = McqRunner([("<answer>A</answer>", "<answer>B</answer>"),   # upper, image right -> UL
                            ("A. Upper jaw", "C"),                          # upper, both sides -> UR + UL
                            ("???", "B")])                                  # unparseable -> geometry later
        adapter = la.FdmAdapter(runner, mode="tagged")
        rows = adapter.adapt(self.images["img1"], self.gt["img1"]["boxes"], "img1", self.root / "marked")
        self.assertEqual([r["regions"] for r in rows], [["UL"], ["UR", "UL"], None])
        self.assertEqual([r["source"] for r in rows], ["fdm", "fdm", None])
        self.assertEqual(len(runner.log), 6)
        self.assertTrue(runner.log[0].startswith(la.JAW_QUESTION) and runner.log[0].endswith(dp.THINK_SUFFIX))
        self.assertTrue(runner.log[1].startswith(la.SIDE_QUESTION))
        self.assertTrue((self.root / "marked" / "img1_3.jpg").is_file())
        with self.assertRaises(ValueError):
            la.FdmAdapter(runner, mode="loud")

        runner = McqRunner([("A", "B"), ("A", "C"), ("???", "B")])
        gt = {"img1": self.gt["img1"]}
        adapted = la.adapt_dataset(la.FdmAdapter(runner), gt, self.root / "truth_fdm")
        self.assertEqual([r["source"] for r in adapted["img1"]["boxes"]], ["fdm", "fdm", "geometry"])
        self.assertEqual(adapted["img1"]["boxes"][2]["regions"], ["LL"])
        truth = ev.apply_adapted(gt, adapted)
        self.assertEqual([b["regions"] for b in truth["img1"]["boxes"]], [["UL"], ["UR", "UL"], ["LL"]])
        self.assertEqual(la.summarize(adapted), {"images": 1, "boxes": 3, "by_source": {"fdm": 2, "geometry": 1},
                                                 "agreement_with_geometry": 0.3333, "multi_region_boxes": 1})


if __name__ == "__main__":
    unittest.main()
