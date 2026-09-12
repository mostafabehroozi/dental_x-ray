"""Offline tests for the report writer: the dense structured input (rationale and crop locations, extra
tasks, untrained and not-assessed findings), the prompt, verification of the model's JSON against that
input, the repair turn, rendering, the fallback, and the resumable loop."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import dental_pipeline as dp
import report_writer as rw

UPPER_LEFT = "the left posterior region of the upper dentition"      # DentVLM's words: the patient's upper right
LOWER_RIGHT = "the right posterior region of the lower dentition"
LOWER_LEFT = "the left posterior region of the lower dentition"
UPPER_ANTERIOR = "the anterior region of the upper dentition"


class ScriptedRunner:
    """Answers by question text (whole image) or by (task, cell) for crop bytes; unscripted answers are No / 0."""

    def __init__(self, script: dict, cell_of: dict | None = None):
        self.script, self.cell_of, self.calls = script, cell_of or {}, 0

    def settings(self):
        return {"model": "fake"}

    def ask(self, image, question):
        self.calls += 1
        if isinstance(image, bytes):
            task = next(t for t in list(dp.TASKS) + list(dp.UNTRAINED_LABELS) if question in dp.questions_for(t))
            text = self.script.get((task, self.cell_of[image]), "No\nNothing of the kind is seen.")
        elif question in dp.COUNT_QUESTIONS.values():
            text = self.script.get(question, "0")
        else:
            text = self.script.get(question, "No\nNothing of the kind is seen.")
        return {"text": text, "finish_reason": "stop", "truncated": False,
                "prompt_tokens": 100, "completion_tokens": 5, "latency_seconds": 0.0}


class FakeClient:
    """OpenAI-style client that records every request and answers from a list of texts."""

    def __init__(self, texts: list[str], finish_reason: str = "stop"):
        self.texts, self.requests, self.finish_reason = list(texts), [], finish_reason
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request):
        self.requests.append(request)
        text = self.texts.pop(0) if self.texts else "{}"
        message = SimpleNamespace(content=text)
        usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=300)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=self.finish_reason)], usage=usage)


def q(task):
    return dp.questions_for(task)[0]


SCRIPT = {
    q("fillings"): f"Yes\nFillings appear as radiopaque material in {UPPER_LEFT}.",
    q("impacted_tooth"): f"Yes\nAn impacted tooth is seen in {LOWER_RIGHT} and {LOWER_LEFT}.",
    q("caries"): "Yes\nCaries is suspected.",  # yes without a region
    q("prosthetic_bridge"): f"Yes\nA prosthetic bridge is seen in {UPPER_ANTERIOR}.",
    q("residual_crown"): "Yes\nA residual crown is visible.",  # extra task, no benchmark class
    q("implant"): "Yes and no.",  # unparseable
}


def good_report(structured: dict, language: str = "English", **overrides) -> dict:
    """A faithful report JSON for the structured findings, as a good model would return it."""
    status = {f["finding"]: f["status"] for f in structured["findings"]}
    sections = [{"category": c["key"], "heading": c["label"],
                 "findings": [{"finding": k, "status": status[k], "statement": f"Statement about {k}."} for k in c["findings"]]}
                for c in structured["categories"]]
    report = {
        "language": language, "title": "Panoramic radiograph: automated findings report",
        "headings": {"image": "Image", "findings": "Findings", "impression": "Impression",
                     "not_assessable": "Not assessable", "limitations": "Limitations"},
        "sections": sections, "impression": ["Caries suspected; impacted teeth in the lower jaw."],
        "not_assessable": [f"{c} could not be assessed." for c in structured["summary"]["unparseable"]],
        "limitations": list(structured["analysis"]["limitations"]),
    }
    report.update(overrides)
    return report


class StructuredInputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.image = Path(self.tmp.name) / "img1.png"
        self.image.write_bytes(b"\x89PNG not a real image")
        self.result = dp.analyze_image(ScriptedRunner(SCRIPT), self.image, protocol=dp.Protocol())
        self.result["image_id"] = "img1"

    def tearDown(self):
        self.tmp.cleanup()

    def test_every_finding_task_and_cell_is_spelled_out(self):
        s = rw.structured_findings(self.result, analyzer="DentVLM")
        self.assertEqual(s["schema"], rw.SCHEMA)
        self.assertEqual([f["finding"] for f in s["findings"]], list(dp.CONDITIONS) + list(dp.EXTRA_TASKS))
        patient_cells = ["upper-right-posterior", "upper-anterior", "upper-left-posterior",
                         "lower-right-posterior", "lower-anterior", "lower-left-posterior"]
        for f in s["findings"]:
            self.assertIn(f["status"], rw.STATUSES)
            self.assertEqual(f["category"], rw.FINDING_CATEGORY[f["finding"]])
            self.assertNotIn(None, (f["count"], f["location_status"], f["detection"], f["multiplicity"]))
            if f["status"] != "not_assessed":
                self.assertEqual(list(f["regions"]), patient_cells)
                self.assertTrue(all(v in ("named", "not_named", "not_applicable") for v in f["regions"].values()))
        by = {f["finding"]: f for f in s["findings"]}
        filling = by["dental_filling"]
        self.assertEqual((filling["status"], filling["whole_image"], filling["detection"]), ("present", "present", "whole-image question"))
        self.assertEqual(filling["tasks"], [{"task": "fillings", "name": "Fillings", "question": q("fillings"), "answer": "present",
                                             "regions_named": ["upper-right-posterior"], "phrasings": 1}])
        self.assertEqual(filling["regions"]["upper-right-posterior"], "named")
        self.assertEqual(filling["regions"]["lower-anterior"], "not_named")
        self.assertEqual((filling["located_in"], filling["multiplicity"], filling["location_status"], filling["count"]),
                         (["upper-right-posterior"], 1, "located", "not_asked"))
        self.assertEqual((filling["trained"], filling["benchmark_class"], filling["countable"]), (True, True, True))
        restoration = by["prosthetic_restoration"]
        self.assertEqual([(t["task"], t["answer"]) for t in restoration["tasks"]],
                         [("prosthetic_crown", "absent"), ("prosthetic_bridge", "present")])
        self.assertEqual((restoration["status"], restoration["located_in"]), ("present", ["upper-anterior"]))
        impacted = by["impacted_tooth"]
        self.assertEqual((impacted["located_in"], impacted["multiplicity"]), (["lower-right-posterior", "lower-left-posterior"], 2))
        caries = by["carious_lesion"]
        self.assertEqual((caries["status"], caries["located_in"], caries["multiplicity"]), ("present", [], 0))
        self.assertTrue(caries["location_status"].startswith("not_stated"))
        implant = by["dental_implant"]
        self.assertEqual((implant["status"], implant["multiplicity"], set(implant["regions"].values())),
                         ("unparseable", "not_applicable", {"not_applicable"}))
        furcation = by["furcation_lesion"]
        self.assertEqual((furcation["status"], furcation["tasks"], furcation["regions"], furcation["detection"], furcation["count"]),
                         ("not_assessed", [], {}, "not_assessed", "not_countable"))
        crown = by["residual_crown"]
        self.assertEqual((crown["status"], crown["benchmark_class"], crown["category"], crown["label"]),
                         ("present", False, "teeth_and_eruption", "Residual Crown"))
        self.assertEqual((by["calculus"]["status"], by["calculus"]["category"]), ("absent", "periodontal"))
        self.assertEqual(s["summary"], {
            "present": ["carious_lesion", "impacted_tooth", "residual_crown", "prosthetic_restoration", "dental_filling"],
            "absent": ["periapical_lesion", "periodontal_bone_loss", "calculus", "insufficient_eruption_space", "root_fragment",
                       "endodontic_treatment"],
            "unparseable": ["dental_implant"],
            "not_assessed": ["furcation_lesion", "root_resorption", "apical_surgery", "orthodontic_device", "surgical_device"],
            "regional_only": []})
        self.assertEqual([r["name"] for r in s["analysis"]["regions"]], patient_cells)
        self.assertIn("the patient's right", s["analysis"]["regions"][0]["location"])
        self.assertIn("incisors and canines", s["analysis"]["regions"][1]["location"])
        self.assertEqual((s["analysis"]["analyzer"], s["analysis"]["questions_asked"], s["analysis"]["location_source"]),
                         ("DentVLM", 13, "rationale"))
        self.assertIn("13 tasks", s["analysis"]["method"])
        self.assertIn("rationale", s["analysis"]["method"])
        self.assertIn(rw.RATIONALE_LIMITATION, s["analysis"]["limitations"])
        self.assertIn("were not assessed", s["analysis"]["limitations"][-1])
        self.assertEqual([c["key"] for c in s["categories"]], list(rw.CATEGORIES))
        self.assertEqual(s["legend"]["regions"], rw.LEGEND["regions (location from the rationale)"])
        self.assertNotIn("model_text", filling["tasks"][0])
        json.dumps(s)  # serialisable

    def test_rationale_text_is_optional(self):
        s = rw.structured_findings(self.result, include_rationale=True)
        by = {f["finding"]: f for f in s["findings"]}
        self.assertEqual(by["dental_filling"]["tasks"][0]["model_text"], SCRIPT[q("fillings")])
        self.assertEqual(by["prosthetic_restoration"]["tasks"][0]["model_text"], "No\nNothing of the kind is seen.")

    def test_untrained_questions_and_counts(self):
        script = dict(SCRIPT)
        script[q("furcation_lesion")] = "Yes\nFurcation involvement is seen."
        script[dp.COUNT_QUESTIONS["dental_filling"]] = "There are 2 teeth with fillings."
        result = dp.analyze_image(ScriptedRunner(script), self.image, protocol=dp.Protocol(ask_untrained=True, count_question=True))
        s = rw.structured_findings(result)
        by = {f["finding"]: f for f in s["findings"]}
        self.assertEqual((by["furcation_lesion"]["status"], by["furcation_lesion"]["trained"]), ("present", False))
        self.assertEqual(by["furcation_lesion"]["tasks"][0]["question"], dp.questions_for("furcation_lesion")[0])
        self.assertEqual((by["dental_filling"]["count"], by["impacted_tooth"]["count"], by["carious_lesion"]["count"]), (2, 0, 0))
        self.assertEqual((by["endodontic_treatment"]["count"], by["periodontal_bone_loss"]["count"]), ("not_asked", "not_countable"))
        self.assertEqual(s["summary"]["not_assessed"], [])
        self.assertNotIn("were not assessed", s["analysis"]["limitations"][-1])
        self.assertIn("out-of-distribution tooth-count", s["analysis"]["method"])

    def test_crop_locations(self):
        if importlib.util.find_spec("PIL") is None:
            self.skipTest("Pillow not installed")
        from PIL import Image, ImageDraw

        image = Image.new("L", (560, 280), color=128)  # each cell gets its own mark, so the crops differ as bytes
        draw = ImageDraw.Draw(image)
        for index, (left, top, _, _) in enumerate(dp.CELL_WINDOWS.values()):
            x, y = int(left * 560) + 5 + index * 3, int(top * 280) + 5
            draw.rectangle((x, y, x + 12, y + 12), fill=20 + 30 * index)
        image.save(self.image)
        cell_of = {png: cell for cell, png in dp.make_crops(self.image).items()}
        self.assertEqual(len(cell_of), 6)
        script = dict(SCRIPT)
        script[("fillings", "upper-left")] = "Yes\nFillings are visible."
        script[("impacted_tooth", "lower-right")] = "Yes"
        script[("root_canal_therapy", "upper-right")] = "Yes"  # missed on the whole image, recovered in a cell
        script[("implant", "lower-anterior")] = "Yes and no."  # unparseable cell
        result = dp.analyze_image(ScriptedRunner(script, cell_of), self.image, protocol=dp.Protocol(location="crops"))
        s = rw.structured_findings(result)
        by = {f["finding"]: f for f in s["findings"]}
        self.assertEqual(s["analysis"]["location_source"], "crops")
        self.assertEqual(s["legend"]["regions"], rw.LEGEND["regions (location from cell crops)"])
        filling = by["dental_filling"]
        self.assertEqual(filling["regions"], {"upper-right-posterior": "present", "upper-anterior": "absent", "upper-left-posterior": "absent",
                                              "lower-right-posterior": "absent", "lower-anterior": "absent", "lower-left-posterior": "absent"})
        self.assertEqual((filling["located_in"], filling["region_source"], filling["detection"]),
                         (["upper-right-posterior"], "cell_crops", "whole-image and cell-crop answers agree"))
        endo = by["endodontic_treatment"]
        self.assertEqual((endo["status"], endo["whole_image"], endo["located_in"]), ("present", "absent", ["upper-left-posterior"]))
        self.assertIn("cell crops only", endo["detection"])
        caries = by["carious_lesion"]
        self.assertEqual((caries["status"], caries["whole_image"]), ("absent", "present"))
        self.assertIn("discordant", caries["detection"])
        implant = by["dental_implant"]
        self.assertEqual((implant["status"], implant["regions"]["lower-anterior"]), ("unparseable", "unparseable"))
        self.assertEqual(s["summary"]["regional_only"], ["endodontic_treatment"])
        self.assertNotIn(rw.RATIONALE_LIMITATION, s["analysis"]["limitations"])

    def test_prompt_is_filled_in(self):
        s = rw.structured_findings(self.result, analyzer="DentVLM")
        prompt = rw.user_prompt(s, "Persian")
        self.assertIn("in Persian", prompt)
        self.assertIn("DentVLM", prompt)
        self.assertIn('"finding": "dental_filling"', prompt)
        self.assertIn(rw.OUTPUT_SCHEMA, prompt)
        for placeholder in ("{language}", "{findings_json}", "{analyzer}", "{method}", "{output_schema}"):
            self.assertNotIn(placeholder, prompt)


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.image = Path(self.tmp.name) / "img1.png"
        self.image.write_bytes(b"\x89PNG")
        self.result = dp.analyze_image(ScriptedRunner(SCRIPT), self.image, protocol=dp.Protocol())
        self.structured = rw.structured_findings(self.result)

    def tearDown(self):
        self.tmp.cleanup()

    def test_faithful_report_passes(self):
        self.assertEqual(rw.verify_report(good_report(self.structured), self.structured), [])

    def test_extract_json(self):
        report = good_report(self.structured)
        text = json.dumps(report)
        self.assertEqual(rw.extract_json(text), report)
        self.assertEqual(rw.extract_json(f"Here it is:\n```json\n{text}\n```\nDone."), report)
        self.assertEqual(rw.extract_json(f"Sure. {text} Let me know."), report)
        self.assertIsNone(rw.extract_json("no json here"))
        self.assertIsNone(rw.extract_json("[1, 2]"))

    def test_problems_are_named(self):
        s = self.structured
        self.assertEqual(rw.verify_report(None, s), ["the reply is not a JSON object"])
        self.assertIn("missing key 'sections'", rw.verify_report({"title": "x"}, s))
        report = good_report(s)
        report["sections"][0]["findings"][0]["status"] = "absent"
        self.assertIn("dental_filling: status must stay 'present', got 'absent'", rw.verify_report(report, s))
        report = good_report(s)
        report["sections"][3]["findings"][1]["status"] = "present"  # calculus was absent
        self.assertIn("calculus: status must stay 'absent', got 'present'", rw.verify_report(report, s))
        report = good_report(s)
        report["sections"][0]["findings"].append({"finding": "cyst", "status": "present", "statement": "x"})
        self.assertIn("unknown finding 'cyst': only the findings in the data may appear", rw.verify_report(report, s))
        report = good_report(s)
        moved = report["sections"][2]["findings"].pop(0)  # caries into the periodontal section
        report["sections"][3]["findings"].append(moved)
        self.assertIn("carious_lesion belongs in section 'caries', not 'periodontal'", rw.verify_report(report, s))
        report = good_report(s)
        report["sections"][0]["findings"].append(dict(report["sections"][0]["findings"][0]))
        self.assertIn("findings listed more than once: dental_filling", rw.verify_report(report, s))
        report = good_report(s)
        del report["sections"][1]
        self.assertIn("missing findings: endodontic_treatment, apical_surgery", rw.verify_report(report, s))
        self.assertIn("'impression' must be a list of 1 to 8 non-empty strings", rw.verify_report(good_report(s, impression=[]), s))
        self.assertIn("'not_assessable' must name the unparseable findings: dental_implant",
                      rw.verify_report(good_report(s, not_assessable=[]), s))
        self.assertIn("'limitations' must be a non-empty list of strings", rw.verify_report(good_report(s, limitations=[]), s))
        report = good_report(s)
        report["sections"][0]["findings"][0]["statement"] = "  "
        self.assertIn("dental_filling: 'statement' must be a non-empty string", rw.verify_report(report, s))
        report = good_report(s)
        report["sections"][0]["category"] = "misc"
        problems = rw.verify_report(report, s)
        self.assertIn("a section has an unknown category: 'misc'", problems)
        self.assertIn("missing findings: dental_implant, prosthetic_restoration, dental_filling", problems)
        report = good_report(s, headings={"image": "Image"})
        self.assertTrue(any(p.startswith("'headings' must hold") for p in rw.verify_report(report, s)))

    def test_not_assessable_must_be_empty_without_unparseable_findings(self):
        clean = dp.analyze_image(ScriptedRunner({q("fillings"): f"Yes\nFillings in {UPPER_LEFT}."}), self.image, protocol=dp.Protocol())
        s = rw.structured_findings(clean)
        self.assertEqual(s["summary"]["unparseable"], [])
        self.assertIn("'not_assessable' must be empty: no finding was unparseable",
                      rw.verify_report(good_report(s, not_assessable=["x"]), s))
        self.assertEqual(rw.verify_report(good_report(s), s), [])


class WriterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        image = self.root / "img1.png"
        image.write_bytes(b"\x89PNG")
        self.result = dp.analyze_image(ScriptedRunner(SCRIPT), image, protocol=dp.Protocol())
        self.result["image_id"] = "img1"
        self.structured = rw.structured_findings(self.result, "DentVLM")

    def tearDown(self):
        self.tmp.cleanup()

    def _writer(self, texts, **options):
        client = FakeClient(texts)
        spec = {"provider": "openai", "model": "gpt-5", "api_key": "sk-test", "base_url": "https://api.openai.com/v1",
                "token_param": "max_completion_tokens",
                "temperature": None, "max_output_tokens": 2048, **options}
        return rw.ReportWriter.from_api(spec, language="English", client=client), client

    def test_verified_report_and_markdown(self):
        writer, client = self._writer([json.dumps(good_report(self.structured))])
        payload = writer.write(self.result, "DentVLM")
        self.assertTrue(payload["verified"])
        self.assertEqual((payload["problems"], len(payload["attempts"]), payload["language"]), ([], 1, "English"))
        self.assertEqual(payload["structured"], self.structured)
        request = client.requests[0]
        self.assertEqual([m["role"] for m in request["messages"]], ["system", "user"])
        self.assertEqual(request["messages"][0]["content"], rw.SYSTEM_PROMPT)
        self.assertEqual(request["messages"][1]["content"], payload["prompt"])
        self.assertEqual((request["model"], request["max_completion_tokens"]), ("gpt-5", 2048))
        self.assertNotIn("temperature", request)
        md = payload["markdown"]
        self.assertTrue(md.startswith("# Panoramic radiograph: automated findings report"))
        self.assertIn("**Image:** img1.png", md)
        self.assertIn("### Restorations and prostheses\n- ● Statement about dental_filling.\n- ● Statement about prosthetic_restoration.\n"
                      "- ? Statement about dental_implant.", md)
        self.assertIn("### Periodontal status\n- ○ Statement about periodontal_bone_loss.\n- ○ Statement about calculus.\n"
                      "- – Statement about furcation_lesion.", md)
        self.assertIn("## Impression\n- Caries suspected; impacted teeth in the lower jaw.", md)
        self.assertIn("## Not assessable\n- dental_implant could not be assessed.", md)
        self.assertIn("## Limitations\n- Experimental output", md)
        self.assertIn("*DentVLM · 13 questions → gpt-5*", md)
        self.assertNotIn("sk-test", json.dumps(payload))
        shuffled = good_report(self.structured)
        shuffled["sections"].reverse()
        self.assertEqual(rw.render_markdown(shuffled, self.structured, "gpt-5"), md)

    def test_repair_turn_fixes_an_unfaithful_reply(self):
        bad = good_report(self.structured)
        bad["sections"][0]["findings"][0]["status"] = "absent"
        del bad["sections"][4]
        writer, client = self._writer([json.dumps(bad), json.dumps(good_report(self.structured))])
        payload = writer.write(self.result, "DentVLM")
        self.assertTrue(payload["verified"])
        self.assertEqual(len(payload["attempts"]), 2)
        self.assertEqual(payload["attempts"][0]["problems"],
                         ["dental_filling: status must stay 'present', got 'absent'", "missing findings: periapical_lesion"])
        self.assertEqual(payload["attempts"][1]["problems"], [])
        repair = client.requests[1]["messages"]
        self.assertEqual([m["role"] for m in repair], ["system", "user", "assistant", "user"])
        self.assertEqual(repair[2]["content"], json.dumps(bad))
        self.assertIn("- missing findings: periapical_lesion", repair[3]["content"])
        self.assertTrue(repair[3]["content"].startswith("Your reply failed these checks"))

    def test_fallback_when_the_model_keeps_failing(self):
        writer, client = self._writer(["I cannot do that.", "{\"title\": \"x\"}"])
        payload = writer.write(self.result, "DentVLM")
        self.assertFalse(payload["verified"])
        self.assertIsNone(payload["report"])
        self.assertEqual(len(payload["attempts"]), 2)
        self.assertEqual(payload["problems"][0], "missing key 'headings'")
        self.assertTrue(payload["markdown"].startswith("# Automatic summary (the report model's reply failed verification)"))
        self.assertIn("Dental filling; in 1 region(s): patient's upper right posterior (image left)", payload["markdown"])
        self.assertIn("Not assessed by this model: Furcation involvement", payload["markdown"])
        self.assertEqual(len(client.requests), 2)
        writer, client = self._writer(["nope"], repairs=0)
        payload = writer.write(self.result)
        self.assertEqual((payload["verified"], len(client.requests)), (False, 1))

    def test_truncated_reply_is_reported(self):
        client = FakeClient(["{\"title\": \"cut"], finish_reason="length")
        writer = rw.ReportWriter(None, "k", "m", client=client, repairs=0)
        payload = writer.write(self.result)
        self.assertEqual(payload["problems"], ["the reply is not a JSON object", "the reply was cut off by max_output_tokens"])

    def test_spec_options_and_names(self):
        writer, _ = self._writer([])
        self.assertEqual((writer.name, writer.run_name, writer.include_rationale), ("report-gpt-5", "report-gpt-5-english", False))
        persian = rw.ReportWriter(None, "k", "openai/gpt-oss-120b", language="Persian (Farsi)", client=FakeClient([]))
        self.assertEqual(persian.run_name, "report-openai-gpt-oss-120b-persian-farsi")
        self.assertNotIn("sk-test", json.dumps(writer.settings()))
        self.assertIn("system_prompt", writer.settings())
        self.assertNotIn("system_prompt", writer.public())
        spec = {"provider": "nvidia", "model": "m", "api_key": "nv", "base_url": "https://integrate.api.nvidia.com/v1",
                "language": "German", "repairs": 2, "include_rationale": True}
        german = rw.ReportWriter.from_api(spec, client=FakeClient([]))
        self.assertEqual((german.language, german.repairs, german.include_rationale, german.base_url),
                         ("German", 2, True, "https://integrate.api.nvidia.com/v1"))
        self.assertEqual(rw.ReportWriter.from_api(spec, language="Turkish", client=FakeClient([])).language, "Turkish")
        # With the rationale switched on, the model's own words reach the prompt.
        client = FakeClient([json.dumps(good_report(rw.structured_findings(self.result, "DentVLM", include_rationale=True)))])
        payload = rw.ReportWriter(None, "k", "m", include_rationale=True, client=client).write(self.result, "DentVLM")
        self.assertTrue(payload["verified"])
        self.assertIn('"model_text": ' + json.dumps(SCRIPT[q("fillings")], ensure_ascii=False), payload["prompt"])
        with self.assertRaises(ValueError):
            rw.ReportWriter(None, "k", "m", token_param="max_new_tokens", client=FakeClient([]))
        with self.assertRaises(ValueError):
            rw.ReportWriter(None, "k", "m", language=" ", client=FakeClient([]))

    def test_dataset_loop_resumes_and_guards_the_language(self):
        results = {"img1": self.result, "img2": {**self.result, "image_id": "img2"}}
        writer, client = self._writer([json.dumps(good_report(self.structured))] * 2)
        out = self.root / "reports" / writer.run_name
        reports = rw.report_dataset(writer, results, out, analyzer="DentVLM")
        self.assertEqual(sorted(reports), ["img1", "img2"])
        self.assertTrue((out / "manifest.json").is_file())
        self.assertTrue((out / "reports" / "img1.json").is_file() and (out / "reports" / "img2.md").is_file())
        self.assertEqual((out / "reports" / "img1.md").read_text(encoding="utf-8"), reports["img1"]["markdown"])
        self.assertEqual(rw.summarize_reports(reports),
                         {"images": 2, "verified": 2, "repaired": 0, "fallback": 0, "mean_completion_tokens": 300})
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual((manifest["writer"]["language"], manifest["writer"]["include_rationale"]), ("English", False))
        self.assertNotIn("sk-test", json.dumps(manifest))
        before = len(client.requests)
        rw.report_dataset(writer, results, out, analyzer="DentVLM")  # resume: nothing to do
        self.assertEqual(len(client.requests), before)
        limited = rw.report_dataset(writer, results, self.root / "limited", limit=1)
        self.assertEqual(list(limited), ["img1"])
        other = rw.ReportWriter.from_api({"provider": "openai", "model": "gpt-5", "api_key": "sk",
                                          "base_url": "https://api.openai.com/v1"}, language="German",
                                         client=FakeClient([]))
        with self.assertRaises(ValueError):
            rw.report_dataset(other, results, out)
        self.assertEqual(rw.load_reports(out).keys(), reports.keys())


if __name__ == "__main__":
    unittest.main()
