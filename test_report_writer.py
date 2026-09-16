"""Offline tests for the report writer: the dense structured input, the prompt, verification of the
model's JSON against that input, the repair turn, rendering, the fallback, and the resumable loop."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import dental_pipeline as dp
import report_writer as rw


class ScriptedRunner:
    """Answers by question text; unscripted presence answers are B, unscripted counts 0."""

    def __init__(self, script: dict):
        self.script, self.calls = script, 0

    def settings(self):
        return {"model": "fake"}

    def ask(self, image, question):
        self.calls += 1
        text = self.script.get(question, "B" if "A. True" in question else "0")
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


def q(condition, region=None):
    return dp.presence_question(condition, region=region)


def n(condition, region=None):
    return dp.count_question(condition, region=region)


SCRIPT = {
    q("dental_filling"): "A", q("dental_filling", "UR"): "A", n("dental_filling", "UR"): "2 teeth with fillings",
    q("endodontic_treatment", "UL"): "A", n("endodontic_treatment", "UL"): "1",  # missed on the whole image
    q("carious_lesion"): "A",  # whole-image alarm, every region B
    q("surgical_device"): "???",  # unparseable on the whole image, every region B
    q("dental_implant", "UR"): "???",  # unparseable region, no A: unparseable finding
    q("impacted_tooth"): "A", q("impacted_tooth", "LL"): "A", q("impacted_tooth", "LR"): "A",
    n("impacted_tooth", "LL"): "1", n("impacted_tooth", "LR"): "???",  # incomplete count
}


def good_report(structured: dict, language: str = "English", **overrides) -> dict:
    """A faithful report JSON for the structured findings, as a good model would return it."""
    sections = []
    for category in structured["categories"]:
        entries = [{"condition": c, "status": next(f["status"] for f in structured["findings"] if f["condition"] == c),
                    "statement": f"Statement about {c}."} for c in category["conditions"]]
        sections.append({"category": category["key"], "heading": category["label"], "findings": entries})
    report = {
        "language": language, "title": "Panoramic radiograph: automated findings report",
        "headings": {"image": "Image", "findings": "Findings", "impression": "Impression",
                     "not_assessable": "Not assessable", "limitations": "Limitations"},
        "sections": sections, "impression": ["Impacted teeth flagged in the lower jaw."],
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
        self.result = dp.analyze_image(ScriptedRunner(SCRIPT), self.image, mode="plain", protocol=dp.Protocol())
        self.result["image_id"] = "img1"

    def tearDown(self):
        self.tmp.cleanup()

    def test_every_finding_region_and_count_is_spelled_out(self):
        s = rw.structured_findings(self.result, analyzer="DentalGPT-7B")
        self.assertEqual(s["schema"], rw.SCHEMA)
        self.assertEqual((s["image"]["id"], s["image"]["file"]), ("img1", "img1.png"))
        self.assertEqual([f["condition"] for f in s["findings"]], list(dp.CONDITIONS))
        for f in s["findings"]:
            self.assertIn(f["status"], rw.STATUSES)
            self.assertEqual(list(f["regions"]), ["UR", "UL", "LL", "LR"])
            self.assertTrue(all(v in ("present", "absent", "unparseable", "not_asked") for v in f["regions"].values()))
            self.assertEqual(f["category"], rw.CONDITION_CATEGORY[f["condition"]])
            self.assertNotIn(None, (f["count"], f["location_status"], f["detection"]))
        by = {f["condition"]: f for f in s["findings"]}
        filling = by["dental_filling"]
        self.assertEqual((filling["status"], filling["whole_image"], filling["count"], filling["located_in"]),
                         ("present", "present", 2, ["UR"]))
        self.assertEqual(filling["regions"], {"UR": "present", "UL": "absent", "LL": "absent", "LR": "absent"})
        self.assertEqual(filling["region_counts"], {"UR": 2, "UL": "not_asked", "LL": "not_asked", "LR": "not_asked"})
        self.assertEqual((filling["detection"], filling["location_status"], filling["count_source"]),
                         ("whole-image and regional questions agree", "located", "sum_of_region_counts"))
        endo = by["endodontic_treatment"]
        self.assertEqual((endo["status"], endo["whole_image"], endo["count"], endo["located_in"]), ("present", "absent", 1, ["UL"]))
        self.assertIn("regional questions only", endo["detection"])
        caries = by["carious_lesion"]
        self.assertEqual((caries["status"], caries["whole_image"], caries["count"], caries["location_status"]),
                         ("absent", "present", "not_asked", "not_applicable"))
        self.assertIn("discordant", caries["detection"])
        self.assertEqual((by["surgical_device"]["status"], by["surgical_device"]["whole_image"], by["surgical_device"]["count"]),
                         ("absent", "unparseable", "not_countable"))
        implant = by["dental_implant"]
        self.assertEqual((implant["status"], implant["regions"]["UR"], implant["count"]), ("unparseable", "unparseable", "not_asked"))
        impacted = by["impacted_tooth"]
        self.assertEqual((impacted["count"], impacted["located_in"]), ("incomplete", ["LL", "LR"]))
        self.assertEqual(impacted["region_counts"], {"UR": "not_asked", "UL": "not_asked", "LL": 1, "LR": "unparseable"})
        self.assertFalse(by["periodontal_bone_loss"]["countable"])
        self.assertEqual(s["summary"], {"present": ["impacted_tooth", "dental_filling", "endodontic_treatment"],
                                        "absent": [c for c in rw.PATHOLOGY + rw.TREATMENT
                                                   if c not in ("impacted_tooth", "dental_filling", "endodontic_treatment", "dental_implant")],
                                        "unparseable": ["dental_implant"], "regional_only": ["endodontic_treatment"]})
        self.assertEqual([r["name"] for r in s["analysis"]["regions"]], ["UR", "UL", "LL", "LR"])
        self.assertIn("patient's right", s["analysis"]["regions"][0]["location"])
        self.assertIn("each of the 4 regions", s["analysis"]["method"])
        self.assertEqual((s["analysis"]["analyzer"], s["analysis"]["questions_asked"]), ("DentalGPT-7B", self.result["call_count"]))
        self.assertEqual([c["key"] for c in s["categories"]], list(rw.CATEGORIES))
        json.dumps(s)  # serialisable

    def test_whole_image_only_protocol(self):
        flat = dp.Protocol(presence_level="overall", count_level="overall")
        result = dp.analyze_image(ScriptedRunner({q("dental_filling"): "A", n("dental_filling"): "3 teeth"}), self.image,
                                  protocol=flat)
        s = rw.structured_findings(result)
        by = {f["condition"]: f for f in s["findings"]}
        self.assertEqual((s["analysis"]["region_scheme"], s["analysis"]["regions"]), ("none", []))
        self.assertEqual((by["dental_filling"]["count"], by["dental_filling"]["count_source"], by["dental_filling"]["regions"]),
                         (3, "whole_image_question", {}))
        self.assertEqual((by["dental_filling"]["detection"], by["dental_filling"]["location_status"]),
                         ("whole-image question", "not_asked"))
        self.assertEqual((by["carious_lesion"]["count"], by["carious_lesion"]["region_counts"]), ("not_asked", {}))
        self.assertEqual(s["summary"]["regional_only"], [])

    def test_counting_off_protocol(self):
        result = dp.analyze_image(ScriptedRunner(SCRIPT), self.image, protocol=dp.Protocol(counting=False))
        s = rw.structured_findings(result)
        by = {f["condition"]: f for f in s["findings"]}
        # Nothing is countable in a run without count questions; the presence and region fields are untouched.
        self.assertTrue(all(not f["countable"] and f["count"] == "not_countable" and f["count_source"] == "none"
                            and f["region_counts"] == {} for f in s["findings"]))
        self.assertEqual((by["dental_filling"]["status"], by["dental_filling"]["located_in"]), ("present", ["UR"]))
        self.assertEqual((by["endodontic_treatment"]["status"], by["endodontic_treatment"]["located_in"]), ("present", ["UL"]))
        self.assertEqual((by["impacted_tooth"]["located_in"], by["impacted_tooth"]["count"]), (["LL", "LR"], "not_countable"))
        self.assertEqual(s["analysis"]["protocol"]["counting"], False)
        self.assertIn("no count question: presence only", s["analysis"]["method"])
        self.assertNotIn("count of affected teeth", s["analysis"]["method"])
        self.assertEqual(rw.verify_report(good_report(s), s), [])

    def test_prompt_is_filled_in(self):
        s = rw.structured_findings(self.result, analyzer="DentalGPT-7B")
        prompt = rw.user_prompt(s, "Persian")
        self.assertIn("in Persian", prompt)
        self.assertIn("DentalGPT-7B", prompt)
        self.assertIn('"condition": "dental_filling"', prompt)
        self.assertIn(rw.OUTPUT_SCHEMA, prompt)
        for placeholder in ("{language}", "{findings_json}", "{analyzer}", "{method}", "{output_schema}"):
            self.assertNotIn(placeholder, prompt)


class MethodTextTests(unittest.TestCase):
    def test_method_text_names_the_combined_question(self):
        regions = ("UR", "UL", "LL", "LR")
        base = {"presence_level": "region", "count_level": "region", "region_scheme": "quadrant"}
        self.assertIn("every region that answered True", rw.method_text({**base, "question_form": "separate"}, regions))
        self.assertIn("every region that answered True", rw.method_text(base, regions))  # results saved before the form knob
        self.assertIn("the regional question also asks for the count of affected teeth", rw.method_text({**base, "question_form": "combined"}, regions))
        self.assertIn("one presence-and-count question in each of the 4 regions (named in the question)",
                      rw.method_text({**base, "presence_level": "overall", "question_form": "combined"}, regions))
        flat = rw.method_text({**base, "presence_level": "overall", "count_level": "overall", "question_form": "combined"}, ())
        self.assertIn("the whole-image question also asks for the count of affected teeth", flat)
        self.assertNotIn("separate whole-image count", flat)
        self.assertIn("with a separate whole-image count for a finding the whole image answered False but a region answered True",
                      rw.method_text({**base, "count_level": "overall", "question_form": "combined"}, regions))
        self.assertIn("one whole-image count of affected teeth", rw.method_text({**base, "count_level": "overall"}, regions))
        off = rw.method_text({**base, "counting": False, "question_form": "combined"}, regions)
        self.assertIn("each of the 4 regions (named in the question, on the same whole image); "
                      "no count question: presence only", off)
        self.assertNotIn("count of affected teeth", off)


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        image = Path(self.tmp.name) / "img1.png"
        image.write_bytes(b"\x89PNG")
        self.result = dp.analyze_image(ScriptedRunner(SCRIPT), image, protocol=dp.Protocol())
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
        report["sections"][0]["findings"].append({"condition": "cyst", "status": "present", "statement": "x"})
        self.assertIn("unknown finding 'cyst': only the conditions in the data may appear", rw.verify_report(report, s))
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
        clean = dp.analyze_image(ScriptedRunner({q("dental_filling"): "A", q("dental_filling", "UR"): "A",
                                                 n("dental_filling", "UR"): "2 teeth"}),
                                 Path(self.tmp.name) / "img1.png", protocol=dp.Protocol())
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
        self.structured = rw.structured_findings(self.result, "DentalGPT-7B")

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
        payload = writer.write(self.result, "DentalGPT-7B")
        self.assertTrue(payload["verified"])
        self.assertEqual((payload["problems"], len(payload["attempts"]), payload["language"]), ([], 1, "English"))
        self.assertEqual(payload["report"]["sections"][0]["category"], "restorations_and_prostheses")
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
        self.assertIn("### Restorations and prostheses\n- ● Statement about dental_filling.\n- ○ Statement about prosthetic_restoration.", md)
        self.assertIn("- ? Statement about dental_implant.", md)
        self.assertIn("## Impression\n- Impacted teeth flagged in the lower jaw.", md)
        self.assertIn("## Not assessable\n- dental_implant could not be assessed.", md)
        self.assertIn("## Limitations\n- Experimental output", md)
        self.assertIn("*DentalGPT-7B · 74 questions → gpt-5*", md)
        self.assertNotIn("sk-test", json.dumps(payload))
        # Sections are rendered in the fixed order whatever order the model used.
        shuffled = good_report(self.structured)
        shuffled["sections"].reverse()
        self.assertEqual(rw.render_markdown(shuffled, self.structured, "gpt-5"), md)

    def test_repair_turn_fixes_an_unfaithful_reply(self):
        bad = good_report(self.structured)
        bad["sections"][0]["findings"][0]["status"] = "absent"
        del bad["sections"][4]
        writer, client = self._writer([json.dumps(bad), json.dumps(good_report(self.structured))])
        payload = writer.write(self.result, "DentalGPT-7B")
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
        payload = writer.write(self.result, "DentalGPT-7B")
        self.assertFalse(payload["verified"])
        self.assertIsNone(payload["report"])
        self.assertEqual(len(payload["attempts"]), 2)
        self.assertEqual(payload["problems"][0], "missing key 'headings'")
        self.assertTrue(payload["markdown"].startswith("# Automatic summary (the report model's reply failed verification)"))
        self.assertIn("Dental filling; count 2 (UR 2); location UR", payload["markdown"])
        self.assertEqual(len(client.requests), 2)
        # No repair turn at all when repairs=0.
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
        self.assertEqual((writer.name, writer.run_name), ("report-gpt-5", "report-gpt-5-english"))
        persian = rw.ReportWriter(None, "k", "openai/gpt-oss-120b", language="Persian (Farsi)", client=FakeClient([]))
        self.assertEqual(persian.run_name, "report-openai-gpt-oss-120b-persian-farsi")
        self.assertNotIn("sk-test", json.dumps(writer.settings()))
        self.assertIn("system_prompt", writer.settings())
        self.assertNotIn("system_prompt", writer.public())
        spec = {"provider": "nvidia", "model": "m", "api_key": "nv", "base_url": "https://integrate.api.nvidia.com/v1",
                "language": "German", "repairs": 2}
        german = rw.ReportWriter.from_api(spec, client=FakeClient([]))
        self.assertEqual((german.language, german.repairs, german.base_url), ("German", 2, "https://integrate.api.nvidia.com/v1"))
        self.assertEqual(rw.ReportWriter.from_api(spec, language="Turkish", client=FakeClient([])).language, "Turkish")
        with self.assertRaises(ValueError):
            rw.ReportWriter(None, "k", "m", token_param="max_new_tokens", client=FakeClient([]))
        with self.assertRaises(ValueError):
            rw.ReportWriter(None, "k", "m", language=" ", client=FakeClient([]))

    def test_dataset_loop_resumes_and_guards_the_language(self):
        results = {"img1": self.result, "img2": {**self.result, "image_id": "img2"}}
        writer, client = self._writer([json.dumps(good_report(self.structured))] * 2)
        out = self.root / "reports" / writer.run_name
        reports = rw.report_dataset(writer, results, out, analyzer="DentalGPT-7B")
        self.assertEqual(sorted(reports), ["img1", "img2"])
        self.assertTrue((out / "manifest.json").is_file())
        self.assertTrue((out / "reports" / "img1.json").is_file() and (out / "reports" / "img2.md").is_file())
        self.assertEqual((out / "reports" / "img1.md").read_text(encoding="utf-8"), reports["img1"]["markdown"])
        self.assertEqual(rw.summarize_reports(reports),
                         {"images": 2, "verified": 2, "repaired": 0, "fallback": 0, "mean_completion_tokens": 300})
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["writer"]["language"], "English")
        self.assertNotIn("sk-test", json.dumps(manifest))
        before = len(client.requests)
        rw.report_dataset(writer, results, out, analyzer="DentalGPT-7B")  # resume: nothing to do
        self.assertEqual(len(client.requests), before)
        # A limit reports only the first images; a different language cannot reuse the directory.
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
