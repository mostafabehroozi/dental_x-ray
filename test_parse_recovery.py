"""Canonical requests, deterministic abstentions and safe resumability."""
import copy
import json
from pathlib import Path
from unittest.mock import patch
import pytest
import dental_pipeline as dp
from pan_test_support import Client, Runner, result

@pytest.mark.parametrize("task", list(dp.TASKS))
def test_source_record_exact_question(task):
    fixture = json.loads(Path("tests_fixtures/pan_source_questions.json").read_text())
    spec = dp.TASKS[task]
    record = next(r for r in fixture["records"] if r["id"] == spec["question_provenance"]["record_id"])
    assert spec["questions"] == [record["question"].removeprefix("<image>")]
    assert spec["question_provenance"]["revision"] == fixture["source_revision"]
    assert spec["question_provenance"]["source_modality"] == ("UPP" if task == "calculus" else "PAN")

def test_twelve_independent_exact_requests(tmp_path):
    image = tmp_path / "full.png"; image.write_bytes(b"FULL ORIGINAL PAN")
    client = Client()
    runner = dp.VisionRunner(client=client)
    output = dp.analyze_image(runner, image)
    assert set(output["findings"]) == set(dp.TASKS)
    assert output["call_count"] == len(client.requests) == 12
    for request, spec, call in zip(client.requests, dp.TASKS.values(), output["calls"]):
        assert request["messages"][0] == {"role":"system", "content":dp.SYSTEM_MESSAGE}
        assert len(request["messages"]) == 2
        content = request["messages"][1]["content"]
        assert content[0]["image_url"]["url"] == dp.image_data_uri(image)
        assert content[1] == {"type":"text", "text":spec["questions"][0]}
        assert call["question"] == spec["questions"][0]
        assert request["temperature"] == .1 and request["top_p"] == .001 and request["max_tokens"] == 512
        assert request["extra_body"]["samplers"] == ["penalties", "temperature", "top_p"]
        assert request["extra_body"]["repeat_last_n"] == -1

@pytest.mark.parametrize("text", ["", "Perhaps", "Yes and no", "No or yes", "Yes\nNo, not present", "No\nYes, present", "Answer: Yes", "```Yes```"])
def test_unreadable_or_contradictory_answers_are_not_retried(tmp_path, text):
    data = result(tmp_path, {"caries": text})
    assert data["findings"]["caries"]["presence"] is None
    assert data["call_count"] == 12
    assert data["parse_recovery"]["retry_calls"] == 0
    assert data["findings"]["caries"]["raw_response"] == text

@pytest.mark.parametrize("reply", [{"text":"Yes", "finish_reason":"length"},
                                    {"text":"No", "truncated":True}, RuntimeError("server failed")])
def test_truncation_and_transport_failure_are_unresolved(tmp_path, reply):
    data = result(tmp_path, {"caries":reply})
    assert data["findings"]["caries"]["presence"] is None
    assert data["findings"]["caries"]["parse_error"]
    assert data["findings"]["implant"]["presence"] == "no"

def test_identical_transport_retry(tmp_path):
    client = Client([ConnectionError("connection reset"), "Yes"])
    runner = dp.VisionRunner(client=client, api_call_retries=1)
    with patch("llm_api.time.sleep"):
        reply = runner.ask(b"image", dp.questions_for("caries")[0])
    assert reply["text"] == "Yes"
    assert len(client.requests) == 2 and client.requests[0] == client.requests[1]

def test_positive_without_region_remains_positive(tmp_path):
    data = result(tmp_path, {"caries":"Yes\nCaries is present. Implant is also visible."})
    assert data["findings"]["caries"]["presence"] == "yes"
    assert data["findings"]["caries"]["location_status"] == "not_stated"
    assert data["findings"]["implant"]["presence"] == "no"

@pytest.mark.parametrize("rationale", ["There is no evidence of caries.", "Caries is not detected.", "Caries is absent."])
def test_positive_with_explicit_global_denial_abstains(tmp_path, rationale):
    data = result(tmp_path, {"caries": "Yes\n" + rationale})
    assert data["findings"]["caries"]["presence"] is None
    assert data["findings"]["caries"]["parse_error"] == "contradictory_response"

@pytest.mark.parametrize("descriptor", list(dp.DESCRIPTORS))
def test_all_nine_source_descriptors_are_preserved(tmp_path, descriptor):
    data = result(tmp_path, {"caries":"Yes\nCaries is visible in " + descriptor + "."})
    task = data["tasks"]["caries"]
    assert set(task["regions"]) == set(dp.DESCRIPTORS[descriptor])
    assert task["report_regions"] == task["regions"]
    assert task["patient_laterality"] == "unresolved"
    assert task["location_matches"][0]["text"] == descriptor

@pytest.mark.parametrize("phrase", ["No caries in", "Caries is absent in", "There may be caries in", "An implant is present in", "A crown is present in", "A root canal filling is present in"])
def test_negated_uncertain_or_other_task_regions_are_audit_only(tmp_path, phrase):
    descriptor = next(iter(dp.DESCRIPTORS))
    data = result(tmp_path, {"caries":"Yes.\nA finding is visible. " + phrase + " " + descriptor + "."})
    task = data["tasks"]["caries"]
    # A separate rationale clause is not a second diagnosis decision.
    assert task["presence"] == "yes"
    assert task["regions"]
    assert not task["report_regions"]

def test_resume_hashes_image_registry_and_runtime(tmp_path):
    image = tmp_path / "image.png"; image.write_bytes(b"one")
    runner = Runner()
    directory = dp.run_dataset(runner, {"image":image}, tmp_path / "out")
    dp.run_dataset(runner, {"image":image}, directory)
    assert len(runner.requests) == 12
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["registry"] and manifest["parser_version"] == dp.PARSER_VERSION
    with patch.dict(dp.TASKS["caries"], {"questions":["Changed question"]}):
        with pytest.raises(ValueError, match="different configuration"):
            dp.run_dataset(runner, {"image":image}, directory)
    image.write_bytes(b"two")
    with pytest.raises(ValueError, match="source image changed"):
        dp.run_dataset(runner, {"image":image}, directory)

def test_manifest_changes_with_runtime_or_parser():
    baseline = dp.run_config(dp.Protocol(), {"runtime":"one"})["hash"]
    assert baseline != dp.run_config(dp.Protocol(), {"runtime":"two"})["hash"]
    with patch.object(dp, "PARSER_VERSION", 99):
        assert baseline != dp.run_config(dp.Protocol(), {"runtime":"one"})["hash"]
