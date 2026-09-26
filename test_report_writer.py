"""Closed clinical report contract and historical evidence tests."""
import copy
import json
import pytest
import dental_pipeline as dp
import report_writer as rw
from pan_test_support import result, report, Client

@pytest.fixture
def structured(tmp_path):
    return rw.structured_findings(result(tmp_path, {"caries": "Yes\nCaries in the left posterior region of the upper dentition.",
                                                       "prosthetic_crown": "Yes", "prosthetic_bridge": "No"}))

def test_canonical_rows_and_separate_restorations(structured):
    rows = {f["finding"]: f for f in structured["findings"]}
    assert set(rows) == set(dp.TASKS) and len(rows) == 12
    assert rows["prosthetic_crown"]["status"] == "present"
    assert rows["prosthetic_bridge"]["status"] == "absent"
    assert all(f["label"] == dp.TASKS[f["finding"]]["name"] for f in rows.values())
    assert "count" not in json.dumps(rows)
    assert "patient's" not in rw.render_facts(structured)
    assert not rw.verify_report(report(structured), structured)

@pytest.mark.parametrize("mutation", ["finding", "status", "region", "impression", "duplicate", "missing", "prose", "category", "malformed", "malformed_category"])
def test_report_rejects_clinical_changes(structured, mutation):
    value = report(structured)
    row = value["sections"][0]["findings"][0]
    if mutation == "finding": row["finding"] = "periodontal_disease"
    elif mutation == "status": row["status"] = "invented"
    elif mutation == "region": row["regions"] = ["patient upper left"]
    elif mutation == "impression": value["impression"].append("prosthetic_bridge")
    elif mutation == "duplicate": value["sections"][0]["findings"].append(copy.deepcopy(row))
    elif mutation == "missing": value["sections"][0]["findings"].pop()
    elif mutation == "prose": row["statement"] = "New diagnosis"
    elif mutation == "category": value["sections"][0]["category"] = "invented"
    elif mutation == "malformed": row["finding"] = []
    elif mutation == "malformed_category": value["sections"][0]["category"] = []
    assert rw.verify_report(value, structured)

def test_reporter_never_receives_rationales(tmp_path):
    data = result(tmp_path, {"caries": "Yes\nSECRET INCIDENTAL PERIODONTAL DIAGNOSIS"})
    data["calls"][0]["question"] = "HISTORICAL EXACT QUESTION"
    structured = rw.structured_findings(data)
    prompt = rw.user_prompt(structured, "English")
    assert "SECRET" not in prompt and "HISTORICAL" not in prompt
    assert structured["evidence"][0]["calls"][0]["question"] == "HISTORICAL EXACT QUESTION"
    assert "SECRET" not in rw.render_facts(structured)

def test_valid_writer_and_saved_resume(tmp_path):
    data = result(tmp_path, {"caries": "Yes"})
    client = Client([json.dumps(report(rw.structured_findings(data)))])
    writer = rw.ReportWriter(None, "unused", "fake", client=client, repairs=0)
    outputs = rw.report_dataset(writer, {"image": data}, tmp_path / "reports")
    assert outputs["image"]["verified"]
    assert "Caries" in outputs["image"]["markdown"]
    rw.report_dataset(writer, {"image": data}, tmp_path / "reports")
    assert len(client.requests) == 1
    data["calls"][0]["question"] = "changed evidence"
    with pytest.raises(ValueError, match="changed"):
        rw.report_dataset(writer, {"image": data}, tmp_path / "reports")

@pytest.mark.parametrize("reply", ["garbage", RuntimeError("down"), ('{}', 'length')])
def test_writer_failure_keeps_deterministic_facts(tmp_path, reply):
    data = result(tmp_path, {"caries": "Yes"})
    writer = rw.ReportWriter(None, "unused", "fake", client=Client([reply]), repairs=0, api_call_retries=0)
    written = writer.write(data)
    assert not written["verified"]
    assert written["report"] is None
    assert written["markdown"] == rw.render_facts(rw.structured_findings(data))

@pytest.mark.parametrize("knob", ["include_rationale", "vote_agreement"])
def test_retired_report_options_fail(knob):
    with pytest.raises(ValueError):
        rw.ReportWriter(None, "unused", "fake", client=Client(), **{knob: True})
