"""Dataset projection, abstention denominators and legacy compatibility."""
import copy
import json
from pathlib import Path
import pytest
import dental_pipeline as dp
import dental_eval as ev
import dental_analysis as da
import benchmark_schema as bs
import report_writer as rw
from pan_test_support import result

def truth(boxes=None, annotated=None):
    return {"image":{"annotated":set(bs.UMFIH_CLASSES if annotated is None else annotated), "boxes":boxes or []}}

def box(condition, **extra):
    return {"condition":condition,"xc":.2,"yc":.2,"w":.05,"h":.05,**extra}

@pytest.mark.parametrize("crown,bridge,expected", [("No","No","no"),("No","?",None),("?","No",None),("?","?",None),("Yes","?","yes"),("No","Yes","yes")])
def test_combined_dataset_restoration_is_or_not_a_clinical_merge(tmp_path,crown,bridge,expected):
    data = result(tmp_path,{"prosthetic_crown":crown,"prosthetic_bridge":bridge})
    projected = bs.project_result(data)
    assert projected["findings"]["prosthetic_restoration"]["presence"] == expected
    assert "prosthetic_restoration" not in data["findings"]

def test_missing_annotation_file_is_not_negative(tmp_path):
    images=tmp_path/"images"; labels=tmp_path/"labels";images.mkdir();labels.mkdir()
    (images/"unlabeled.png").write_bytes(b"image")
    (images/"empty.png").write_bytes(b"image");(labels/"empty.txt").write_text("")
    loaded=ev.load_yolo(images,labels)
    assert loaded["unlabeled"]["annotated"] == set()
    assert loaded["empty"]["annotated"] == set(bs.UMFIH_CLASSES)

@pytest.mark.parametrize("class_id",range(14))
def test_fixed_yolo_class_order(tmp_path,class_id):
    images=tmp_path/"images";labels=tmp_path/"labels";images.mkdir();labels.mkdir()
    (images/"image.png").write_bytes(b"image")
    (labels/"image.txt").write_text(f"{class_id} .2 .3 .1 .1")
    expected=("dental_implant","prosthetic_restoration","dental_filling","endodontic_treatment","carious_lesion",
              "periodontal_bone_loss","impacted_tooth","periapical_lesion","root_fragment","furcation_lesion",
              "apical_surgery","root_resorption","orthodontic_device","surgical_device")
    assert ev.load_yolo(images,labels)["image"]["boxes"][0]["condition"] == expected[class_id]

def test_proxy_excluded_and_unsupported_never_scored(tmp_path):
    data=result(tmp_path,{"apical_periodontitis":"Yes","caries":"Yes","calculus":"Yes"})
    original=copy.deepcopy(data)
    scored=ev.evaluate(truth(),{"image":data})
    summary=scored["summary"]
    assert summary["FP"] == 1 and summary["negative_scored"] == 7
    assert summary["false_positive_rate"] == round(1/7,4)
    assert scored["proxy_presence"][0]["FP"] == 1
    assert len(scored["presence"]) == 8 and len(summary["not_assessed"]) == 6
    assert summary["without_benchmark_truth"] == ["residual_crown","insufficient_eruption_space","calculus"]
    assert data == original
    assert rw.structured_findings(data)["summary"]["present"] == ["apical_periodontitis","caries","calculus"]

def test_unknown_truth_counts_and_no_success_credit(tmp_path):
    data=result(tmp_path,{k:"?" for k in dp.TASKS})
    scored=ev.evaluate(truth([box("carious_lesion")]),{"image":data})["summary"]
    assert scored["unresolved_positive_truth"] == 1 and scored["unresolved_negative_truth"] == 6
    assert sum(scored[k] for k in ("TP","FP","TN","FN")) == 0
    assert scored["all_eligible_accuracy"] == scored["resolved_coverage"] == 0
    assert scored["false_positive_rate"] is None and scored["precision"] is None
    assert scored["finding_check_invariant_ok"]

def test_unannotated_class_has_no_score(tmp_path):
    data=result(tmp_path,{"calculus":"Yes"})
    scored=ev.evaluate(truth(annotated=[]),{"image":data})
    assert scored["presence"] == [] and scored["summary"]["all_eligible_accuracy"] is None

def test_dentex_uses_only_declared_annotation_classes(tmp_path):
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps({"images": [{"id": 1, "file_name": "image.png"}],
                               "categories_3": [{"id": 0, "name": "caries"}], "annotations": []}))
    assert ev.load_dentex(tmp_path, path)["image"]["annotated"] == {"carious_lesion"}

def test_unmentioned_regions_are_not_negative(tmp_path):
    data = result(tmp_path, {"caries": "Yes\nCaries in the left posterior region of the upper dentition."})
    cells = ev.predicted_cells(data, "carious_lesion")
    assert cells["upper-left"] is True
    assert all(value is None for key, value in cells.items() if key != "upper-left")
    assert "side unresolved" in dp.describe_cell("upper-left")

def test_fdi_has_priority_and_partial_truth_is_excluded(tmp_path):
    b=box("carious_lesion",fdi=(1,6),regions=["lower-right"],region_source="llm")
    assert ev.box_regions(b) == {ev.fdi_cell(1,6)} and ev.box_source(b) == "fdi"
    assert ev.apply_adapted(truth([b]),{})["image"]["boxes"][0] == b
    data=result(tmp_path,{"caries":"Yes\nCaries in the left posterior region of the upper dentition."})
    scored=ev.evaluate(truth([b,box("carious_lesion",location_excluded=True)]),{"image":data})
    row=next(r for r in scored["regions"] if r["condition"]=="carious_lesion")
    assert row["location_truth_excluded"] == 1 and row["mean_jaccard"] is None

def test_unlocated_positive_and_conditional_localization(tmp_path):
    data=result(tmp_path,{"caries":"Yes"})
    scored=ev.evaluate(truth([box("carious_lesion")]),{"image":data})
    row=next(r for r in scored["regions"] if r["condition"]=="carious_lesion")
    assert row["unlocated_cases"] == 1 and row["overall_localization_coverage"] == 0
    assert scored["summary"]["TP"] == 1

def test_legacy_read_preserves_questions_and_cannot_resume(tmp_path):
    from pan_test_support import Runner
    data=result(tmp_path)
    legacy=bs.project_result(data)
    legacy.pop("schema");legacy.pop("profile");legacy.pop("benchmark_projection")
    legacy["protocol"]={"location":"rationale","phrasings":3}
    legacy["calls"][0]["question"]="Original old question"
    out=tmp_path/"legacy"; (out/"results").mkdir(parents=True)
    (out/"results/image.json").write_text(json.dumps(legacy))
    loaded=dp.load_results(out)["image"]
    assert loaded["legacy_artifact"] and "schema" not in loaded
    assert rw.structured_findings(loaded)["evidence"][0]["calls"][0]["question"]=="Original old question"
    with pytest.raises(ValueError,match="Legacy or incompatible"):
        dp.run_dataset(Runner(),{"image":data["image"]},out)

def test_comparison_rejects_different_image_hashes(tmp_path):
    data=result(tmp_path)
    runs={}
    for name in ("old","new"):
        directory=tmp_path/name;(directory/"results").mkdir(parents=True)
        payload=copy.deepcopy(data)
        if name=="new":payload["image_sha256"]="different"
        (directory/"results/image.json").write_text(json.dumps(payload))
        (directory/"manifest.json").write_text(json.dumps({"protocol":data["protocol"]}))
        runs[name]=directory
    with pytest.raises(ValueError,match="image content differs"):
        da.compare_runs(truth(),runs,evaluate_location=False,counting=False)
