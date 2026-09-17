"""Deterministic evaluation of saved pipeline results against box-level ground truth.

Ground truth comes from YOLO label files (UMFIH 14-class set) or DENTEX JSON.
Metrics stay simple: image-level TP/FP/TN/FN per finding (and the same table
for the whole-image answers alone in the region comparison, to show what the
region questions recovered and what they cost), presence per cell (every cell of every
image, present or absent, against the cells the true boxes occupy, so a finding
class is scored once per cell rather than counted), cell-level TP/FP/TN/FN for
the localized true positives, and two per-image numbers a dentist cares about
(complete-case rate, false alarms).

Findings the model was not asked about are listed as not assessed and skipped.
Unparseable answers are excluded from the per-finding confusion tables and
reported as counts. Both per-image metrics also exclude unresolved results: a
true finding whose answer was unparseable is excluded from recall. Complete-case
rate excludes images with unresolved findings. Neither metric gives them credit.

Location truth (which cells a true box occupies) comes, in this order, from
regions attached to the box by location_adapter (apply_adapted), from DENTEX
FDI tooth numbers, or from the fixed cell windows. The evaluation summary
reports which source placed how many boxes.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import run_monitor as mon
from dental_pipeline import (CELL_WINDOWS, CELLS, CONDITIONS, LEFT_IS_IMAGE_LEFT, TRAINED, cell_answers,
                             fdi_unit, unit_cell, units_to_cells)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}  # formats llama.cpp can decode

# Findings with a DentVLM panoramic task; the others are outside its training distribution.
TRAINED_TASK = set(TRAINED)

# Geometric location truth is scored against the cell windows the model's descriptors map
# onto (dental_pipeline.CELL_WINDOWS, overlapping on the canine line and occlusal plane).
OVERLAP_FRACTION = 0.25  # a box counts in every window holding >= 25% of its area

DENTEX_DISEASES = {
    "caries": "carious_lesion",
    "deep caries": "carious_lesion",
    "periapical lesion": "periapical_lesion",
    "impacted": "impacted_tooth",
}


# ----------------------------------------------------------------------------
# Ground truth loaders -> {image_id: {"path", "boxes", "annotated"}}
# box = {"condition", "xc", "yc", "w", "h", "fdi" (optional (quadrant, tooth))}
# ----------------------------------------------------------------------------
def load_yolo(images_dir: str | Path, labels_dir: str | Path) -> dict[str, dict]:
    """UMFIH 14-class layout. An image with no label file is scored as all-negative, and a label
    file with no image is never scored at all, so both are counted and reported: silently, they
    look exactly like a correct dataset."""
    images_root, labels_root = Path(images_dir), Path(labels_dir)
    dataset = {}
    unlabeled = []
    for path in sorted(p for p in images_root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS):
        boxes = []
        label_path = labels_root / path.relative_to(images_root).with_suffix(".txt")
        if not label_path.is_file():
            unlabeled.append(path.stem)
        else:
            for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                parts = line.split()
                if not parts:
                    continue
                if len(parts) != 5:
                    raise ValueError(f"{label_path}:{line_number} must have 5 values")
                class_id = int(parts[0])
                if not 0 <= class_id < len(CONDITIONS):
                    raise ValueError(f"{label_path}:{line_number} unknown class {class_id}")
                xc, yc, w, h = (float(v) for v in parts[1:])
                boxes.append({"condition": CONDITIONS[class_id], "xc": xc, "yc": yc, "w": w, "h": h})
        if path.stem in dataset:
            raise ValueError(f"duplicate image id {path.stem}")
        dataset[path.stem] = {"path": str(path), "boxes": boxes, "annotated": set(CONDITIONS)}
    if not dataset:
        raise ValueError(f"no images under {images_root}")
    orphans = sorted(p.stem for p in labels_root.rglob("*.txt") if p.stem not in dataset)
    for reason, names in (("images with no label file (scored as all-negative)", unlabeled),
                          ("label files with no image (never scored)", orphans)):
        if names:
            mon.monitor("PROBLEM", str(images_root), reason=reason, count=len(names),
                        examples=mon.clip(", ".join(names[:5]), 80))
    return dataset


def load_dentex(images_dir: str | Path, annotations_json: str | Path) -> dict[str, dict]:
    """DENTEX quadrant-enumeration-disease split (COCO-style JSON; train or validation_triple)."""
    payload = json.loads(Path(annotations_json).read_text(encoding="utf-8"))
    quadrants = {c["id"]: int(c["name"]) for c in payload.get("categories_1", []) if str(c["name"]).strip().isdigit()}
    teeth = {c["id"]: int(c["name"]) for c in payload.get("categories_2", []) if str(c["name"]).strip().isdigit()}
    disease_names = {c["id"]: str(c["name"]).strip().lower() for c in payload.get("categories_3", [])}
    images = {img["id"]: img for img in payload["images"]}
    dataset = {}
    for img in payload["images"]:
        image_path = Path(images_dir) / img["file_name"]
        dataset[Path(img["file_name"]).stem] = {
            "path": str(image_path), "boxes": [],
            "annotated": {"carious_lesion", "periapical_lesion", "impacted_tooth"},
        }
    for ann in payload["annotations"]:
        if ann.get("image_id") not in images:
            raise ValueError(f"{annotations_json}: annotation {ann.get('id')} names image_id "
                             f"{ann.get('image_id')!r}, which the file does not list under 'images'")
        img = images[ann["image_id"]]
        disease = disease_names.get(ann.get("category_id_3"))
        condition = DENTEX_DISEASES.get(disease)
        if condition is None:
            raise ValueError(f"{annotations_json}: unknown DENTEX disease label {disease!r} "
                             f"(annotation {ann.get('id')}); extend DENTEX_DISEASES")
        x, y, w, h = ann["bbox"]
        width, height = img["width"], img["height"]
        box = {"condition": condition, "xc": (x + w / 2) / width, "yc": (y + h / 2) / height,
               "w": w / width, "h": h / height}
        # FDI quadrant and tooth number give exact region truth through DentVLM's own mapping
        # (Supplementary Table S6). They agree with box geometry on 97% of validation boxes,
        # which confirms the image-left = patient-right display convention.
        if ann.get("category_id_1") in quadrants and ann.get("category_id_2") in teeth:
            box["fdi"] = (quadrants[ann["category_id_1"]], teeth[ann["category_id_2"]])
        dataset[Path(img["file_name"]).stem]["boxes"].append(box)
    missing = sorted(i for i, e in dataset.items() if not Path(e["path"]).is_file())
    if missing:
        mon.monitor("PROBLEM", str(annotations_json), reason="annotated images not found on disk",
                    count=len(missing), examples=mon.clip(", ".join(missing[:5]), 80))
    return dataset


def truth_report(gt: dict[str, dict], name: str = "dataset") -> dict:
    """One dense line of what a loaded benchmark holds, and a problem line for anything unusable.

    Printed before any model call, because a dataset that is half missing is cheaper to find
    here than after an hour of questions.
    """
    boxes = [b for entry in gt.values() for b in entry["boxes"]]
    per_condition: dict[str, int] = {}
    for box in boxes:
        per_condition[box["condition"]] = per_condition.get(box["condition"], 0) + 1
    missing = sorted(i for i, entry in gt.items() if not Path(entry["path"]).is_file())
    summary = {"images": len(gt), "with_findings": sum(1 for e in gt.values() if e["boxes"]),
               "boxes": len(boxes), "boxes_with_fdi": sum(1 for b in boxes if b.get("fdi")),
               "findings_annotated": len(set().union(*(e["annotated"] for e in gt.values())) if gt else set()),
               "untrained_findings": len(set(per_condition) - TRAINED_TASK),
               "missing_image_files": len(missing), "per_condition": per_condition}
    mon.monitor("TRUTH", name, images=summary["images"], with_findings=summary["with_findings"],
                boxes=summary["boxes"], fdi=summary["boxes_with_fdi"] or None,
                findings=summary["findings_annotated"],
                findings_without_a_dentvlm_task=summary["untrained_findings"] or None)
    ranked = sorted(per_condition.items(), key=lambda kv: -kv[1])
    if ranked:
        shown = " ".join(f"{c}={n}" for c, n in ranked[:6])
        print(f"  boxes per finding: {shown}" + (f" (+{len(ranked) - 6} more findings)" if len(ranked) > 6 else ""),
              flush=True)
    if missing:
        mon.monitor("PROBLEM", name, reason="image files missing; those images will fail",
                    count=len(missing), examples=mon.clip(", ".join(missing[:5]), 80))
    return summary


# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------
def fdi_cell(quadrant: int, tooth: int, left_is_image_left: bool = LEFT_IS_IMAGE_LEFT) -> str:
    """DentVLM's cell for an FDI tooth (Table S6): its 'left' is FDI quadrants 1/4 (patient's right)."""
    return unit_cell(fdi_unit(quadrant, tooth), left_is_image_left)


def geometric_regions(box: dict, windows: dict | None = None) -> set[str]:
    """Cells whose fixed window holds >= 25% of the box area (the model-free fallback)."""
    windows = windows or CELL_WINDOWS
    left, top = box["xc"] - box["w"] / 2, box["yc"] - box["h"] / 2
    right, bottom = box["xc"] + box["w"] / 2, box["yc"] + box["h"] / 2
    area = max(box["w"] * box["h"], 1e-9)
    hits = set()
    for name, (wl, wt, wr, wb) in windows.items():
        overlap = max(0.0, min(right, wr) - max(left, wl)) * max(0.0, min(bottom, wb) - max(top, wt))
        if overlap / area >= OVERLAP_FRACTION:
            hits.add(name)
    return hits


def box_regions(box: dict, windows: dict | None = None) -> set[str]:
    """Cells holding the box: adapted regions if attached, else exact from FDI, else geometry."""
    if box.get("regions") is not None:
        return set(box["regions"])
    if box.get("fdi"):
        return {fdi_cell(*box["fdi"])}
    return geometric_regions(box, windows)


def box_source(box: dict) -> str:
    """Which method decides this box's cells (see box_regions)."""
    if box.get("regions") is not None:
        return box.get("region_source", "adapted")
    return "fdi" if box.get("fdi") else "geometry"


def gt_regions(boxes: list[dict]) -> set[str]:
    regions = set()
    for box in boxes:
        regions |= box_regions(box)
    return regions


def straddling(box: dict) -> bool:
    return len(box_regions(box)) > 1


# ----------------------------------------------------------------------------
# Adapted location truth (location_adapter output)
# ----------------------------------------------------------------------------
def apply_adapted(gt: dict[str, dict], adapted: dict[str, dict]) -> dict[str, dict]:
    """Copy of gt whose boxes carry the adapter's cells as box['regions'] (+ 'region_source').

    Units (from the LLM adapter) and geometry fallbacks are re-mapped here, so flipping
    LEFT_IS_IMAGE_LEFT changes the truth without new adapter calls; cells named by the
    local model itself (source "fdm") are already in its own convention and stay as saved.
    """
    out = {}
    for image_id, entry in gt.items():
        boxes = [dict(b) for b in entry["boxes"]]
        records = adapted.get(image_id, {}).get("boxes") if boxes else []
        if records is None or len(records) != len(boxes):
            raise ValueError(f"{image_id}: {len(boxes)} boxes but adapted truth for "
                             f"{len(records) if records else 0}; run the adapter on every image of this dataset")
        for box, record in zip(boxes, records):
            expected = [box["xc"], box["yc"], box["w"], box["h"]]
            if record.get("condition") != box["condition"] or record.get("box") != expected:
                raise ValueError(f"{image_id}: adapted box order/content does not match ground truth")
            if record["source"] == "excluded":
                box["regions"], box["location_excluded"] = [], True
                box["region_source"] = "excluded"
                continue
            if record["source"] == "llm" and record.get("units"):
                box["regions"] = units_to_cells(record["units"])
            elif record["source"] == "fdm":
                box["regions"] = list(record["regions"])
            else:
                box["regions"] = [c for c in CELLS if c in geometric_regions(box)]
            box["region_source"] = record["source"]
        out[image_id] = {**entry, "boxes": boxes}
    return out


def location_truth_summary(gt: dict[str, dict]) -> dict:
    """How many true boxes each truth source placed (adapted llm/fdm, fdi, geometry)."""
    counts: dict[str, int] = {}
    for entry in gt.values():
        for box in entry["boxes"]:
            counts[box_source(box)] = counts.get(box_source(box), 0) + 1
    return {"boxes": sum(counts.values()), "by_source": dict(sorted(counts.items()))}


def truth_agreement(gt: dict[str, dict], adapted: dict[str, dict]) -> dict:
    """On boxes with FDI tooth numbers (DENTEX): how often the adapter's and the geometric cells
    match the exact FDI cell. This is the adapter's own accuracy check."""
    n = adapter_exact = adapter_contains = geometry_exact = geometry_contains = 0
    for image_id, entry in gt.items():
        records = adapted.get(image_id, {}).get("boxes", [])
        for box, record in zip(entry["boxes"], records):
            if not box.get("fdi"):
                continue
            truth = fdi_cell(*box["fdi"])
            if record["source"] == "llm" and record.get("units"):
                predicted = set(units_to_cells(record["units"]))
            else:
                predicted = set(record["regions"])
            geometry = geometric_regions(box)
            n += 1
            adapter_exact += predicted == {truth}
            adapter_contains += truth in predicted
            geometry_exact += geometry == {truth}
            geometry_contains += truth in geometry
    return {"boxes_with_fdi": n, "adapter_exact": _ratio(adapter_exact, n), "adapter_contains": _ratio(adapter_contains, n),
            "geometry_exact": _ratio(geometry_exact, n), "geometry_contains": _ratio(geometry_contains, n)}


# ----------------------------------------------------------------------------
# Reading saved results
# ----------------------------------------------------------------------------
def predicted_cells(result: dict, condition: str, parser=None) -> dict[str, bool | None] | None:
    """{cell: True | False | None (unparseable)} for one finding, or None when no location was asked, the
    finding was not asked, or (rationale) its whole-image answer was unparseable.

    Rationale: the cells the model named are the prediction and every other cell counts as not predicted,
    the same reading as regions.csv (a cell the rationale did not name is not evidence of absence there).
    Regions: each region question's own answer, merged over the finding's tasks (any yes, all no, else
    unparseable); a result saved without its calls or task list falls back to the finding's cell set.
    """
    finding = result["findings"][condition]
    level = result.get("location_level", "none")
    if not finding["asked"] or level == "none":
        return None
    if level == "regions":
        answers, tasks = cell_answers(result, parser), finding.get("tasks") or []
        if tasks and all(task in answers for task in tasks):
            cells = {}
            for cell in CELLS:
                votes = [answers[task].get(cell) for task in tasks]
                cells[cell] = True if "yes" in votes else False if all(v == "no" for v in votes) else None
            return cells
    if finding["presence"] is None:
        return None
    # Present, but the location itself could not be read (only an LLM parser produces this): that is
    # unresolved, not "the model named no cell", so the image is excluded rather than scored as six
    # negatives. Runs read with code alone never reach this.
    if finding["presence"] == "yes" and finding["regions"] is None:
        return None
    named = set(finding["regions"] or [])
    return {cell: cell in named for cell in CELLS}


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def _ratio(a: float, b: float):
    return round(a / b, 4) if b else None


def _prf(tp, fp, tn, fn) -> dict:
    return {
        "sensitivity": _ratio(tp, tp + fn), "specificity": _ratio(tn, tn + fp),
        "ppv": _ratio(tp, tp + fp), "f1": _ratio(2 * tp, 2 * tp + fp + fn),
    }


def _tally(table: dict, truth: bool, answer: str | None) -> bool:
    """Add one image to a TP/FP/TN/FN table; False (and counted) when the answer was unparseable."""
    if answer is None:
        table["unparseable"] += 1
        return False
    positive = answer == "yes"
    table["TP"] += truth and positive
    table["FP"] += (not truth) and positive
    table["TN"] += (not truth) and (not positive)
    table["FN"] += truth and (not positive)
    return True


def evaluate(gt: dict[str, dict], results: dict[str, dict], dataset: str = "dataset",
             out_dir: str | Path | None = None, *, evaluate_location: bool = True,
             include_analysis: bool = True) -> dict:
    """Score saved results against ground truth. Images missing from either side are skipped."""
    ids = sorted(set(gt) & set(results))
    missing = sorted(set(gt) - set(results))
    presence, whole_image, region_presence, regions, per_image, not_assessed = [], [], [], [], [], []
    protocol = results[ids[0]].get("protocol") if ids else None
    level = next((results[i]["location_level"] for i in ids), "none")
    # The whole-image answers are a separate result only in the region comparison.
    whole_image_kept = level == "regions" and "whole_image" in results[ids[0]]["findings"][CONDITIONS[0]]

    for condition in CONDITIONS:
        annotated = [i for i in ids if condition in gt[i]["annotated"]]
        if not annotated:
            continue
        asked = [i for i in annotated if results[i]["findings"][condition]["asked"]]
        if not asked:
            not_assessed.append(condition)
            continue
        table = {"TP": 0, "FP": 0, "TN": 0, "FN": 0, "unparseable": 0}
        whole = dict(table)
        positives = 0
        r_tp = r_fp = r_tn = r_fn = set_match = n_loc = unlocalized = straddle = 0
        region_unparseable = location_truth_excluded = 0
        jaccard_sum = 0.0
        rp = {name: {"TP": 0, "FP": 0, "TN": 0, "FN": 0, "unparseable": 0, "positives": 0} for name in CELLS}
        rp_excluded = 0
        for image_id in asked:
            boxes = [b for b in gt[image_id]["boxes"] if b["condition"] == condition]
            truth = len(boxes) > 0
            positives += truth
            finding = results[image_id]["findings"][condition]
            if whole_image_kept:
                _tally(whole, truth, finding["whole_image"])
            if evaluate_location and level != "none":
                # Presence per cell: every cell of every image, whatever the whole image said, against the cells
                # the true boxes occupy (none when the finding is absent). One cell per image, so several boxes in
                # one cell are one presence, and an unparseable cell answer is one excluded cell.
                pred_map = predicted_cells(results[image_id], condition)
                placed = [b for b in boxes if not b.get("location_excluded")]
                if pred_map is not None and boxes and not placed:
                    rp_excluded += 1
                elif pred_map is not None:
                    truth_regions = gt_regions(placed)
                    for name, hit in pred_map.items():
                        rp[name]["positives"] += name in truth_regions
                        _tally(rp[name], name in truth_regions, None if hit is None else "yes" if hit else "no")
            if not _tally(table, truth, finding["presence"]):
                continue
            positive = finding["presence"] == "yes"

            if evaluate_location and level != "none" and truth and positive:
                location_boxes = [b for b in boxes if not b.get("location_excluded")]
                if not location_boxes:
                    location_truth_excluded += 1
                    continue
                if finding["regions"] is None:
                    region_unparseable += 1
                    continue
                n_loc += 1
                truth_regions = gt_regions(location_boxes)
                pred_regions = set(finding["regions"])
                straddle += sum(straddling(b) for b in location_boxes)
                if not pred_regions:
                    unlocalized += 1
                for name in CELLS:
                    t, p = name in truth_regions, name in pred_regions
                    r_tp += t and p
                    r_fp += (not t) and p
                    r_tn += (not t) and (not p)
                    r_fn += t and (not p)
                set_match += truth_regions == pred_regions
                union = truth_regions | pred_regions
                jaccard_sum += len(truth_regions & pred_regions) / len(union) if union else 1.0

        row = {"dataset": dataset, "condition": condition, "trained_task": condition in TRAINED_TASK,
               "images": len(asked), "positives": positives}
        presence.append({**row, **table, **_prf(table["TP"], table["FP"], table["TN"], table["FN"])})
        if whole_image_kept:
            whole_image.append({**row, **whole, **_prf(whole["TP"], whole["FP"], whole["TN"], whole["FN"])})
        if evaluate_location and level != "none":
            for name in CELLS:
                cell = rp[name]
                scored = cell["TP"] + cell["FP"] + cell["TN"] + cell["FN"]
                if not scored and not cell["unparseable"] and not rp_excluded:
                    continue  # every image of this finding was unresolved before any cell could be read
                region_presence.append({
                    "dataset": dataset, "condition": condition, "region": name, "level": level,
                    "images": scored + cell["unparseable"], "positives": cell["positives"],
                    **{k: cell[k] for k in ("TP", "FP", "TN", "FN", "unparseable")},
                    **_prf(cell["TP"], cell["FP"], cell["TN"], cell["FN"]),
                    "location_truth_excluded": rp_excluded,
                })
        if evaluate_location and level != "none":
            regions.append({
                "dataset": dataset, "condition": condition, "level": level, "n_localized_cases": n_loc,
                "TP": r_tp, "FP": r_fp, "TN": r_tn, "FN": r_fn, **_prf(r_tp, r_fp, r_tn, r_fn),
                "exact_set_match_rate": _ratio(set_match, n_loc), "mean_jaccard": _ratio(jaccard_sum, n_loc),
                "unlocalized_rate": _ratio(unlocalized, n_loc), "straddling_boxes": straddle,
                "region_unparseable": region_unparseable,
                "expected_location_checks": n_loc + region_unparseable + location_truth_excluded,
                "scored_location_checks": n_loc,
                "excluded_location_checks": region_unparseable + location_truth_excluded,
                "location_truth_excluded": location_truth_excluded,
            })

    for image_id in ids:
        findings = results[image_id]["findings"]
        annotated = {c for c in gt[image_id]["annotated"] if findings[c]["asked"]}
        truths = {c for c in annotated if any(b["condition"] == c for b in gt[image_id]["boxes"])}
        preds = {c for c in annotated if findings[c]["presence"] == "yes"}
        unparsed = sum(findings[c]["presence"] is None for c in annotated)
        scored = {c for c in annotated if results[image_id]["findings"][c]["presence"] is not None}
        caught = truths & preds
        per_image.append({
            "dataset": dataset, "image_id": image_id, "gt_present": len(truths), "caught": len(caught),
            "gt_present_scored": len(truths & scored), "scored_findings": len(scored),
            "complete_case": (truths <= preds) if scored and not unparsed else None,
            "false_alarms": len(preds - truths), "unparseable": unparsed,
            "calls": results[image_id].get("call_count"),
            "inference_calls": results[image_id].get("inference_call_count", results[image_id].get("call_count")),
            "cache_hits": results[image_id].get("cache_hit_count", 0),
        })

    micro = {k: sum(r[k] for r in presence) for k in ("TP", "FP", "TN", "FN")}
    f1s = [r["f1"] for r in presence if r["f1"] is not None]
    complete_images = [r for r in per_image if r["complete_case"] is not None]
    scored_images = [r for r in per_image if r["scored_findings"]]
    logical_calls = sum(r["calls"] or 0 for r in per_image)
    inference_calls = sum(r["inference_calls"] or 0 for r in per_image)
    cache_hits = sum(r["cache_hits"] or 0 for r in per_image)
    summary = {
        "dataset": dataset, "images_scored": len(ids), "images_missing_results": len(missing),
        "protocol": protocol,
        "location_level": level, "not_assessed": not_assessed,
        "evaluate_location": evaluate_location,
        "location_truth": location_truth_summary({i: gt[i] for i in ids}) if evaluate_location else None,
        **micro, **_prf(micro["TP"], micro["FP"], micro["TN"], micro["FN"]),
        "macro_f1": _ratio(sum(f1s), len(f1s)),
        "unparseable_rate": _ratio(sum(r["unparseable"] for r in presence), sum(r["images"] for r in presence)),
        "unparseable_policy": "exclude",
        "expected_finding_checks": sum(r["images"] for r in presence),
        "scored_finding_checks": sum(micro.values()),
        "excluded_unparseable_checks": sum(r["unparseable"] for r in presence),
        "finding_check_invariant_ok": (sum(r["images"] for r in presence)
                                       == sum(micro.values()) + sum(r["unparseable"] for r in presence)),
        "complete_case_images_scored": len(complete_images),
        "complete_case_rate": _ratio(sum(r["complete_case"] for r in complete_images), len(complete_images)),
        "mean_recall_per_image": _ratio(sum(_ratio(r["caught"], r["gt_present_scored"]) or 0
                                          for r in per_image if r["gt_present_scored"]),
                                        sum(1 for r in per_image if r["gt_present_scored"])),
        "mean_false_alarms_per_image": _ratio(sum(r["false_alarms"] for r in scored_images), len(scored_images)),
        "images_with_false_alarm_rate": _ratio(sum(r["false_alarms"] > 0 for r in scored_images), len(scored_images)),
        "logical_calls": logical_calls,
        "inference_calls": inference_calls,
        "cache_hits": cache_hits,
        "mean_calls_per_image": _ratio(logical_calls, len(per_image)),
        "mean_inference_calls_per_image": _ratio(inference_calls, len(per_image)),
        "mean_cache_hits_per_image": _ratio(cache_hits, len(per_image)),
        "cache_hit_rate": _ratio(cache_hits, logical_calls),
    }
    if whole_image:
        micro_whole = {k: sum(r[k] for r in whole_image) for k in ("TP", "FP", "TN", "FN")}
        summary["whole_image"] = {**micro_whole, **_prf(*(micro_whole[k] for k in ("TP", "FP", "TN", "FN")))}
    if region_presence:
        micro_cells = {k: sum(r[k] for r in region_presence) for k in ("TP", "FP", "TN", "FN")}
        summary["region_presence"] = {**micro_cells, **_prf(*(micro_cells[k] for k in ("TP", "FP", "TN", "FN"))),
                                      "unparseable": sum(r["unparseable"] for r in region_presence)}
    if evaluate_location and level != "none":
        summary["side_agreement"] = side_agreement({i: gt[i] for i in ids}, results)
    report = {"summary": summary, "presence": presence, "whole_image": whole_image,
              "region_presence": region_presence, "regions": regions, "per_image": per_image,
              "missing_results": missing}
    if include_analysis:
        from dental_analysis import analyze
        report.update(analyze({i: gt[i] for i in ids}, results, dataset=dataset,
                              evaluate_location=evaluate_location))
    if out_dir:
        write_report(report, out_dir)
    summary = report["summary"]
    mon.monitor("SCORED", dataset, images=len(ids), missing_results=len(missing) or None,
                checks=summary["scored_finding_checks"],
                unparseable=summary["excluded_unparseable_checks"] or None,
                not_assessed=len(summary["not_assessed"]) or None,
                f1=summary["f1"], sens=summary["sensitivity"], spec=summary["specificity"])
    if not summary["finding_check_invariant_ok"]:
        # Every expected check must end up scored or explicitly excluded; anything else is a bug here.
        mon.monitor("EVAL INVARIANT BROKEN", dataset, expected=summary["expected_finding_checks"],
                    scored=summary["scored_finding_checks"],
                    excluded=summary["excluded_unparseable_checks"])
    return report


def side_agreement(gt: dict[str, dict], results: dict[str, dict]) -> dict:
    """How often DentVLM's 'left'/'right' matches the image side of the true boxes.

    Counts every left/right side the model names for a correctly found finding as agreeing when
    a true box of that finding lies on that image side under the current LEFT_IS_IMAGE_LEFT
    reading. A rate far above 50% confirms the reading; far below means it should be flipped.
    """
    named = agree = 0
    for image_id in sorted(set(gt) & set(results)):
        for condition in CONDITIONS:
            finding = results[image_id]["findings"][condition]
            boxes = [b for b in gt[image_id]["boxes"] if b["condition"] == condition]
            if not boxes or finding["presence"] != "yes" or not finding["regions"]:
                continue
            box_sides = {"left" if b["xc"] < 0.5 else "right" for b in boxes}
            for cell in finding["regions"]:
                col = cell.split("-")[1]
                if col == "anterior":
                    continue
                image_side = col if LEFT_IS_IMAGE_LEFT else {"left": "right", "right": "left"}[col]
                named += 1
                agree += image_side in box_sides
    return {"sides_named": named, "agree": agree, "agreement_rate": _ratio(agree, named),
            "left_is_image_left": LEFT_IS_IMAGE_LEFT}


def pooled_presence(reports: list[dict]) -> list[dict]:
    """Sum confusions over the conditions every dataset scores."""
    shared = set.intersection(*(set(r["condition"] for r in rep["presence"]) for rep in reports))
    rows = []
    for condition in CONDITIONS:
        if condition not in shared:
            continue
        cells = {k: 0 for k in ("TP", "FP", "TN", "FN", "unparseable", "images", "positives")}
        for rep in reports:
            row = next(r for r in rep["presence"] if r["condition"] == condition)
            for k in cells:
                cells[k] += row[k]
        rows.append({"dataset": "pooled", "condition": condition, "trained_task": condition in TRAINED_TASK,
                     **cells, **_prf(cells["TP"], cells["FP"], cells["TN"], cells["FN"])})
    return rows


# Tables this branch no longer produces. Their CSVs are deleted on re-export so a directory
# written by an older version never shows stale metrics next to fresh ones.
RETIRED_TABLES = ("counts",)


def write_report(report: dict, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation.json").write_text(json.dumps(report, indent=1, default=list), encoding="utf-8")
    for name in RETIRED_TABLES:
        (out / f"{name}.csv").unlink(missing_ok=True)
    for name in ("presence", "whole_image", "region_presence", "regions", "per_image", "stage_changes",
                 "phrasing_votes", "region_vote_comparison", "parse_recovery", "call_usage", "parser_usage",
                 "case_breakdown", "case_condition_breakdown", "run_comparison", "run_changes",
                 "experiment_overview", "finding_comparison", "situation_comparison",
                 "situation_finding_comparison", "stage_comparison", "phrasing_comparison",
                 "vote_replay_comparison", "parse_recovery_comparison", "call_usage_comparison",
                 "parser_usage_comparison"):
        rows = report.get(name) or []
        if not rows:
            (out / f"{name}.csv").unlink(missing_ok=True)
            continue
        with (out / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v
                              for k, v in row.items()} for row in rows)
