"""Deterministic evaluation of saved pipeline results against box-level ground truth.

Ground truth comes from YOLO label files (UMFIH 14-class set) or DENTEX JSON.
Metrics stay simple: image-level TP/FP/TN/FN per finding (and the same table
for the whole-image answers alone in the crop comparison, to show what the
cells recovered and what it cost), cell-level TP/FP/TN/FN for the six
dental-arch regions DentVLM names, count agreement when the optional count
question was asked, and two per-image numbers a dentist cares about
(complete-case rate, false alarms).

Findings the model was not asked about are listed as not assessed and skipped.
Unparseable answers are excluded from the per-finding confusion tables and
reported as counts. The per-image complete-case rate and recall are strict: a
true finding whose answer was unparseable counts as not caught.

Location truth (which cells a true box occupies) comes, in this order, from
regions attached to the box by location_adapter (apply_adapted), from DENTEX
FDI tooth numbers, or from the fixed cell windows. The evaluation summary
reports which source placed how many boxes.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from dental_pipeline import (CELL_WINDOWS, CELLS, CONDITIONS, COUNTABLE, LEFT_IS_IMAGE_LEFT, TRAINED, fdi_unit,
                             unit_cell, units_to_cells)

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
    images_root, labels_root = Path(images_dir), Path(labels_dir)
    dataset = {}
    for path in sorted(p for p in images_root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS):
        boxes = []
        label_path = labels_root / path.relative_to(images_root).with_suffix(".txt")
        if label_path.is_file():
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
        img = images[ann["image_id"]]
        disease = disease_names.get(ann.get("category_id_3"))
        condition = DENTEX_DISEASES.get(disease)
        if condition is None:
            raise ValueError(f"unknown DENTEX disease label {disease!r}")
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
    return dataset


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
             out_dir: str | Path | None = None, *, evaluate_location: bool = True) -> dict:
    """Score saved results against ground truth. Images missing from either side are skipped."""
    ids = sorted(set(gt) & set(results))
    missing = sorted(set(gt) - set(results))
    presence, whole_image, counts, regions, per_image, not_assessed = [], [], [], [], [], []
    level = next((results[i]["location_level"] for i in ids), "none")
    count_asked = any(results[i].get("protocol", {}).get("count_question") for i in ids)
    # The whole-image answers are a separate result only in the crop comparison.
    whole_image_kept = level == "crops" and "whole_image" in results[ids[0]]["findings"][CONDITIONS[0]]

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
        exact = within1 = abs_err = signed_err = n_count = strict_n = strict_abs = unparsed_count = 0
        r_tp = r_fp = r_tn = r_fn = set_match = n_loc = unlocalized = straddle = 0
        jaccard_sum = 0.0
        for image_id in asked:
            boxes = [b for b in gt[image_id]["boxes"] if b["condition"] == condition]
            truth = len(boxes) > 0
            positives += truth
            finding = results[image_id]["findings"][condition]
            if whole_image_kept:
                _tally(whole, truth, finding["whole_image"])
            if not _tally(table, truth, finding["presence"]):
                continue
            positive = finding["presence"] == "yes"

            if count_asked and condition in COUNTABLE and truth:
                strict_pred = 0 if not positive else finding["count"]
                if strict_pred is None:
                    unparsed_count += 1
                else:
                    strict_n += 1
                    strict_abs += abs(len(boxes) - strict_pred)
                if positive and finding["count"] is not None:
                    n_count += 1
                    diff = finding["count"] - len(boxes)
                    exact += diff == 0
                    within1 += abs(diff) <= 1
                    abs_err += abs(diff)
                    signed_err += diff

            if evaluate_location and level != "none" and truth and positive and finding["regions"] is not None:
                n_loc += 1
                truth_regions = gt_regions(boxes)
                pred_regions = set(finding["regions"])
                straddle += sum(straddling(b) for b in boxes)
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
        if count_asked and condition in COUNTABLE:
            counts.append({
                "dataset": dataset, "condition": condition, "n_scored": n_count,
                "exact_rate": _ratio(exact, n_count), "within_1_rate": _ratio(within1, n_count),
                "mae": _ratio(abs_err, n_count), "mean_signed_error": _ratio(signed_err, n_count),
                "strict_n": strict_n, "strict_mae": _ratio(strict_abs, strict_n), "count_unparseable": unparsed_count,
            })
        if evaluate_location and level != "none":
            regions.append({
                "dataset": dataset, "condition": condition, "level": level, "n_localized_cases": n_loc,
                "TP": r_tp, "FP": r_fp, "TN": r_tn, "FN": r_fn, **_prf(r_tp, r_fp, r_tn, r_fn),
                "exact_set_match_rate": _ratio(set_match, n_loc), "mean_jaccard": _ratio(jaccard_sum, n_loc),
                "unlocalized_rate": _ratio(unlocalized, n_loc), "straddling_boxes": straddle,
            })

    for image_id in ids:
        findings = results[image_id]["findings"]
        annotated = {c for c in gt[image_id]["annotated"] if findings[c]["asked"]}
        truths = {c for c in annotated if any(b["condition"] == c for b in gt[image_id]["boxes"])}
        preds = {c for c in annotated if findings[c]["presence"] == "yes"}
        unparsed = sum(findings[c]["presence"] is None for c in annotated)
        caught = truths & preds
        per_image.append({
            "dataset": dataset, "image_id": image_id, "gt_present": len(truths), "caught": len(caught),
            "complete_case": truths <= preds, "false_alarms": len(preds - truths), "unparseable": unparsed,
            "calls": results[image_id].get("call_count"),
        })

    micro = {k: sum(r[k] for r in presence) for k in ("TP", "FP", "TN", "FN")}
    f1s = [r["f1"] for r in presence if r["f1"] is not None]
    summary = {
        "dataset": dataset, "images_scored": len(ids), "images_missing_results": len(missing),
        "location_level": level, "not_assessed": not_assessed,
        "evaluate_location": evaluate_location,
        "location_truth": location_truth_summary({i: gt[i] for i in ids}) if evaluate_location else None,
        **micro, **_prf(micro["TP"], micro["FP"], micro["TN"], micro["FN"]),
        "macro_f1": _ratio(sum(f1s), len(f1s)),
        "unparseable_rate": _ratio(sum(r["unparseable"] for r in presence), sum(r["images"] for r in presence)),
        "complete_case_rate": _ratio(sum(r["complete_case"] for r in per_image), len(per_image)),
        "mean_recall_per_image": _ratio(sum(_ratio(r["caught"], r["gt_present"]) or 0 for r in per_image if r["gt_present"]),
                                        sum(1 for r in per_image if r["gt_present"])),
        "mean_false_alarms_per_image": _ratio(sum(r["false_alarms"] for r in per_image), len(per_image)),
        "images_with_false_alarm_rate": _ratio(sum(r["false_alarms"] > 0 for r in per_image), len(per_image)),
        "mean_calls_per_image": _ratio(sum(r["calls"] or 0 for r in per_image), len(per_image)),
    }
    if whole_image:
        micro_whole = {k: sum(r[k] for r in whole_image) for k in ("TP", "FP", "TN", "FN")}
        summary["whole_image"] = {**micro_whole, **_prf(*(micro_whole[k] for k in ("TP", "FP", "TN", "FN")))}
    report = {"summary": summary, "presence": presence, "whole_image": whole_image, "counts": counts,
              "regions": regions, "per_image": per_image, "missing_results": missing}
    if out_dir:
        write_report(report, out_dir)
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


def write_report(report: dict, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation.json").write_text(json.dumps(report, indent=1, default=list), encoding="utf-8")
    for name in ("presence", "whole_image", "counts", "regions", "per_image"):
        rows = report.get(name) or []
        if not rows:
            if name == "regions":
                (out / f"{name}.csv").unlink(missing_ok=True)
            continue
        with (out / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
