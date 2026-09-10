"""Deterministic evaluation of saved pipeline results against box-level ground truth.

Ground truth comes from YOLO label files (UMFIH 14-class set) or DENTEX JSON.
Metrics stay simple: image-level TP/FP/TN/FN per finding, cell-level TP/FP/TN/FN
for the six dental-arch regions DentVLM names, count agreement when the optional
count question was asked, and two per-image numbers a dentist cares about
(complete-case rate, false alarms).

Findings the model was not asked about are listed as not assessed and skipped.
Unparseable answers are excluded from the per-finding confusion tables and
reported as counts. The per-image complete-case rate and recall are strict: a
true finding whose answer was unparseable counts as not caught.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from dental_pipeline import CELL_WINDOWS, CELLS, CONDITIONS, COUNTABLE, LEFT_IS_IMAGE_LEFT, TRAINED

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
    row = "upper" if quadrant in (1, 2, 5, 6) else "lower"
    if tooth <= 3:
        return f"{row}-anterior"
    patient_right = quadrant in (1, 4, 5, 8)
    col = ("left" if patient_right else "right") if left_is_image_left else ("right" if patient_right else "left")
    return f"{row}-{col}"


def box_regions(box: dict, windows: dict | None = None) -> set[str]:
    """Cells holding the box: exact from FDI when present, else windows holding >= 25% of its area."""
    if box.get("fdi"):
        return {fdi_cell(*box["fdi"])}
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


def gt_regions(boxes: list[dict]) -> set[str]:
    regions = set()
    for box in boxes:
        regions |= box_regions(box)
    return regions


def straddling(box: dict) -> bool:
    return len(box_regions(box)) > 1


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


def evaluate(gt: dict[str, dict], results: dict[str, dict], dataset: str = "dataset",
             out_dir: str | Path | None = None) -> dict:
    """Score saved results against ground truth. Images missing from either side are skipped."""
    ids = sorted(set(gt) & set(results))
    missing = sorted(set(gt) - set(results))
    presence, counts, regions, per_image, not_assessed = [], [], [], [], []
    level = next((results[i]["location_level"] for i in ids), "none")
    count_asked = any(results[i].get("protocol", {}).get("count_question") for i in ids)

    for condition in CONDITIONS:
        annotated = [i for i in ids if condition in gt[i]["annotated"]]
        if not annotated:
            continue
        asked = [i for i in annotated if results[i]["findings"][condition]["asked"]]
        if not asked:
            not_assessed.append(condition)
            continue
        tp = fp = tn = fn = unparseable = positives = 0
        exact = within1 = abs_err = signed_err = n_count = strict_n = strict_abs = unparsed_count = 0
        r_tp = r_fp = r_tn = r_fn = set_match = n_loc = unlocalized = straddle = 0
        jaccard_sum = 0.0
        for image_id in asked:
            boxes = [b for b in gt[image_id]["boxes"] if b["condition"] == condition]
            truth = len(boxes) > 0
            positives += truth
            finding = results[image_id]["findings"][condition]
            pred = finding["presence"]
            if pred is None:
                unparseable += 1
                continue
            positive = pred == "yes"
            tp += truth and positive
            fp += (not truth) and positive
            tn += (not truth) and (not positive)
            fn += truth and (not positive)

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

            if level != "none" and truth and positive and finding["regions"] is not None:
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

        presence.append({
            "dataset": dataset, "condition": condition, "trained_task": condition in TRAINED_TASK,
            "images": len(asked), "positives": positives,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn, "unparseable": unparseable, **_prf(tp, fp, tn, fn),
        })
        if count_asked and condition in COUNTABLE:
            counts.append({
                "dataset": dataset, "condition": condition, "n_scored": n_count,
                "exact_rate": _ratio(exact, n_count), "within_1_rate": _ratio(within1, n_count),
                "mae": _ratio(abs_err, n_count), "mean_signed_error": _ratio(signed_err, n_count),
                "strict_n": strict_n, "strict_mae": _ratio(strict_abs, strict_n), "count_unparseable": unparsed_count,
            })
        if level != "none":
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
    report = {"summary": summary, "presence": presence, "counts": counts, "regions": regions,
              "per_image": per_image, "missing_results": missing}
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
    for name in ("presence", "counts", "regions", "per_image"):
        rows = report[name]
        if not rows:
            continue
        with (out / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
