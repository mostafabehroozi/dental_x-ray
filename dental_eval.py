"""Deterministic evaluation of saved pipeline results against box-level ground truth.

Ground truth comes from YOLO label files (UMFIH 14-class set) or DENTEX JSON.
Metrics stay simple: image-level TP/FP/TN/FN per finding, count agreement on
true positives, quadrant-level TP/FP/TN/FN for localized findings, and two
per-image numbers a dentist cares about (complete-case rate, false alarms).

Unparseable answers are excluded from the per-finding confusion tables and
reported as counts. The per-image complete-case rate and recall are strict: a
true finding whose answer was unparseable counts as not caught.

Location truth (which crop windows a true box occupies) comes, in this order,
from regions attached to the box by location_adapter (apply_adapted), from
DENTEX FDI quadrant labels, or from the fixed crop windows. The evaluation
summary reports which source placed how many boxes.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from dental_pipeline import CONDITIONS, COUNTABLE, CROPS, UNIT_QUADRANT, quadrants_to_regions, units_to_regions

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}  # formats llama.cpp can decode

# Findings whose category (or a near synonym) appears in the DentalGPT paper's
# own label sets; the other findings are outside its documented distribution.
PAPER_COVERED = {"endodontic_treatment", "periapical_lesion", "impacted_tooth",
                 "periodontal_bone_loss", "carious_lesion", "dental_filling"}

# Location truth is scored against the crop windows the model actually saw
# (dental_pipeline.CROPS, which overlap on the midline and occlusal plane).
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
        # FDI quadrant and tooth number give exact quadrant truth. They agree with box geometry on
        # 97% of validation boxes, which confirms the image-left = patient-right display convention.
        if ann.get("category_id_1") in quadrants and ann.get("category_id_2") in teeth:
            box["fdi"] = (quadrants[ann["category_id_1"]], teeth[ann["category_id_2"]])
        dataset[Path(img["file_name"]).stem]["boxes"].append(box)
    return dataset


# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------
def fdi_quadrant(quadrant: int) -> str:
    """Quadrant window name of an FDI quadrant (primary-dentition quadrants 5-8 fold onto 1-4)."""
    return UNIT_QUADRANT[f"Q{quadrant - 4 if quadrant > 4 else quadrant}"]


def geometric_regions(box: dict, level: str) -> set[str]:
    """Crop windows of the given level holding >= 25% of the box area (the model-free fallback)."""
    left, top = box["xc"] - box["w"] / 2, box["yc"] - box["h"] / 2
    right, bottom = box["xc"] + box["w"] / 2, box["yc"] + box["h"] / 2
    area = max(box["w"] * box["h"], 1e-9)
    hits = set()
    for name, (wl, wt, wr, wb) in CROPS[level].items():
        overlap = max(0.0, min(right, wr) - max(left, wl)) * max(0.0, min(bottom, wb) - max(top, wt))
        if overlap / area >= OVERLAP_FRACTION:
            hits.add(name)
    return hits


def box_regions(box: dict, level: str) -> set[str]:
    """Windows holding the box: adapted quadrants if attached, else exact from FDI, else geometry."""
    if box.get("regions") is not None:
        return set(quadrants_to_regions(box["regions"], level))
    if box.get("fdi"):
        return set(quadrants_to_regions([fdi_quadrant(box["fdi"][0])], level))
    return geometric_regions(box, level)


def box_source(box: dict) -> str:
    """Which method decides this box's windows (see box_regions)."""
    if box.get("regions") is not None:
        return box.get("region_source", "adapted")
    return "fdi" if box.get("fdi") else "geometry"


def gt_regions(boxes: list[dict], level: str) -> set[str]:
    regions = set()
    for box in boxes:
        regions |= box_regions(box, level)
    return regions


def straddling(box: dict, level: str) -> bool:
    return len(box_regions(box, level)) > 1


# ----------------------------------------------------------------------------
# Adapted location truth (location_adapter output)
# ----------------------------------------------------------------------------
def apply_adapted(gt: dict[str, dict], adapted: dict[str, dict]) -> dict[str, dict]:
    """Copy of gt whose boxes carry the adapter's quadrants as box['regions'] (+ 'region_source').

    Units from the LLM adapter and geometry fallbacks are re-mapped here; quadrants named by the
    local model itself (source "fdm") stay as saved.
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
                box["regions"] = units_to_regions(record["units"])
            elif record["source"] == "fdm":
                box["regions"] = list(record["regions"])
            else:
                box["regions"] = quadrants_to_regions(geometric_regions(box, "quadrant"))
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
    """On boxes with FDI labels (DENTEX): how often the adapter's and the geometric quadrants match
    the exact FDI quadrant. This is the adapter's own accuracy check."""
    n = adapter_exact = adapter_contains = geometry_exact = geometry_contains = 0
    for image_id, entry in gt.items():
        records = adapted.get(image_id, {}).get("boxes", [])
        for box, record in zip(entry["boxes"], records):
            if not box.get("fdi"):
                continue
            truth = fdi_quadrant(box["fdi"][0])
            if record["source"] == "llm" and record.get("units"):
                predicted = set(units_to_regions(record["units"]))
            else:
                predicted = set(record["regions"])
            geometry = geometric_regions(box, "quadrant")
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


def evaluate(gt: dict[str, dict], results: dict[str, dict], dataset: str = "dataset",
             out_dir: str | Path | None = None) -> dict:
    """Score saved results against ground truth. Images missing from either side are skipped."""
    ids = sorted(set(gt) & set(results))
    missing = sorted(set(gt) - set(results))
    presence, counts, regions, per_image = [], [], [], []
    level = next((results[i]["location_level"] for i in ids), "none")

    for condition in CONDITIONS:
        annotated = [i for i in ids if condition in gt[i]["annotated"]]
        if not annotated:
            continue
        tp = fp = tn = fn = unparseable = positives = 0
        exact = within1 = abs_err = signed_err = n_count = strict_n = strict_abs = unparsed_count = 0
        r_tp = r_fp = r_tn = r_fn = set_match = n_loc = unlocalized = straddle = region_unparseable = 0
        jaccard_sum = 0.0
        for image_id in annotated:
            boxes = [b for b in gt[image_id]["boxes"] if b["condition"] == condition]
            truth = len(boxes) > 0
            positives += truth
            finding = results[image_id]["findings"][condition]
            pred = finding["presence"]
            if pred is None:
                unparseable += 1
                continue
            positive = pred == "A"
            tp += truth and positive
            fp += (not truth) and positive
            tn += (not truth) and (not positive)
            fn += truth and (not positive)

            if condition in COUNTABLE and truth:
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

            if level != "none" and truth and positive and finding["regions"]:
                if any(a is None for a in finding["regions"].values()):
                    region_unparseable += 1
                    continue
                n_loc += 1
                truth_regions = gt_regions(boxes, level)
                pred_regions = {r for r, a in finding["regions"].items() if a == "A"}
                straddle += sum(straddling(b, level) for b in boxes)
                if not pred_regions:
                    unlocalized += 1
                for name in CROPS[level]:
                    t, p = name in truth_regions, name in pred_regions
                    r_tp += t and p
                    r_fp += (not t) and p
                    r_tn += (not t) and (not p)
                    r_fn += t and (not p)
                set_match += truth_regions == pred_regions
                union = truth_regions | pred_regions
                jaccard_sum += len(truth_regions & pred_regions) / len(union) if union else 1.0

        presence.append({
            "dataset": dataset, "condition": condition, "paper_covered": condition in PAPER_COVERED,
            "images": len(annotated), "positives": positives,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn, "unparseable": unparseable, **_prf(tp, fp, tn, fn),
        })
        if condition in COUNTABLE:
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
                "region_unparseable": region_unparseable,
            })

    for image_id in ids:
        annotated = gt[image_id]["annotated"]
        truths = {c for c in annotated if any(b["condition"] == c for b in gt[image_id]["boxes"])}
        preds = {c for c in annotated if results[image_id]["findings"][c]["presence"] == "A"}
        unparsed = sum(results[image_id]["findings"][c]["presence"] is None for c in annotated)
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
        "location_level": level, "location_truth": location_truth_summary({i: gt[i] for i in ids}),
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


def pooled_presence(reports: list[dict]) -> list[dict]:
    """Sum confusions over the conditions every dataset annotates."""
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
        rows.append({"dataset": "pooled", "condition": condition, "paper_covered": condition in PAPER_COVERED,
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
