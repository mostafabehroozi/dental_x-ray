"""Deterministic evaluation of saved pipeline results against box-level ground truth.

Ground truth comes from YOLO label files (UMFIH 14-class set) or DENTEX JSON.
Metrics stay simple: image-level TP/FP/TN/FN per finding, count agreement on
true positives (whole-image counts, and per region when counts were taken per
region), region-level TP/FP/TN/FN for localized findings, and two per-image
numbers a dentist cares about (complete-case rate, false alarms).

Unparseable answers are excluded from the per-finding confusion tables and
reported as counts. The per-image complete-case rate and recall are strict: a
true finding whose answer was unparseable counts as not caught.

Location truth (which region windows a true box occupies) comes, in this order,
from regions attached to the box by location_adapter (apply_adapted), from
DENTEX FDI quadrant labels, or from the fixed crop windows. The evaluation
summary reports which source placed how many boxes.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from dental_pipeline import (CONDITIONS, COUNTABLE, CROPS, QUADRANT_WORDS_ARE_PATIENT_SIDE, UNIT_QUADRANT,
                             quadrants_to_regions, units_to_regions)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}  # formats llama.cpp can decode

# Findings whose category (or a near synonym) appears in the DentalGPT paper's
# own label sets; the other findings are outside its documented distribution.
PAPER_COVERED = {"endodontic_treatment", "periapical_lesion", "impacted_tooth",
                 "periodontal_bone_loss", "carious_lesion", "dental_filling"}

# Location truth is scored against the windows the regions are named after
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
    """Windows of the given level holding >= 25% of the box area (the model-free fallback)."""
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


def box_primary_region(box: dict, level: str) -> str | None:
    """The one window a box is counted in: the first, in window order, of the windows holding it."""
    hits = box_regions(box, level)
    return next((name for name in CROPS[level] if name in hits), None)


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
# Reading saved results (both levels, and results saved before the levels existed)
# ----------------------------------------------------------------------------
def result_protocol(result: dict) -> dict:
    """The protocol a result was produced with; results from before the two levels are translated."""
    protocol = result.get("protocol")
    if protocol:
        return dict(protocol)
    level = result.get("location_level", "none")
    return {"presence_level": "overall" if level == "none" else "region", "count_level": "overall",
            "region_scheme": "quadrant" if level == "none" else level, "region_prompt": "crop"}


def result_scheme(result: dict) -> str:
    """Region scheme of a saved result ("quadrant", "arch") or "none" when no region was asked."""
    protocol = result_protocol(result)
    if "region" in (protocol["presence_level"], protocol["count_level"]):
        return protocol["region_scheme"]
    return "none"


def predicted_regions(finding: dict) -> tuple[dict | None, str | None]:
    """{region: True | False | None (unparseable)} and its source, or (None, None) when no region
    question was asked. Regions come from the region presence answers, or, when presence was
    resolved on the whole image only, from the region counts (a count above zero is a hit)."""
    if finding.get("regions") is not None:
        return {r: (None if a is None else a == "A") for r, a in finding["regions"].items()}, "presence"
    counts = finding.get("region_counts")
    if counts:
        return {r: (None if n is None else n > 0) for r, n in counts.items()}, "count"
    return None, None


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
    presence, counts, region_counts, regions, per_image = [], [], [], [], []
    protocol = result_protocol(results[ids[0]]) if ids else None
    level = result_scheme(results[ids[0]]) if ids else "none"
    region_names = tuple(CROPS[level]) if level != "none" else ()
    per_region_counts = bool(protocol) and protocol["count_level"] == "region" and level != "none"

    for condition in CONDITIONS:
        annotated = [i for i in ids if condition in gt[i]["annotated"]]
        if not annotated:
            continue
        tp = fp = tn = fn = unparseable = positives = 0
        exact = within1 = abs_err = signed_err = n_count = strict_n = strict_abs = unparsed_count = unasked_count = 0
        r_tp = r_fp = r_tn = r_fn = set_match = n_loc = unlocalized = straddle = region_unparseable = 0
        from_counts = pred_all = truth_all = 0
        jaccard_sum = 0.0
        rc = {name: {"n": 0, "exact": 0, "within1": 0, "abs": 0, "signed": 0, "strict_n": 0, "strict_abs": 0,
                     "unparseable": 0, "truth_positive": 0} for name in region_names}
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
                    if finding.get("region_counts") == {}:
                        unasked_count += 1  # positive, but no region answered A, so no region was counted
                    else:
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
                if per_region_counts and positive and finding.get("region_counts") is not None:
                    asked = finding["region_counts"]
                    for name in region_names:
                        truth_n = sum(box_primary_region(b, level) == name for b in boxes)
                        cell = rc[name]
                        cell["truth_positive"] += truth_n > 0
                        if name in asked and asked[name] is None:
                            cell["unparseable"] += 1
                            continue
                        pred_n = asked.get(name, 0)  # a region not asked answered B to presence: counted as 0
                        cell["strict_n"] += 1
                        cell["strict_abs"] += abs(truth_n - pred_n)
                        if name in asked:
                            cell["n"] += 1
                            cell["exact"] += pred_n == truth_n
                            cell["within1"] += abs(pred_n - truth_n) <= 1
                            cell["abs"] += abs(pred_n - truth_n)
                            cell["signed"] += pred_n - truth_n

            if level != "none" and truth and positive:
                pred_regions_map, source = predicted_regions(finding)
                if not pred_regions_map:
                    continue
                if any(v is None for v in pred_regions_map.values()):
                    region_unparseable += 1
                    continue
                n_loc += 1
                from_counts += source == "count"
                truth_regions = gt_regions(boxes, level)
                pred_regions = {r for r, hit in pred_regions_map.items() if hit}
                straddle += sum(straddling(b, level) for b in boxes)
                if not pred_regions:
                    unlocalized += 1
                for name in region_names:
                    t, p = name in truth_regions, name in pred_regions
                    r_tp += t and p
                    r_fp += (not t) and p
                    r_tn += (not t) and (not p)
                    r_fn += t and (not p)
                set_match += truth_regions == pred_regions
                pred_all += pred_regions == set(region_names)
                truth_all += truth_regions == set(region_names)
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
                "strict_n": strict_n, "strict_mae": _ratio(strict_abs, strict_n),
                "count_unparseable": unparsed_count, "count_unasked": unasked_count,
            })
            if per_region_counts:
                for name in region_names:
                    cell = rc[name]
                    region_counts.append({
                        "dataset": dataset, "condition": condition, "region": name, "n_scored": cell["n"],
                        "truth_positive_cases": cell["truth_positive"],
                        "exact_rate": _ratio(cell["exact"], cell["n"]), "within_1_rate": _ratio(cell["within1"], cell["n"]),
                        "mae": _ratio(cell["abs"], cell["n"]), "mean_signed_error": _ratio(cell["signed"], cell["n"]),
                        "strict_n": cell["strict_n"], "strict_mae": _ratio(cell["strict_abs"], cell["strict_n"]),
                        "count_unparseable": cell["unparseable"],
                    })
        if level != "none":
            regions.append({
                "dataset": dataset, "condition": condition, "level": level, "n_localized_cases": n_loc,
                "from_counts": from_counts,
                "TP": r_tp, "FP": r_fp, "TN": r_tn, "FN": r_fn, **_prf(r_tp, r_fp, r_tn, r_fn),
                "exact_set_match_rate": _ratio(set_match, n_loc), "mean_jaccard": _ratio(jaccard_sum, n_loc),
                "unlocalized_rate": _ratio(unlocalized, n_loc),
                # A model that ignores the region clause answers A everywhere: compare the two rates.
                "pred_all_regions_rate": _ratio(pred_all, n_loc), "truth_all_regions_rate": _ratio(truth_all, n_loc),
                "straddling_boxes": straddle, "region_unparseable": region_unparseable,
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
        "protocol": protocol, "location_level": level,
        "location_truth": location_truth_summary({i: gt[i] for i in ids}),
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
    if level == "quadrant":
        summary["side_agreement"] = side_agreement(gt, results)
    report = {"summary": summary, "presence": presence, "counts": counts, "region_counts": region_counts,
              "regions": regions, "per_image": per_image, "missing_results": missing}
    if out_dir:
        write_report(report, out_dir)
    return report


def side_agreement(gt: dict[str, dict], results: dict[str, dict]) -> dict:
    """How often a quadrant the model answered for holds a true box on that image side.

    The windows are fixed to the image (UR and LR are image-left), so this reads, for word-based
    regions, whether the model takes "right" as the patient's right the way the phrases assume
    (QUADRANT_WORDS_ARE_PATIENT_SIDE): a rate far above 50% confirms it, far below means the
    words should be flipped. For crops it is plain localization accuracy. Only findings whose true
    boxes all lie on one side of the image are informative, so the others are skipped.
    """
    named = agree = 0
    for image_id in sorted(set(gt) & set(results)):
        if result_scheme(results[image_id]) != "quadrant":
            continue
        for condition in CONDITIONS:
            boxes = [b for b in gt[image_id]["boxes"] if b["condition"] == condition]
            finding = results[image_id]["findings"][condition]
            pred, _ = predicted_regions(finding)
            if not boxes or finding["presence"] != "A" or not pred:
                continue
            box_sides = {"left" if b["xc"] < 0.5 else "right" for b in boxes}
            if len(box_sides) != 1:
                continue
            for region, hit in pred.items():
                if not hit:
                    continue
                named += 1
                agree += ("left" if region in ("UR", "LR") else "right") in box_sides
    return {"sides_named": named, "agree": agree, "agreement_rate": _ratio(agree, named),
            "quadrant_words_are_patient_side": QUADRANT_WORDS_ARE_PATIENT_SIDE}


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
    for name in ("presence", "counts", "region_counts", "regions", "per_image"):
        rows = report.get(name) or []
        if not rows:
            continue
        with (out / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
