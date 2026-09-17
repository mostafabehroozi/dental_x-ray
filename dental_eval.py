"""Deterministic evaluation of saved pipeline results against box-level ground truth.

Ground truth comes from YOLO label files (UMFIH 14-class set) or DENTEX JSON.
Metrics stay simple: image-level TP/FP/TN/FN per finding (and the same table
for the whole-image answers when presence was resolved per region, to show what
the regional pass recovered and what it cost), presence per region (every region
answer of every image against the regions the true boxes occupy, so a finding
class is scored once per region rather than counted), count agreement on true
positives (whole-image counts, and per region when counts were taken per
region; left out when counting was off), region-level TP/FP/TN/FN for localized
findings, and two per-image numbers a dentist cares about (complete-case rate,
false alarms).

Unparseable answers are neutral: excluded from confusion tables and per-image
recall, with coverage reported separately. Complete-case rate excludes images
with unresolved findings rather than treating them as a success or failure.

Location truth (which region windows a true box occupies) comes, in this order,
from regions attached to the box by location_adapter (apply_adapted), from
DENTEX FDI quadrant labels, or from the fixed region windows. The evaluation
summary reports which source placed how many boxes.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import run_monitor as mon
from dental_pipeline import (CONDITIONS, COUNTABLE, REGION_WINDOWS, QUADRANT_WORDS_ARE_PATIENT_SIDE, UNIT_QUADRANT,
                             quadrants_to_regions, units_to_regions)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}  # formats llama.cpp can decode

# Findings whose category (or a near synonym) appears in the DentalGPT paper's
# own label sets; the other findings are outside its documented distribution.
PAPER_COVERED = {"endodontic_treatment", "periapical_lesion", "impacted_tooth",
                 "periodontal_bone_loss", "carious_lesion", "dental_filling"}

# Location truth is scored against the windows the regions are named after
# (dental_pipeline.REGION_WINDOWS, which overlap on the midline and occlusal plane).
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
        # FDI quadrant and tooth number give exact quadrant truth. They agree with box geometry on
        # 97% of validation boxes, which confirms the image-left = patient-right display convention.
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
               "missing_image_files": len(missing), "per_condition": per_condition}
    mon.monitor("TRUTH", name, images=summary["images"], with_findings=summary["with_findings"],
                boxes=summary["boxes"], fdi=summary["boxes_with_fdi"] or None,
                findings=summary["findings_annotated"])
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
def fdi_quadrant(quadrant: int) -> str:
    """Quadrant window name of an FDI quadrant (primary-dentition quadrants 5-8 fold onto 1-4)."""
    return UNIT_QUADRANT[f"Q{quadrant - 4 if quadrant > 4 else quadrant}"]


def geometric_regions(box: dict, level: str) -> set[str]:
    """Windows of the given level holding >= 25% of the box area (the model-free fallback)."""
    left, top = box["xc"] - box["w"] / 2, box["yc"] - box["h"] / 2
    right, bottom = box["xc"] + box["w"] / 2, box["yc"] + box["h"] / 2
    area = max(box["w"] * box["h"], 1e-9)
    hits = set()
    for name, (wl, wt, wr, wb) in REGION_WINDOWS[level].items():
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
    return next((name for name in REGION_WINDOWS[level] if name in hits), None)


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

    Units from the LLM adapter and geometry fallbacks are re-mapped here; quadrants the area
    adapter placed a box in ("areas") and quadrants named by the local model itself ("fdm") stay
    as saved.
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
                box["regions"] = units_to_regions(record["units"])
            elif record["source"] in ("fdm", "areas"):
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
    if protocol:  # the form and counting knobs were added later; older results took counts in the separate form
        return {**protocol, "question_form": protocol.get("question_form", "separate"),
                "counting": protocol.get("counting", True)}
    level = result.get("location_level", "none")
    return {"presence_level": "overall" if level == "none" else "region", "count_level": "overall",
            "region_scheme": "quadrant" if level == "none" else level,
            "question_form": "separate", "counting": True}


def result_scheme(result: dict) -> str:
    """Region scheme of a saved result ("quadrant", "arch") or "none" when no region was asked."""
    protocol = result_protocol(result)
    if protocol["presence_level"] == "region" or (protocol["counting"] and protocol["count_level"] == "region"):
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


def _tally(table: dict, truth: bool, answer: str | None) -> bool:
    """Add one image to a TP/FP/TN/FN table; False (and counted) when the answer was unparseable."""
    if answer is None:
        table["unparseable"] += 1
        return False
    positive = answer == "A"
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
    presence, whole_image, counts, region_counts, region_presence, regions, per_image = [], [], [], [], [], [], []
    protocol = result_protocol(results[ids[0]]) if ids else None
    counting = protocol["counting"] if protocol else True
    level = result_scheme(results[ids[0]]) if ids else "none"
    region_names = tuple(REGION_WINDOWS[level]) if evaluate_location and level != "none" else ()
    per_region_counts = counting and bool(region_names) and protocol["count_level"] == "region"
    # The whole-image answers are a separate result only when presence was resolved per region.
    whole_image_kept = (bool(protocol) and protocol["presence_level"] == "region"
                        and "whole_image" in results[ids[0]]["findings"][CONDITIONS[0]])

    for condition in CONDITIONS:
        annotated = [i for i in ids if condition in gt[i]["annotated"]]
        if not annotated:
            continue
        table = {"TP": 0, "FP": 0, "TN": 0, "FN": 0, "unparseable": 0}
        whole = dict(table)
        positives = 0
        exact = within1 = abs_err = signed_err = n_count = strict_n = strict_abs = unparsed_count = 0
        r_tp = r_fp = r_tn = r_fn = set_match = n_loc = unlocalized = straddle = region_unparseable = 0
        location_truth_excluded = 0
        from_counts = pred_all = truth_all = 0
        jaccard_sum = 0.0
        rc = {name: {"n": 0, "exact": 0, "within1": 0, "abs": 0, "signed": 0, "strict_n": 0, "strict_abs": 0,
                     "unparseable": 0, "truth_positive": 0} for name in region_names}
        rp = {name: {"TP": 0, "FP": 0, "TN": 0, "FN": 0, "unparseable": 0, "positives": 0} for name in region_names}
        rp_excluded = 0
        for image_id in annotated:
            boxes = [b for b in gt[image_id]["boxes"] if b["condition"] == condition]
            truth = len(boxes) > 0
            positives += truth
            finding = results[image_id]["findings"][condition]
            if whole_image_kept:
                _tally(whole, truth, finding["whole_image"])
            if region_names:
                # Presence per region: every region answer of every image, whatever the whole image said, against
                # the regions the true boxes occupy (none when the finding is absent). One cell per region, so
                # several boxes in one region are one presence, and an unparseable region is one excluded cell.
                pred_map, _ = predicted_regions(finding)
                placed = [b for b in boxes if not b.get("location_excluded")]
                if pred_map is not None and boxes and not placed:
                    rp_excluded += 1
                elif pred_map is not None:
                    truth_regions = gt_regions(placed, level)
                    for name, hit in pred_map.items():
                        if name in rp:
                            rp[name]["positives"] += name in truth_regions
                            _tally(rp[name], name in truth_regions, None if hit is None else "A" if hit else "B")
            if not _tally(table, truth, finding["presence"]):
                continue
            positive = finding["presence"] == "A"

            if counting and condition in COUNTABLE and truth:
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

            if evaluate_location and level != "none" and truth and positive:
                location_boxes = [b for b in boxes if not b.get("location_excluded")]
                if not location_boxes:
                    location_truth_excluded += 1
                    continue
                pred_regions_map, source = predicted_regions(finding)
                if not pred_regions_map:
                    continue
                if any(v is None for v in pred_regions_map.values()):
                    region_unparseable += 1
                    continue
                n_loc += 1
                from_counts += source == "count"
                truth_regions = gt_regions(location_boxes, level)
                pred_regions = {r for r, hit in pred_regions_map.items() if hit}
                straddle += sum(straddling(b, level) for b in location_boxes)
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

        row = {"dataset": dataset, "condition": condition, "paper_covered": condition in PAPER_COVERED,
               "images": len(annotated), "positives": positives}
        presence.append({**row, **table, **_prf(table["TP"], table["FP"], table["TN"], table["FN"])})
        if whole_image_kept:
            whole_image.append({**row, **whole, **_prf(whole["TP"], whole["FP"], whole["TN"], whole["FN"])})
        for name in region_names:
            cell = rp[name]
            scored = cell["TP"] + cell["FP"] + cell["TN"] + cell["FN"]
            if not scored and not cell["unparseable"] and not rp_excluded:
                continue  # no region answer for this finding (a presence-only finding under region counts)
            region_presence.append({
                "dataset": dataset, "condition": condition, "region": name, "level": level,
                "images": scored + cell["unparseable"], "positives": cell["positives"],
                **{k: cell[k] for k in ("TP", "FP", "TN", "FN", "unparseable")},
                **_prf(cell["TP"], cell["FP"], cell["TN"], cell["FN"]),
                "location_truth_excluded": rp_excluded,
            })
        if counting and condition in COUNTABLE:
            counts.append({
                "dataset": dataset, "condition": condition, "n_scored": n_count,
                "exact_rate": _ratio(exact, n_count), "within_1_rate": _ratio(within1, n_count),
                "mae": _ratio(abs_err, n_count), "mean_signed_error": _ratio(signed_err, n_count),
                "strict_n": strict_n, "strict_mae": _ratio(strict_abs, strict_n),
                "count_unparseable": unparsed_count,
                "expected_count_checks": strict_n + unparsed_count,
                "excluded_count_checks": unparsed_count,
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
        if evaluate_location and level != "none":
            regions.append({
                "dataset": dataset, "condition": condition, "level": level, "n_localized_cases": n_loc,
                "from_counts": from_counts,
                "TP": r_tp, "FP": r_fp, "TN": r_tn, "FN": r_fn, **_prf(r_tp, r_fp, r_tn, r_fn),
                "exact_set_match_rate": _ratio(set_match, n_loc), "mean_jaccard": _ratio(jaccard_sum, n_loc),
                "unlocalized_rate": _ratio(unlocalized, n_loc),
                # A model that ignores the region clause answers A everywhere: compare the two rates.
                "pred_all_regions_rate": _ratio(pred_all, n_loc), "truth_all_regions_rate": _ratio(truth_all, n_loc),
                "straddling_boxes": straddle, "region_unparseable": region_unparseable,
                "expected_location_checks": n_loc + region_unparseable + location_truth_excluded,
                "scored_location_checks": n_loc,
                "excluded_location_checks": region_unparseable + location_truth_excluded,
                "location_truth_excluded": location_truth_excluded,
            })

    for image_id in ids:
        annotated = gt[image_id]["annotated"]
        truths = {c for c in annotated if any(b["condition"] == c for b in gt[image_id]["boxes"])}
        preds = {c for c in annotated if results[image_id]["findings"][c]["presence"] == "A"}
        unparsed = sum(results[image_id]["findings"][c]["presence"] is None for c in annotated)
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
        "protocol": protocol, "location_level": level,
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
        micro_regions = {k: sum(r[k] for r in region_presence) for k in ("TP", "FP", "TN", "FN")}
        summary["region_presence"] = {**micro_regions, **_prf(*(micro_regions[k] for k in ("TP", "FP", "TN", "FN"))),
                                      "unparseable": sum(r["unparseable"] for r in region_presence)}
    if evaluate_location and level == "quadrant":
        summary["side_agreement"] = side_agreement(gt, results)
    report = {"summary": summary, "presence": presence, "whole_image": whole_image, "counts": counts,
              "region_counts": region_counts, "region_presence": region_presence, "regions": regions,
              "per_image": per_image, "missing_results": missing}
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
                f1=summary["f1"], sens=summary["sensitivity"], spec=summary["specificity"])
    if not summary["finding_check_invariant_ok"]:
        # Every expected check must end up scored or explicitly excluded; anything else is a bug here.
        mon.monitor("EVAL INVARIANT BROKEN", dataset, expected=summary["expected_finding_checks"],
                    scored=summary["scored_finding_checks"],
                    excluded=summary["excluded_unparseable_checks"])
    return report


def side_agreement(gt: dict[str, dict], results: dict[str, dict]) -> dict:
    """How often a quadrant the model answered for holds a true box on that image side.

    The windows are fixed to the image (UR and LR are image-left), so this reads whether the model
    takes "right" as the patient's right the way the phrases assume (QUADRANT_WORDS_ARE_PATIENT_SIDE):
    a rate far above 50% confirms it, far below means the words should be flipped. Only findings whose
    true boxes all lie on one side of the image are informative, so the others are skipped.
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
    for name in ("presence", "whole_image", "counts", "region_counts", "region_presence", "regions", "per_image",
                 "stage_changes", "parse_recovery", "call_usage", "case_breakdown", "case_condition_breakdown",
                 "run_comparison", "run_changes", "experiment_overview", "finding_comparison",
                 "situation_comparison", "situation_finding_comparison"):
        rows = report.get(name) or []
        if not rows:
            (out / f"{name}.csv").unlink(missing_ok=True)
            continue
        with (out / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v
                              for k, v in row.items()} for row in rows)
