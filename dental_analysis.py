"""Small offline diagnostics for saved dental runs. No inference or prediction changes.

Slices reuse the evaluator, including its unresolved-answer and conditional-location
rules. Retry rows describe recorded questions, not a counterfactual no-retry run.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import dental_eval as ev
from dental_pipeline import load_results


def _outcome(truth, answer):
    if answer not in ("A", "B"):
        return "unresolved"
    return ("TP" if truth else "FP") if answer == "A" else ("FN" if truth else "TN")


def presence_changes(gt, before, after, before_field="presence"):
    """Paired outcomes, including unresolved transitions; ALL and per finding."""
    groups = defaultdict(list)
    for image_id, entry in gt.items():
        for condition in sorted(entry["annotated"]):
            old = before[image_id]["findings"][condition]
            if before_field not in old:
                continue  # legacy artifacts without whole-image answers
            truth = any(b["condition"] == condition for b in entry["boxes"])
            transition = (_outcome(truth, old[before_field]) + " -> " +
                          _outcome(truth, after[image_id]["findings"][condition]["presence"]))
            for label in ("ALL", condition):
                groups[label, transition].append(image_id)
    return [{"condition": condition, "transition": transition, "checks": len(ids),
             "image_ids": sorted(set(ids))}
            for (condition, transition), ids in sorted(groups.items())]


def metrics(report):
    summary = report["summary"]
    row = {k: summary[k] for k in ("images_scored", "expected_finding_checks", "scored_finding_checks",
           "excluded_unparseable_checks", "TP", "TN", "FP", "FN", "sensitivity", "specificity",
           "ppv", "f1", "mean_false_alarms_per_image")}
    row["coverage"] = ev._ratio(row["scored_finding_checks"], row["expected_finding_checks"])
    # Weight by the actual scored cases, never average class percentages equally.
    for table, weight, metrics in (("counts", "n_scored", ("mae", "exact_rate", "within_1_rate")),
                                   ("regions", "n_localized_cases", ("exact_set_match_rate",))):
        rows = report[table]
        row[table + "_scored"] = sum(r[weight] for r in rows)
        for metric in metrics:
            usable = [r for r in rows if r[metric] is not None]
            row[table + "_" + metric] = ev._ratio(
                sum(r[metric] * r[weight] for r in usable), sum(r[weight] for r in usable))
    row["counts_unparseable"] = sum(r["count_unparseable"] for r in report["counts"])
    strict = [r for r in report["counts"] if r["strict_mae"] is not None]
    row["counts_strict_scored"] = sum(r["strict_n"] for r in strict)
    row["counts_strict_mae"] = ev._ratio(sum(r["strict_mae"] * r["strict_n"] for r in strict),
                                         row["counts_strict_scored"])
    row["regions_excluded"] = sum(r["excluded_location_checks"] for r in report["regions"])
    location = ({k: sum(r[k] for r in report["regions"]) for k in ("TP", "TN", "FP", "FN")}
                if report["regions"] else {k: None for k in ("TP", "TN", "FP", "FN")})
    row.update({"regions_" + k: value for k, value in location.items()})
    row["regions_f1"] = (ev._prf(location["TP"], location["FP"], location["TN"], location["FN"])["f1"]
                         if report["regions"] else None)
    row["region_presence_f1"] = (summary.get("region_presence") or {}).get("f1")  # None without region answers or location
    row["side_agreement_rate"] = (summary.get("side_agreement") or {}).get("agreement_rate")
    return row


def finding_rows(report):
    """One compact row per finding, joining presence, count, and location evidence."""
    whole = {r["condition"]: r for r in report.get("whole_image", [])}
    counts = {r["condition"]: r for r in report.get("counts", [])}
    regions = {r["condition"]: r for r in report.get("regions", [])}
    region_presence = defaultdict(list)
    for row in report.get("region_presence", []):
        region_presence[row["condition"]].append(row)

    rows = []
    for presence in report.get("presence", []):
        condition = presence["condition"]
        before, count, location = whole.get(condition, {}), counts.get(condition, {}), regions.get(condition, {})
        regional = region_presence.get(condition, [])
        regional_cells = {k: sum(r.get(k, 0) for r in regional) for k in ("TP", "TN", "FP", "FN")}
        regional_scores = (ev._prf(regional_cells["TP"], regional_cells["FP"],
                                   regional_cells["TN"], regional_cells["FN"])
                           if regional else {})
        rows.append({
            **presence,
            "whole_image_f1": before.get("f1"),
            "count_n_scored": count.get("n_scored"),
            "count_exact_rate": count.get("exact_rate"),
            "count_within_1_rate": count.get("within_1_rate"),
            "count_mae": count.get("mae"),
            "count_strict_mae": count.get("strict_mae"),
            "count_unparseable": count.get("count_unparseable"),
            "region_presence_TP": regional_cells["TP"] if regional else None,
            "region_presence_TN": regional_cells["TN"] if regional else None,
            "region_presence_FP": regional_cells["FP"] if regional else None,
            "region_presence_FN": regional_cells["FN"] if regional else None,
            "region_presence_f1": regional_scores.get("f1"),
            "localized_cases": location.get("n_localized_cases"),
            "location_TP": location.get("TP"),
            "location_TN": location.get("TN"),
            "location_FP": location.get("FP"),
            "location_FN": location.get("FN"),
            "location_f1": location.get("f1"),
            "region_exact_rate": location.get("exact_set_match_rate"),
            "region_jaccard": location.get("mean_jaccard"),
        })
    return rows


def call_usage(results):
    """Recorded model completions; transport attempts are not persisted as calls."""
    groups = defaultdict(list)
    for result in results.values():
        for call in result.get("calls", []):
            attempt = call.get("parse_recovery", {}).get("attempt")
            kind = "unknown" if attempt is None else "retry" if attempt > 1 else "first"
            groups[call.get("stage", "unknown"), kind].append(call)
    rows = []
    for (stage, attempt), calls in sorted(groups.items()):
        row = {"stage": stage, "attempt": attempt, "calls": len(calls)}
        for field in ("prompt_tokens", "completion_tokens", "latency_seconds"):
            values = [c[field] for c in calls if c.get(field) is not None]
            row[field + "_recorded_calls"] = len(values)
            row[field] = round(sum(values), 4) if values else None
        rows.append(row)
    return rows


def recovery_rows(gt, results, evaluate_location):
    groups = defaultdict(list)
    for image_id, entry in gt.items():
        # Attempts of one question share stage/condition/region. attempt=1 starts a new chain.
        chains = []
        active = {}
        for call in results[image_id].get("calls", []):
            recovery = call.get("parse_recovery")
            if not recovery or call.get("condition") not in entry["annotated"]:
                continue
            key = (call["stage"], call["condition"], call.get("region"))
            if recovery["attempt"] == 1:
                active[key] = [call]
                chains.append(active[key])
            elif key in active:
                active[key].append(call)
        for chain in chains:
            first, last = chain[0], chain[-1]
            stage, condition, region = first["stage"], first["condition"], first.get("region")
            initial = first["parse_recovery"]["value"]
            final = last["parse_recovery"]["value"]
            fields = ("presence", "count") if isinstance(initial, (list, tuple)) else (
                "count" if stage in ("count", "region_count") else "presence",)
            boxes = [b for b in entry["boxes"] if b["condition"] == condition]
            scheme = ev.result_scheme(results[image_id])
            truth_available = region is None or (evaluate_location and scheme != "none"
                                                 and not any(b.get("location_excluded") for b in boxes))
            for index, field in enumerate(fields):
                old = initial[index] if len(fields) == 2 else initial
                new = final[index] if len(fields) == 2 else final
                status = "first_pass" if old is not None else "recovered" if new is not None else "unresolved"
                correct = None
                if new is not None and truth_available:
                    if field == "presence":
                        truth = bool(boxes) if region is None else region in ev.gt_regions(boxes, scheme)
                        correct = (new == "A") == truth
                    else:
                        truth = len(boxes) if region is None else sum(
                            ev.box_primary_region(b, scheme) == region for b in boxes)
                        correct = new == truth
                groups[stage, field, status].append((image_id, correct))
    return [{"stage": stage, "field": field, "status": status, "checks": len(items),
             "correctness_scored": sum(ok is not None for _, ok in items),
             "correct": sum(ok is True for _, ok in items),
             "correct_rate": ev._ratio(sum(ok is True for _, ok in items),
                                       sum(ok is not None for _, ok in items)),
             "image_ids": sorted({i for i, _ in items})}
            for (stage, field, status), items in sorted(groups.items())]


def analyze(gt, results, *, dataset="dataset", evaluate_location=True):
    results = {i: results[i] for i in gt}
    buckets = defaultdict(lambda: defaultdict(set))
    regional = {}
    for image_id, entry in gt.items():
        result = results[image_id]
        if ev.result_protocol(result)["presence_level"] == "region":
            regional[image_id] = entry
        present = {b["condition"] for b in entry["boxes"]} & set(entry["annotated"])
        scheme = ev.result_scheme(result)
        for condition in entry["annotated"]:
            boxes = [b for b in entry["boxes"] if b["condition"] == condition]
            labels = [("finding_types_in_image", "0" if not present else "1-2" if len(present) <= 2 else "3+")]
            if boxes:
                labels.append(("instances_of_finding", "1" if len(boxes) == 1 else "2+"))
            else:
                labels.append(("absent_finding_context", "other_findings_present" if present else "no_annotated_findings"))
            if evaluate_location and scheme != "none" and boxes:
                # Use mixed/excluded groups rather than attributing a case to an arbitrary box.
                sources = {ev.box_source(b) for b in boxes}
                labels.append(("location_truth_source", next(iter(sources)) if len(sources) == 1 else "mixed"))
                if not any(b.get("location_excluded") for b in boxes):
                    regions = ev.gt_regions(boxes, scheme)
                    labels.append(("true_regions", "0" if not regions else "1" if len(regions) == 1 else "2+"))
                    labels.append(("crosses_region_boundary", "yes" if any(ev.straddling(b, scheme) for b in boxes) else "no"))
            for label in labels:
                buckets[label][image_id].add(condition)
    rows, condition_rows = [], []
    for (situation, group), members in sorted(buckets.items()):
        subset = {i: {**gt[i], "annotated": conditions} for i, conditions in members.items()}
        report = ev.evaluate(subset, results, dataset=dataset, evaluate_location=evaluate_location,
                             include_analysis=False)
        rows.append({"dataset": dataset, "situation": situation, "group": group,
                     **metrics(report), "image_ids": sorted(members)})
        for row in finding_rows(report):
            condition_rows.append({"situation": situation, "group": group, **row,
                                   "image_ids": sorted(i for i, conditions in members.items()
                                                       if row["condition"] in conditions)})
    return {"stage_changes": presence_changes(regional, results, results, "whole_image"),
            "parse_recovery": recovery_rows(gt, results, evaluate_location),
            "call_usage": call_usage(results), "case_breakdown": rows,
            "case_condition_breakdown": condition_rows}


def compare_runs(gt, run_dirs, *, dataset="dataset", evaluate_location=True):
    """Compare dataset run directories on exactly gt's image IDs and identical image hashes.

    First dictionary entry is the reference. Each directory contains manifest.json
    and results/. Extra saved images are allowed; missing selected images are errors.
    The same supplied ground truth is used for every run. Region schemes remain
    visible because localization scores at different granularities are not equivalent.
    """
    if len(run_dirs) < 2 or not gt:
        raise ValueError("comparison needs at least two runs and a non-empty ground-truth selection")
    rows, changes = [], []
    reference = reference_name = None
    for name, directory in run_dirs.items():
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if not isinstance(manifest.get("protocol"), dict) or manifest.get("mode") not in ("plain", "tagged"):
            raise ValueError(f"{name}: manifest needs a saved protocol and resolved mode")
        loaded = load_results(directory)
        missing = set(gt) - set(loaded)
        if missing:
            raise ValueError(f"{name}: missing results for {len(missing)} selected images: {sorted(missing)[:5]}")
        results = {i: loaded[i] for i in gt}
        for image_id, result in results.items():
            if not result.get("image_sha256"):
                raise ValueError(f"{name}/{image_id}: missing image_sha256; cannot verify paired comparison")
            if result.get("protocol") != manifest.get("protocol") or result.get("mode") != manifest.get("mode"):
                raise ValueError(f"{name}/{image_id}: saved protocol/mode differs from manifest")
            if reference is not None and result["image_sha256"] != reference[image_id]["image_sha256"]:
                raise ValueError(f"{name}/{image_id}: image content differs from reference")
        if reference is None:
            reference, reference_name = results, name
        report = ev.evaluate(gt, results, dataset=dataset, evaluate_location=evaluate_location, include_analysis=False)
        saved = ev.result_protocol(manifest)  # older manifests get the defaults of the knobs added since
        transitions = presence_changes(gt, reference, results)
        changes.extend({"run": name, "reference": reference_name, **r} for r in transitions)
        totals = {r["transition"]: r["checks"] for r in transitions if r["condition"] == "ALL"}
        # Paired metrics use only checks resolved by both runs; coverage still uses all checks.
        paired_gt = {i: {**entry, "annotated": {c for c in entry["annotated"]
                     if reference[i]["findings"][c]["presence"] in ("A", "B")
                     and results[i]["findings"][c]["presence"] in ("A", "B")}}
                     for i, entry in gt.items()}
        old = ev.evaluate(paired_gt, reference, evaluate_location=False, include_analysis=False)["summary"]
        new = ev.evaluate(paired_gt, results, evaluate_location=False, include_analysis=False)["summary"]
        row = {"run": name, "reference": reference_name, "dataset": dataset,
               "config_hash": manifest.get("hash"), "evaluate_location": evaluate_location,
               "location_truth": report["summary"]["location_truth"],
               "model": manifest.get("runner", {}).get("model"), "mode": manifest.get("mode"),
               "runner_settings": manifest.get("runner", {}),
               **{k: saved.get(k) for k in ("presence_level", "counting", "count_level", "region_scheme", "region_prompt",
                                            "question_form", "parse_retries")},
               **metrics(report),
               "paired_checks": new["scored_finding_checks"], "paired_reference_f1": old["f1"],
               "paired_run_f1": new["f1"],
               "paired_f1_delta": round(new["f1"] - old["f1"], 4) if None not in (old["f1"], new["f1"]) else None,
               "corrected": totals.get("FN -> TP", 0) + totals.get("FP -> TN", 0),
               "worsened": totals.get("TP -> FN", 0) + totals.get("TN -> FP", 0),
               "newly_resolved": sum(n for t, n in totals.items() if t.startswith("unresolved -> ") and not t.endswith("unresolved")),
               "newly_unresolved": sum(n for t, n in totals.items() if t.endswith(" -> unresolved") and not t.startswith("unresolved")),
               "image_ids": sorted(gt)}
        call_counts = [r["call_count"] for r in results.values() if r.get("call_count") is not None]
        row["call_counts_recorded_images"] = len(call_counts)
        row["mean_calls_per_image"] = ev._ratio(sum(call_counts), len(call_counts))
        calls = [c for r in results.values() for c in r.get("calls", [])]
        row["recorded_calls"] = len(calls)
        for field in ("prompt_tokens", "completion_tokens", "latency_seconds"):
            values = [c[field] for c in calls if c.get(field) is not None]
            row[field + "_recorded_calls"] = len(values)
            row[field] = round(sum(values), 4) if values else None
        rows.append(row)
    return {"run_comparison": rows, "run_changes": changes}


def compact_views(reports):
    """Join per-run reports into compact experiment, finding, and situation tables.

    ``reports`` is the notebook's ``{(experiment, dataset): report}`` mapping.
    The full per-experiment artifacts remain authoritative; these rows make the
    same metrics directly comparable without opening many separate CSV files.
    """
    overview, findings, situations, situation_findings = [], [], [], []
    for (experiment, dataset), report in sorted(reports.items()):
        summary, extra = report["summary"], metrics(report)
        protocol = summary.get("protocol") or {}
        overview.append({
            "dataset": dataset, "experiment": experiment,
            **{k: protocol.get(k) for k in ("presence_level", "counting", "count_level", "region_scheme",
                                             "region_prompt", "question_form", "parse_retries")},
            "evaluate_location": summary.get("evaluate_location"),
            "images": summary["images_scored"],
            "expected_checks": summary["expected_finding_checks"],
            "scored_checks": summary["scored_finding_checks"],
            "coverage": extra["coverage"],
            **{k: summary[k] for k in ("TP", "TN", "FP", "FN", "excluded_unparseable_checks",
                                        "unparseable_rate", "sensitivity", "specificity", "ppv", "f1", "macro_f1",
                                        "complete_case_rate", "mean_false_alarms_per_image", "mean_calls_per_image")},
            "count_n_scored": extra["counts_scored"],
            "count_exact_rate": extra["counts_exact_rate"],
            "count_within_1_rate": extra["counts_within_1_rate"],
            "count_mae": extra["counts_mae"],
            "count_strict_n": extra["counts_strict_scored"],
            "count_strict_mae": extra["counts_strict_mae"],
            "count_unparseable": extra["counts_unparseable"],
            "localized_cases": extra["regions_scored"],
            **{"location_" + k: extra["regions_" + k] for k in ("TP", "TN", "FP", "FN")},
            "location_f1": extra["regions_f1"],
            "region_exact_rate": extra["regions_exact_set_match_rate"],
            **{"region_presence_" + k: (summary.get("region_presence") or {}).get(k)
               for k in ("TP", "TN", "FP", "FN")},
            "region_presence_f1": extra["region_presence_f1"],
            "side_agreement_rate": extra["side_agreement_rate"],
            "location_excluded": extra["regions_excluded"],
        })
        findings.extend({"experiment": experiment, **row} for row in finding_rows(report))
        situations.extend({"experiment": experiment, **row} for row in report.get("case_breakdown", []))
        situation_findings.extend({"experiment": experiment, **row}
                                  for row in report.get("case_condition_breakdown", []))
    return {"experiment_overview": overview, "finding_comparison": findings,
            "situation_comparison": situations, "situation_finding_comparison": situation_findings}
