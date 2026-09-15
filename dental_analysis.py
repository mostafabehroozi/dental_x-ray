"""Offline DentVLM diagnostics. Reuses the branch's scorer and vote/OR rules.

Unasked findings are not unresolved; named-cell multiplicity is not a tooth count.
Saved phrasings include any parse repairs, so replay is not a retries-OFF run.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import dental_eval as ev
import dental_pipeline as dp


def _outcome(truth, finding, field="presence"):
    if not finding["asked"]:
        return "not_assessed"
    answer = finding.get(field)
    if answer not in ("yes", "no"):
        return "unresolved"
    return ("TP" if truth else "FP") if answer == "yes" else ("FN" if truth else "TN")


def presence_changes(gt, before, after, comparison, before_field="presence", after_field="presence"):
    groups = defaultdict(list)
    for image_id, entry in gt.items():
        for condition in sorted(entry["annotated"]):
            old, new = before[image_id]["findings"][condition], after[image_id]["findings"][condition]
            if before_field not in old or after_field not in new:
                continue
            truth = any(b["condition"] == condition for b in entry["boxes"])
            transition = _outcome(truth, old, before_field) + " -> " + _outcome(truth, new, after_field)
            for label in ("ALL", condition):
                groups[label, transition].append(image_id)
    return [{"comparison": comparison, "condition": condition, "transition": transition,
             "checks": len(ids), "image_ids": sorted(set(ids))}
            for (condition, transition), ids in sorted(groups.items())]


def _score(gt, results, evaluate_location):
    return ev.evaluate(gt, results, evaluate_location=evaluate_location, include_analysis=False)


def metrics(gt, report):
    summary = report["summary"]
    row = {k: summary[k] for k in ("images_scored", "expected_finding_checks", "scored_finding_checks",
           "excluded_unparseable_checks", "TP", "TN", "FP", "FN", "sensitivity", "specificity", "ppv", "f1")}
    row["annotated_checks"] = sum(len(e["annotated"]) for e in gt.values())
    row["not_assessed_checks"] = row["annotated_checks"] - row["expected_finding_checks"]
    row["coverage"] = ev._ratio(row["scored_finding_checks"], row["expected_finding_checks"])
    for table, weight, metrics in (("counts", "n_scored", ("mae", "exact_rate", "within_1_rate")),
                                   ("regions", "n_localized_cases", ("exact_set_match_rate", "mean_jaccard"))):
        rows = report[table]
        row[table + "_scored"] = sum(r[weight] for r in rows)
        for metric in metrics:
            usable = [r for r in rows if r[metric] is not None]
            row[table + "_" + metric] = ev._ratio(
                sum(r[metric] * r[weight] for r in usable), sum(r[weight] for r in usable))
    strict = [r for r in report["counts"] if r["strict_mae"] is not None]
    row["counts_strict_scored"] = sum(r["strict_n"] for r in strict)
    row["counts_strict_mae"] = ev._ratio(sum(r["strict_mae"] * r["strict_n"] for r in strict),
                                         row["counts_strict_scored"])
    row["counts_unparseable"] = sum(r["count_unparseable"] for r in report["counts"])
    row["regions_excluded"] = sum(r["excluded_location_checks"] for r in report["regions"])
    location = ({k: sum(r[k] for r in report["regions"]) for k in ("TP", "TN", "FP", "FN")}
                if report["regions"] else {k: None for k in ("TP", "TN", "FP", "FN")})
    row.update({"regions_" + k: value for k, value in location.items()})
    row["regions_f1"] = (ev._prf(location["TP"], location["FP"], location["TN"], location["FN"])["f1"]
                         if report["regions"] else None)
    row["region_presence_f1"] = (summary.get("region_presence") or {}).get("f1")  # None without cells or location
    row["side_agreement_rate"] = (summary.get("side_agreement") or {}).get("agreement_rate")
    return row


def finding_rows(gt, report):
    """Every annotated finding, keeping not-assessed distinct from unresolved."""
    summary = report["summary"]
    dataset = summary["dataset"]
    presence = {r["condition"]: r for r in report.get("presence", [])}
    whole = {r["condition"]: r for r in report.get("whole_image", [])}
    counts = {r["condition"]: r for r in report.get("counts", [])}
    regions = {r["condition"]: r for r in report.get("regions", [])}
    region_presence = defaultdict(list)
    for row in report.get("region_presence", []):
        region_presence[row["condition"]].append(row)

    rows = []
    for condition in dp.CONDITIONS:
        annotated = [entry for entry in gt.values() if condition in entry["annotated"]]
        if not annotated:
            continue
        base = presence.get(condition)
        annotated_positives = sum(any(b["condition"] == condition for b in entry["boxes"])
                                  for entry in annotated)
        if base is None:
            base = {"dataset": dataset, "condition": condition, "trained_task": condition in dp.TRAINED,
                    "images": 0, "positives": 0,
                    **{k: None for k in ("TP", "TN", "FP", "FN")}, "unparseable": 0,
                    **{k: None for k in ("sensitivity", "specificity", "ppv", "f1")}}
        scored = sum(base[k] or 0 for k in ("TP", "TN", "FP", "FN"))
        before, count, location = whole.get(condition, {}), counts.get(condition, {}), regions.get(condition, {})
        regional = region_presence.get(condition, [])
        regional_cells = {k: sum(r.get(k, 0) for r in regional) for k in ("TP", "TN", "FP", "FN")}
        regional_scores = (ev._prf(regional_cells["TP"], regional_cells["FP"],
                                   regional_cells["TN"], regional_cells["FN"])
                           if regional else {})
        not_assessed_checks = len(annotated) - base["images"]
        assessment_status = ("not_assessed" if not base["images"] else
                             "partly_assessed" if not_assessed_checks else "assessed")
        rows.append({
            **base,
            "assessment_status": assessment_status,
            "annotated_images": len(annotated),
            "annotated_positives": annotated_positives,
            "not_assessed_checks": not_assessed_checks,
            "scored_checks": scored,
            "coverage": ev._ratio(scored, base["images"]),
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
            "location_TP": location.get("TP"), "location_TN": location.get("TN"),
            "location_FP": location.get("FP"), "location_FN": location.get("FN"),
            "location_f1": location.get("f1"),
            "region_exact_rate": location.get("exact_set_match_rate"),
            "region_jaccard": location.get("mean_jaccard"),
        })
    return rows


def call_usage(results):
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
    # Individual crown/bridge questions cannot be scored from a merged restoration label.
    task_conditions = {t: c for c in dp.CONDITIONS for t in dp.condition_tasks(c, True)
                       if len(dp.condition_tasks(c, True)) == 1}
    groups = defaultdict(list)
    for image_id, entry in gt.items():
        chains, active = [], {}
        for call in results[image_id].get("calls", []):
            recovery = call.get("parse_recovery")
            if not recovery:
                continue
            key = (call["stage"], call["task"], call.get("cell"))
            if recovery["attempt"] == 1:
                active[key] = [call]
                chains.append(active[key])
            elif key in active:
                active[key].append(call)
        for chain in chains:
            first, last = chain[0], chain[-1]
            stage, task, cell = first["stage"], first["task"], first.get("cell")
            old, new = first["parse_recovery"]["value"], last["parse_recovery"]["value"]
            status = "first_pass" if old is not None else "recovered" if new is not None else "unresolved"
            condition = task if stage == "count" else task_conditions.get(task)
            correct = None
            if condition in entry["annotated"] and new is not None:
                boxes = [b for b in entry["boxes"] if b["condition"] == condition]
                if stage == "count":
                    correct = new == len(boxes)
                elif cell is None:
                    correct = (new == "yes") == bool(boxes)
                elif evaluate_location and not any(b.get("location_excluded") for b in boxes):
                    correct = (new == "yes") == (cell in ev.gt_regions(boxes))
            groups[stage, task, status].append((image_id, correct))
    return [{"stage": stage, "task": task, "status": status, "checks": len(items),
             "correctness_scored": sum(ok is not None for _, ok in items),
             "correct": sum(ok is True for _, ok in items),
             "correct_rate": ev._ratio(sum(ok is True for _, ok in items), sum(ok is not None for _, ok in items)),
             "image_ids": sorted({i for i, _ in items})}
            for (stage, task, status), items in sorted(groups.items())]


def phrasing_analysis(gt, results, evaluate_location):
    first_results, eligible, votes = {}, {}, defaultdict(list)
    for image_id, entry in gt.items():
        result = results[image_id]
        tasks = result.get("tasks", {})
        for task, record in tasks.items():
            answers = record.get("answers", [])
            if len(answers) < 2:
                continue
            yes = sum(a["answer"] == "yes" for a in answers)
            no = sum(a["answer"] == "no" for a in answers)
            status = ("all_unresolved" if not yes + no else "tie" if yes == no else
                      "disagreement" if yes and no else "agreement")
            votes[task, status].append((image_id, len(answers) - yes - no))
        findings, conditions = {}, set()
        for condition in entry["annotated"]:
            finding = result["findings"][condition]
            keys = finding.get("tasks", [])
            if (not finding["asked"] or "whole_image" not in finding or not keys
                    or any(not tasks.get(k, {}).get("answers") for k in keys)
                    or not any(len(tasks[k]["answers"]) > 1 for k in keys)):
                continue
            conditions.add(condition)
            findings[condition] = {**finding, "presence": dp._any_yes(tasks[k]["answers"][0]["answer"] for k in keys)}
        if conditions:
            eligible[image_id] = {**entry, "annotated": conditions}
            first_results[image_id] = {**result, "findings": {**result["findings"], **findings}}
    changes = presence_changes(eligible, first_results, results, "first_phrasing_to_vote", after_field="whole_image")
    vote_rows = [{"task": task, "status": status, "checks": len(items),
                  "unresolved_phrasings": sum(n for _, n in items), "image_ids": sorted({i for i, _ in items})}
                 for (task, status), items in sorted(votes.items())]
    region_rows = []
    # Rationale regions only: crops overwrite task regions and do not use region_vote.
    subset = {i: e for i, e in eligible.items() if results[i]["location_level"] == "rationale"}
    if evaluate_location and subset:
        for mode in ("union", "majority"):
            replay = {}
            for image_id, entry in subset.items():
                result = results[image_id]
                findings = dict(result["findings"])
                for condition in entry["annotated"]:
                    finding = findings[condition]
                    decisions = [dp.vote(result["tasks"][k]["answers"], mode) for k in finding["tasks"]]
                    presence = dp._any_yes(d["presence"] for d in decisions)
                    regions = sorted({r for d in decisions if d["presence"] == "yes" for r in (d["regions"] or [])})
                    findings[condition] = {**finding, "presence": presence,
                                           "regions": regions if presence == "yes" else None,
                                           "region_count": len(regions) if presence == "yes" else None}
                replay[image_id] = {**result, "findings": findings}
            region_rows.append({"region_vote": mode, **metrics(subset, _score(subset, replay, True)),
                                "image_ids": sorted(subset)})
    return changes, vote_rows, region_rows


def analyze(gt, results, *, dataset="dataset", evaluate_location=True):
    results = {i: results[i] for i in gt}
    buckets = defaultdict(lambda: defaultdict(set))
    for image_id, entry in gt.items():
        result = results[image_id]
        present = {b["condition"] for b in entry["boxes"]} & set(entry["annotated"])
        for condition in entry["annotated"]:
            finding = result["findings"][condition]
            labels = [("task_support", "trained" if condition in dp.TRAINED else "untrained")]
            if finding["asked"]:
                boxes = [b for b in entry["boxes"] if b["condition"] == condition]
                labels.append(("finding_types_in_image", "0" if not present else "1-2" if len(present) <= 2 else "3+"))
                labels.append(("instances_of_finding", "1" if len(boxes) == 1 else "2+") if boxes else
                              ("absent_finding_context", "other_findings_present" if present else "no_annotated_findings"))
                if evaluate_location and result["location_level"] != "none":
                    if finding["presence"] == "yes":
                        named = finding["regions"]
                        labels.append(("predicted_named_cells", "unresolved" if named is None else
                                       "0" if not named else "1" if len(named) == 1 else "2+"))
                    if boxes:
                        sources = {ev.box_source(b) for b in boxes}
                        labels.append(("location_truth_source", next(iter(sources)) if len(sources) == 1 else "mixed"))
                        if not any(b.get("location_excluded") for b in boxes):
                            regions = ev.gt_regions(boxes)
                            labels.append(("true_cells", "0" if not regions else "1" if len(regions) == 1 else "2+"))
                            labels.append(("crosses_cell_boundary", "yes" if any(ev.straddling(b) for b in boxes) else "no"))
            for label in labels:
                buckets[label][image_id].add(condition)
    rows, condition_rows = [], []
    for (situation, group), members in sorted(buckets.items()):
        subset = {i: {**gt[i], "annotated": conditions} for i, conditions in members.items()}
        report = ev.evaluate(subset, results, dataset=dataset, evaluate_location=evaluate_location,
                             include_analysis=False)
        rows.append({"dataset": dataset, "situation": situation, "group": group,
                     **metrics(subset, report), "image_ids": sorted(members)})
        for row in finding_rows(subset, report):
            condition_rows.append({"situation": situation, "group": group, **row,
                                   "image_ids": sorted(i for i, conditions in members.items()
                                                       if row["condition"] in conditions)})
    crops = {i: e for i, e in gt.items() if results[i]["location_level"] == "crops"}
    changes, votes, region_votes = phrasing_analysis(gt, results, evaluate_location)
    changes += presence_changes(crops, results, results, "whole_image_to_crops", before_field="whole_image")
    return {"stage_changes": changes, "phrasing_votes": votes, "region_vote_comparison": region_votes,
            "parse_recovery": recovery_rows(gt, results, evaluate_location),
            "call_usage": call_usage(results), "case_breakdown": rows,
            "case_condition_breakdown": condition_rows}


def compare_runs(gt, run_dirs, *, dataset="dataset", evaluate_location=True):
    """First named dataset directory is the reference; rescore selected images on one truth.

    Pair only findings asked and resolved in both runs. Additional assessed classes
    are reported separately, especially when ask_untrained changes.
    """
    if len(run_dirs) < 2 or not gt:
        raise ValueError("comparison needs at least two runs and non-empty ground truth")
    rows, changes, reference, reference_name = [], [], None, None
    for name, directory in run_dirs.items():
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if not isinstance(manifest.get("protocol"), dict):
            raise ValueError(f"{name}: manifest needs a saved DentVLM protocol")
        loaded = dp.load_results(directory)
        missing = set(gt) - set(loaded)
        if missing:
            raise ValueError(f"{name}: missing selected images: {sorted(missing)[:5]}")
        results = {i: loaded[i] for i in gt}
        for image_id, result in results.items():
            if not result.get("image_sha256"):
                raise ValueError(f"{name}/{image_id}: missing image_sha256")
            if result.get("protocol") != manifest["protocol"] or result["location_level"] != manifest["protocol"].get("location"):
                raise ValueError(f"{name}/{image_id}: saved protocol differs from manifest")
            if reference is not None:
                if result["image_sha256"] != reference[image_id]["image_sha256"]:
                    raise ValueError(f"{name}/{image_id}: image content differs from reference")
                if evaluate_location and result.get("left_is_image_left") != reference[image_id].get("left_is_image_left"):
                    raise ValueError(f"{name}/{image_id}: cell-side conventions differ")
        if reference is None:
            reference, reference_name = results, name
        report = _score(gt, results, evaluate_location)
        transitions = presence_changes(gt, reference, results, "reference_to_run")
        changes.extend({"run": name, "reference": reference_name, **r} for r in transitions)
        totals = {r["transition"]: r["checks"] for r in transitions if r["condition"] == "ALL"}
        paired = {i: {**e, "annotated": {c for c in e["annotated"]
                  if all(run[i]["findings"][c]["asked"] and run[i]["findings"][c]["presence"] in ("yes", "no")
                         for run in (reference, results))}} for i, e in gt.items()}
        old, new = _score(paired, reference, False)["summary"], _score(paired, results, False)["summary"]
        row = {"run": name, "reference": reference_name, "dataset": dataset, "config_hash": manifest.get("hash"),
               "model": manifest.get("runner", {}).get("model"), "runner_settings": manifest.get("runner", {}),
               **{k: manifest["protocol"].get(k) for k in ("phrasings", "region_vote", "location", "count_question",
                                                         "ask_untrained", "extra_tasks", "parse_retries")},
               "evaluate_location": evaluate_location, "location_truth": report["summary"]["location_truth"],
               **metrics(gt, report), "paired_checks": new["scored_finding_checks"],
               "paired_reference_f1": old["f1"], "paired_run_f1": new["f1"],
               "paired_f1_delta": round(new["f1"] - old["f1"], 4) if None not in (old["f1"], new["f1"]) else None,
               "corrected": totals.get("FN -> TP", 0) + totals.get("FP -> TN", 0),
               "worsened": totals.get("TP -> FN", 0) + totals.get("TN -> FP", 0), "image_ids": sorted(gt)}
        for state in ("not_assessed", "unresolved"):
            row["left_" + state] = sum(n for t, n in totals.items() if t.startswith(state + " -> ") and not t.endswith(state))
            row["became_" + state] = sum(n for t, n in totals.items() if t.endswith(" -> " + state) and not t.startswith(state))
        counts = [r["call_count"] for r in results.values() if r.get("call_count") is not None]
        row["call_counts_recorded_images"] = len(counts)
        row["mean_calls_per_image"] = ev._ratio(sum(counts), len(counts))
        usage = call_usage(results)
        row["recorded_calls"] = sum(r["calls"] for r in usage)
        for field in ("prompt_tokens", "completion_tokens", "latency_seconds"):
            recorded = sum(r[field + "_recorded_calls"] for r in usage)
            row[field + "_recorded_calls"] = recorded
            row[field] = round(sum(r[field] or 0 for r in usage), 4) if recorded else None
        rows.append(row)
    return {"run_comparison": rows, "run_changes": changes}


def compact_views(ground_truth, reports):
    """Join DentVLM reports without losing task support, votes, crops, or not-assessed states."""
    overview, findings, situations, situation_findings = [], [], [], []
    stages, phrasings, vote_replays, recoveries, usage = [], [], [], [], []
    for (experiment, dataset), report in sorted(reports.items()):
        gt = ground_truth[dataset]
        summary, extra = report["summary"], metrics(gt, report)
        protocol = summary.get("protocol") or {}
        overview.append({
            "dataset": dataset, "experiment": experiment,
            **{k: protocol.get(k) for k in ("phrasings", "region_vote", "location", "count_question",
                                             "ask_untrained", "extra_tasks", "parse_retries")},
            "evaluate_location": summary.get("evaluate_location"),
            "images": summary["images_scored"], "annotated_checks": extra["annotated_checks"],
            "not_assessed_checks": extra["not_assessed_checks"],
            "expected_checks": summary["expected_finding_checks"],
            "scored_checks": summary["scored_finding_checks"], "coverage": extra["coverage"],
            **{k: summary[k] for k in ("TP", "TN", "FP", "FN", "excluded_unparseable_checks",
                                        "unparseable_rate", "sensitivity", "specificity", "ppv", "f1", "macro_f1",
                                        "complete_case_rate", "mean_false_alarms_per_image", "mean_calls_per_image")},
            "count_n_scored": extra["counts_scored"], "count_exact_rate": extra["counts_exact_rate"],
            "count_within_1_rate": extra["counts_within_1_rate"], "count_mae": extra["counts_mae"],
            "count_strict_n": extra["counts_strict_scored"], "count_strict_mae": extra["counts_strict_mae"],
            "count_unparseable": extra["counts_unparseable"],
            "localized_cases": extra["regions_scored"],
            **{"location_" + k: extra["regions_" + k] for k in ("TP", "TN", "FP", "FN")},
            "location_f1": extra["regions_f1"], "region_exact_rate": extra["regions_exact_set_match_rate"],
            "region_jaccard": extra["regions_mean_jaccard"],
            **{"region_presence_" + k: (summary.get("region_presence") or {}).get(k)
               for k in ("TP", "TN", "FP", "FN")},
            "region_presence_f1": extra["region_presence_f1"],
            "side_agreement_rate": extra["side_agreement_rate"], "location_excluded": extra["regions_excluded"],
        })
        findings.extend({"experiment": experiment, **row} for row in finding_rows(gt, report))
        situations.extend({"experiment": experiment, **row} for row in report.get("case_breakdown", []))
        situation_findings.extend({"experiment": experiment, **row}
                                  for row in report.get("case_condition_breakdown", []))
        for source, target in (("stage_changes", stages), ("phrasing_votes", phrasings),
                               ("region_vote_comparison", vote_replays), ("parse_recovery", recoveries),
                               ("call_usage", usage)):
            target.extend({"dataset": dataset, "experiment": experiment, **row}
                          for row in report.get(source, []))
    return {"experiment_overview": overview, "finding_comparison": findings,
            "situation_comparison": situations, "situation_finding_comparison": situation_findings,
            "stage_comparison": stages, "phrasing_comparison": phrasings,
            "vote_replay_comparison": vote_replays, "parse_recovery_comparison": recoveries,
            "call_usage_comparison": usage}
