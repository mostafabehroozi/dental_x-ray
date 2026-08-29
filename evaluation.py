from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

from benchmark import LOCATION_VALUES, GeometryThresholds, YoloBenchmark, geometry_region
from prompts import CONDITIONS


VALID_STATUSES = {"PRESENT", "ABSENT", "UNCERTAIN"}
DEFAULT_COMPARISON_METRICS = (
    "overall_metrics.micro_recall",
    "overall_metrics.micro_precision",
    "overall_metrics.macro_f1",
    "overall_metrics.complete_image_rate",
    "overall_metrics.average_fp_per_image",
    "overall_metrics.average_fn_per_image",
    "overall_metrics.uncertainty_rate",
)


@dataclass(frozen=True)
class EvaluationConfig:
    evaluate_findings: bool = True
    evaluate_location: bool = False
    location_level: Literal[0, 1, 2] = 0
    location_adapters: tuple[Literal["geometry", "vision"], ...] = ("vision",)
    vision_backend: Literal["llm", "expert_model"] = "llm"
    evaluate_per_class: bool = True
    evaluate_per_image: bool = True
    evaluate_uncertainty: bool = True
    geometry_thresholds: GeometryThresholds = field(default_factory=GeometryThresholds)

    def __post_init__(self) -> None:
        if self.location_level not in {0, 1, 2}:
            raise ValueError("location_level must be 0, 1, or 2.")
        adapters = tuple(self.location_adapters)
        if len(adapters) != len(set(adapters)) or not set(adapters).issubset({"geometry", "vision"}):
            raise ValueError("location_adapters must contain unique 'geometry' and/or 'vision' values.")
        if self.evaluate_location and self.location_level == 0:
            raise ValueError("evaluate_location=True requires location_level 1 or 2.")
        if self.evaluate_location and not adapters:
            raise ValueError("At least one location adapter is required when location evaluation is enabled.")
        if self.evaluate_location and not self.evaluate_findings:
            raise ValueError("Location evaluation requires finding evaluation to identify true positives.")
        if self.vision_backend not in {"llm", "expert_model"}:
            raise ValueError("vision_backend must be 'llm' or 'expert_model'.")


@dataclass(frozen=True)
class NormalizedPrediction:
    statuses: dict[str, str]
    locations: dict[str, Any]


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _metrics_from_counts(counts: Mapping[str, int]) -> dict[str, float | int]:
    tp, tn, fp, fn = (counts[key] for key in ("tp", "tn", "fp", "fn"))
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "f1": _safe_divide(2 * tp, 2 * tp + fp + fn),
        "false_positive_rate": _safe_divide(fp, fp + tn),
        "false_negative_rate": _safe_divide(fn, fn + tp),
    }


def normalize_prediction(payload: Mapping[str, Any]) -> NormalizedPrediction:
    report = payload.get("evaluation_adaptation_report")
    if isinstance(report, Mapping):
        findings = report.get("findings")
    elif isinstance(payload.get("findings"), list):
        findings = payload.get("findings")
    else:
        findings = None

    statuses: dict[str, str] = {}
    locations: dict[str, Any] = {}
    if isinstance(findings, list):
        for index, finding in enumerate(findings):
            if not isinstance(finding, Mapping):
                raise ValueError(f"Prediction finding at index {index} must be an object.")
            condition = finding.get("condition")
            if not isinstance(condition, str) or condition not in CONDITIONS:
                raise ValueError(f"Prediction finding at index {index} has an invalid condition.")
            if condition in statuses:
                raise ValueError(f"Prediction contains duplicate condition {condition!r}.")
            status = str(finding.get("status", "")).upper()
            if status not in VALID_STATUSES:
                raise ValueError(f"Prediction for {condition} has invalid status {status!r}.")
            statuses[condition] = status
            locations[condition] = finding.get("location")
    else:
        raw_statuses = payload.get("deterministic_atomic_statuses")
        if not isinstance(raw_statuses, Mapping):
            raw_statuses = payload.get("statuses")
        if not isinstance(raw_statuses, Mapping):
            raise ValueError(
                "Prediction must contain evaluation_adaptation_report.findings, findings, "
                "deterministic_atomic_statuses, or statuses."
            )
        for condition, raw_status in raw_statuses.items():
            if condition not in CONDITIONS:
                raise ValueError(f"Prediction contains unknown condition {condition!r}.")
            status = str(raw_status).upper()
            if status not in VALID_STATUSES:
                raise ValueError(f"Prediction for {condition} has invalid status {status!r}.")
            statuses[str(condition)] = status

    missing = sorted(set(CONDITIONS) - set(statuses))
    extra = sorted(set(statuses) - set(CONDITIONS))
    if missing or extra:
        raise ValueError(f"Prediction ontology mismatch; missing={missing}, extra={extra}.")
    return NormalizedPrediction(statuses=statuses, locations=locations)


def _looks_like_prediction(payload: Mapping[str, Any]) -> bool:
    return any(
        key in payload
        for key in ("evaluation_adaptation_report", "findings", "deterministic_atomic_statuses", "statuses")
    )


def load_prediction_directory(
    prediction_dir: str | Path,
    expected_image_ids: Iterable[str],
) -> dict[str, dict[str, Any]]:
    directory = Path(prediction_dir)
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    predictions: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON prediction file: {path}") from exc
        if not isinstance(payload, dict) or not _looks_like_prediction(payload):
            continue
        raw_image_id = payload.get("image_id")
        if isinstance(raw_image_id, str) and raw_image_id:
            image_id = raw_image_id
        elif isinstance(payload.get("image_path"), str):
            image_id = Path(payload["image_path"]).stem
        else:
            raise ValueError(f"Prediction file {path} has no image_id or image_path.")
        if image_id in predictions:
            raise ValueError(f"Duplicate prediction for image ID {image_id!r}.")
        predictions[image_id] = payload

    expected = set(expected_image_ids)
    actual = set(predictions)
    if expected != actual:
        raise ValueError(
            f"Prediction image IDs do not match the benchmark; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}."
        )
    return predictions


def _normalize_component(value: Any, field_name: str) -> tuple[set[str], bool]:
    if value is None:
        return set(), False
    if isinstance(value, (list, tuple, set)):
        combined: set[str] = set()
        unrecognized = False
        for item in value:
            values, item_unrecognized = _normalize_component(item, field_name)
            combined.update(values)
            unrecognized = unrecognized or item_unrecognized
        return combined, unrecognized

    text = re.sub(r"[_-]+", " ", str(value).strip().lower())
    if not text or text in {"none", "null", "unknown", "uncertain", "n/a", "not applicable"}:
        return set(), False

    if field_name == "arch":
        values = set()
        if re.search(r"\b(upper|maxilla|maxillary)\b", text):
            values.add("upper")
        if re.search(r"\b(lower|mandible|mandibular)\b", text):
            values.add("lower")
        if text in {"both", "bilateral", "both arches"}:
            values.update({"upper", "lower"})
    elif field_name == "side":
        values = set()
        if re.search(r"\b(left)\b", text):
            values.add("left")
        if re.search(r"\b(right)\b", text):
            values.add("right")
        if re.search(r"\b(center|central|midline|anterior midline)\b", text):
            values.add("center")
        if text in {"both", "bilateral", "both sides"}:
            values.update({"left", "right"})
    elif field_name == "area":
        values = set()
        if re.search(r"\b(anterior|front)\b", text):
            values.add("anterior")
        if re.search(r"\b(posterior|back)\b", text):
            values.add("posterior")
        if text in {"both", "generalized", "anterior and posterior"}:
            values.update({"anterior", "posterior"})
    else:  # pragma: no cover - internal programming error
        raise KeyError(field_name)
    return values, not bool(values)


def _prediction_regions(location: Any, level: int) -> tuple[set[tuple[str, ...]], list[str]]:
    if location is None:
        return set(), []
    location_items = location if isinstance(location, list) else [location]
    regions: set[tuple[str, ...]] = set()
    unrecognized_fields: list[str] = []
    for item in location_items:
        if not isinstance(item, Mapping):
            unrecognized_fields.append("location")
            continue
        arches, bad_arch = _normalize_component(item.get("arch"), "arch")
        sides, bad_side = _normalize_component(item.get("side"), "side")
        if not sides and isinstance(item.get("distribution"), str):
            distribution_sides, distribution_bad = _normalize_component(item.get("distribution"), "side")
            sides.update(distribution_sides)
            bad_side = bad_side or distribution_bad
        areas, bad_area = _normalize_component(item.get("area"), "area")
        if bad_arch:
            unrecognized_fields.append("arch")
        if bad_side:
            unrecognized_fields.append("side")
        if level == 2 and bad_area:
            unrecognized_fields.append("area")
        if not arches or not sides or (level == 2 and not areas):
            continue
        value_groups = (sorted(arches), sorted(sides)) if level == 1 else (sorted(arches), sorted(sides), sorted(areas))
        regions.update(tuple(values) for values in product(*value_groups))
    return regions, sorted(set(unrecognized_fields))


def _load_vision_cache(vision_cache: Mapping[str, Any] | str | Path | None) -> Mapping[str, Any] | None:
    if vision_cache is None:
        return None
    if isinstance(vision_cache, Mapping):
        return vision_cache
    path = Path(vision_cache)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid vision location cache JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Vision location cache must be a JSON object.")
    return payload


def _evaluate_location_source(
    source: str,
    benchmark: YoloBenchmark,
    normalized: Mapping[str, NormalizedPrediction],
    true_positive_conditions: Mapping[str, set[str]],
    config: EvaluationConfig,
    vision_cache: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if source == "vision":
        if vision_cache is None:
            raise ValueError("The vision location adapter requires a prepared vision location cache.")
        if vision_cache.get("dataset_fingerprint") != benchmark.fingerprint:
            raise ValueError("Vision location cache does not match this benchmark dataset.")
        resolver_metadata = vision_cache.get("resolver")
        if not isinstance(resolver_metadata, Mapping) or resolver_metadata.get("backend") != config.vision_backend:
            raise ValueError("Vision location cache backend does not match EvaluationConfig.vision_backend.")
        vision_images = vision_cache.get("images")
        if not isinstance(vision_images, Mapping):
            raise ValueError("Vision location cache is missing its images mapping.")
    else:
        vision_images = {}

    location_tp = location_fp = location_fn = 0
    eligible_findings = scored_findings = unscorable_gt_locations = 0
    errors: list[dict[str, Any]] = []
    for image_id in benchmark.image_ids:
        image = benchmark.images[image_id]
        prediction = normalized[image_id]
        boxes_by_condition = {
            condition: [box for box in image.boxes if box.condition == condition]
            for condition in true_positive_conditions[image_id]
        }
        for condition in sorted(true_positive_conditions[image_id]):
            eligible_findings += 1
            gt_regions: set[tuple[str, ...]] = set()
            for box in boxes_by_condition[condition]:
                if source == "geometry":
                    gt_regions.add(geometry_region(box, config.geometry_thresholds, config.location_level))
                    continue
                image_cache = vision_images.get(image_id)
                if not isinstance(image_cache, Mapping):
                    raise ValueError(f"Vision location cache is missing image {image_id!r}.")
                locations = image_cache.get("locations")
                if not isinstance(locations, Mapping) or box.box_id not in locations:
                    raise ValueError(f"Vision location cache is missing {image_id}/{box.box_id}.")
                location = locations[box.box_id]
                if not isinstance(location, Mapping):
                    raise ValueError(f"Vision location {image_id}/{box.box_id} must be an object.")
                for component_name, allowed in LOCATION_VALUES.items():
                    if location.get(component_name) not in allowed:
                        raise ValueError(
                            f"Vision location {image_id}/{box.box_id} has invalid {component_name}."
                        )
                required = ("arch", "side") if config.location_level == 1 else ("arch", "side", "area")
                values = tuple(str(location.get(field, "unknown")) for field in required)
                if "unknown" in values:
                    unscorable_gt_locations += 1
                else:
                    gt_regions.add(values)

            if not gt_regions:
                continue
            scored_findings += 1
            predicted_regions, unrecognized_fields = _prediction_regions(
                prediction.locations.get(condition), config.location_level
            )
            matched = gt_regions & predicted_regions
            missed = gt_regions - predicted_regions
            extra = predicted_regions - gt_regions
            location_tp += len(matched)
            location_fn += len(missed)
            location_fp += len(extra)
            if missed or extra or unrecognized_fields:
                errors.append(
                    {
                        "image_id": image_id,
                        "condition": condition,
                        "missed_regions": [list(region) for region in sorted(missed)],
                        "extra_regions": [list(region) for region in sorted(extra)],
                        "unrecognized_prediction_fields": unrecognized_fields,
                    }
                )

    return {
        "level": config.location_level,
        "eligible_true_positive_findings": eligible_findings,
        "scored_findings": scored_findings,
        "unscorable_gt_locations": unscorable_gt_locations,
        "tp": location_tp,
        "fp": location_fp,
        "fn": location_fn,
        "precision": _safe_divide(location_tp, location_tp + location_fp),
        "recall": _safe_divide(location_tp, location_tp + location_fn),
        "f1": _safe_divide(2 * location_tp, 2 * location_tp + location_fp + location_fn),
        "missed_gt_regions": location_fn,
        "extra_predicted_regions": location_fp,
        "errors": errors,
    }


def evaluate_experiment(
    benchmark: YoloBenchmark,
    predictions: Mapping[str, Mapping[str, Any]] | str | Path,
    *,
    config: EvaluationConfig | None = None,
    experiment_metadata: Mapping[str, Any] | None = None,
    vision_cache: Mapping[str, Any] | str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    config = config or EvaluationConfig()
    if isinstance(predictions, (str, Path)):
        raw_predictions = load_prediction_directory(predictions, benchmark.image_ids)
    else:
        raw_predictions = {str(key): dict(value) for key, value in predictions.items()}
        expected, actual = set(benchmark.image_ids), set(raw_predictions)
        if expected != actual:
            raise ValueError(
                f"Prediction image IDs do not match the benchmark; "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}."
            )

    normalized = {image_id: normalize_prediction(raw_predictions[image_id]) for image_id in benchmark.image_ids}
    class_counts = {
        condition: {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "uncertain": 0}
        for condition in CONDITIONS
    }
    per_image_internal: list[dict[str, Any]] = []
    finding_errors: list[dict[str, Any]] = []
    true_positive_conditions: dict[str, set[str]] = {}

    if config.evaluate_findings:
        for image_id in benchmark.image_ids:
            gt = benchmark.images[image_id].findings
            prediction = normalized[image_id]
            correct: list[str] = []
            missed: list[str] = []
            false_added: list[str] = []
            uncertain: list[str] = []
            true_positive_conditions[image_id] = set()
            for condition in CONDITIONS:
                gt_present = gt[condition]
                status = prediction.statuses[condition]
                if status == "UNCERTAIN":
                    class_counts[condition]["uncertain"] += 1
                    uncertain.append(condition)
                if gt_present and status == "PRESENT":
                    class_counts[condition]["tp"] += 1
                    correct.append(condition)
                    true_positive_conditions[image_id].add(condition)
                elif not gt_present and status == "ABSENT":
                    class_counts[condition]["tn"] += 1
                elif gt_present:
                    class_counts[condition]["fn"] += 1
                    missed.append(condition)
                else:
                    class_counts[condition]["fp"] += 1
                    false_added.append(condition)

            gt_positive_count = sum(gt.values())
            coverage = _safe_divide(len(correct), gt_positive_count)
            image_metrics = {
                "image_id": image_id,
                "gt_positive_count": gt_positive_count,
                "correctly_detected_count": len(correct),
                "missed_count": len(missed),
                "falsely_added_count": len(false_added),
                "finding_coverage": coverage,
                "all_gt_findings_discovered": not missed,
                "correctly_detected": correct,
                "missed": missed,
                "falsely_added": false_added,
            }
            if config.evaluate_uncertainty:
                image_metrics["uncertain_count"] = len(uncertain)
                image_metrics["uncertain"] = uncertain
            per_image_internal.append(image_metrics)
            if missed or false_added or uncertain:
                error = {
                    "image_id": image_id,
                    "false_negatives": missed,
                    "false_positives": false_added,
                }
                if config.evaluate_uncertainty:
                    error["uncertain"] = uncertain
                finding_errors.append(error)

        per_class_internal: dict[str, dict[str, Any]] = {}
        for condition in CONDITIONS:
            metrics = _metrics_from_counts(class_counts[condition])
            if config.evaluate_uncertainty:
                metrics["uncertain_count"] = class_counts[condition]["uncertain"]
                metrics["uncertainty_rate"] = _safe_divide(
                    class_counts[condition]["uncertain"], len(benchmark.images)
                )
            per_class_internal[condition] = metrics

        totals = {
            key: sum(class_counts[condition][key] for condition in CONDITIONS)
            for key in ("tp", "tn", "fp", "fn")
        }
        micro = _metrics_from_counts(totals)
        image_count = len(benchmark.images)
        decision_count = image_count * len(CONDITIONS)
        complete_count = sum(row["all_gt_findings_discovered"] for row in per_image_internal)
        overall_metrics: dict[str, Any] = {
            "macro_precision": _mean(per_class_internal[c]["precision"] for c in CONDITIONS),
            "macro_recall": _mean(per_class_internal[c]["recall"] for c in CONDITIONS),
            "macro_f1": _mean(per_class_internal[c]["f1"] for c in CONDITIONS),
            "micro_precision": micro["precision"],
            "micro_recall": micro["recall"],
            "micro_f1": micro["f1"],
            "accuracy": _safe_divide(totals["tp"] + totals["tn"], decision_count),
            "average_fp_per_image": _safe_divide(totals["fp"], image_count),
            "average_fn_per_image": _safe_divide(totals["fn"], image_count),
            "average_false_findings_per_image": _safe_divide(totals["fp"], image_count),
            "average_missed_findings_per_image": _safe_divide(totals["fn"], image_count),
            "mean_finding_coverage": _mean(row["finding_coverage"] for row in per_image_internal),
            "complete_image_count": complete_count,
            "complete_image_rate": _safe_divide(complete_count, image_count),
            "total_tp": totals["tp"],
            "total_tn": totals["tn"],
            "total_fp": totals["fp"],
            "total_fn": totals["fn"],
        }
        if config.evaluate_uncertainty:
            uncertain_count = sum(class_counts[c]["uncertain"] for c in CONDITIONS)
            overall_metrics["uncertain_prediction_count"] = uncertain_count
            overall_metrics["uncertainty_rate"] = _safe_divide(uncertain_count, decision_count)
    else:
        per_class_internal = {}
        overall_metrics = {}
        for image_id in benchmark.image_ids:
            true_positive_conditions[image_id] = set()

    location_metrics: dict[str, Any] = {}
    if config.evaluate_location:
        loaded_vision_cache = _load_vision_cache(vision_cache)
        for source in config.location_adapters:
            location_metrics[source] = _evaluate_location_source(
                source,
                benchmark,
                normalized,
                true_positive_conditions,
                config,
                loaded_vision_cache,
            )

    result = {
        "schema_version": 1,
        "evaluation_config": asdict(config),
        "dataset": {
            "fingerprint": benchmark.fingerprint,
            "image_count": len(benchmark.images),
            "image_ids": list(benchmark.image_ids),
        },
        "experiment_metadata": dict(experiment_metadata or {}),
        "overall_metrics": overall_metrics,
        "per_class_metrics": per_class_internal if config.evaluate_per_class else {},
        "per_image_metrics": per_image_internal if config.evaluate_per_image else [],
        "location_metrics": location_metrics,
        "errors": {
            "findings": finding_errors,
            "locations": {source: metrics["errors"] for source, metrics in location_metrics.items()},
        },
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _metric_value(payload: Mapping[str, Any], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(f"Evaluation result has no metric path {path!r}.")
        value = value[part]
    return value


def compare_experiments(
    results: Iterable[Mapping[str, Any] | str | Path],
    metric_paths: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    loaded: list[tuple[Mapping[str, Any], str | None]] = []
    for item in results:
        if isinstance(item, Mapping):
            loaded.append((item, None))
        else:
            path = Path(item)
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise ValueError(f"Evaluation result must be a JSON object: {path}")
            loaded.append((payload, str(path)))
    if not loaded:
        return []

    reference_dataset = loaded[0][0].get("dataset")
    if not isinstance(reference_dataset, Mapping):
        raise ValueError("Evaluation result is missing dataset metadata.")
    reference_fingerprint = reference_dataset.get("fingerprint")
    reference_ids = reference_dataset.get("image_ids")
    for payload, _ in loaded[1:]:
        dataset = payload.get("dataset")
        if not isinstance(dataset, Mapping):
            raise ValueError("Evaluation result is missing dataset metadata.")
        if dataset.get("fingerprint") != reference_fingerprint or dataset.get("image_ids") != reference_ids:
            raise ValueError("Experiments can only be compared on the exact same benchmark images.")

    selected_metrics = tuple(metric_paths or DEFAULT_COMPARISON_METRICS)
    rows: list[dict[str, Any]] = []
    for index, (payload, source_path) in enumerate(loaded, start=1):
        metadata = payload.get("experiment_metadata")
        name = metadata.get("name") if isinstance(metadata, Mapping) else None
        if not name:
            name = Path(source_path).parent.name if source_path else f"experiment_{index}"
        row = {"experiment": name}
        for metric_path in selected_metrics:
            row[metric_path.split(".")[-1]] = _metric_value(payload, metric_path)
        rows.append(row)
    return rows
