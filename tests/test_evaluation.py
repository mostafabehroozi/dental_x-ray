from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from benchmark import (
    GeometryThresholds,
    VisionLocationResolver,
    geometry_region,
    load_yolo_benchmark,
    prepare_vision_location_cache,
    validate_dataset_yaml,
)
from evaluation import (
    EvaluationConfig,
    compare_experiments,
    evaluate_experiment,
    normalize_prediction,
)
from prompts import CONDITIONS


def _statuses(default: str = "ABSENT") -> dict[str, str]:
    return {condition: default for condition in CONDITIONS}


def _adaptation_prediction(statuses: dict[str, str], locations: dict | None = None) -> dict:
    locations = locations or {}
    return {
        "evaluation_adaptation_report": {
            "findings": [
                {
                    "condition": condition,
                    "status": statuses[condition],
                    "evidence": "synthetic test",
                    "location": locations.get(condition),
                    "conflict": None,
                }
                for condition in CONDITIONS
            ],
            "adaptation_notes": "",
        }
    }


class FakeExpertRunner:
    model_id = "fake-vision-model"

    def __init__(self) -> None:
        self.calls = 0

    def ask(self, image_path, question, **kwargs):
        self.calls += 1
        self.last_image_path = image_path
        self.last_question = question
        return {
            "raw_answer": (
                '<answer>{"locations":['
                '{"box_id":"box_0000","arch":"upper","side":"right","area":"posterior"}'
                "]}</answer>"
            )
        }


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.images_dir = self.root / "images"
        self.labels_dir = self.root / "labels"
        self.images_dir.mkdir()
        self.labels_dir.mkdir()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _image(self, name: str) -> None:
        Image.new("RGB", (100, 80), "white").save(self.images_dir / name)

    def test_yolo_parser_enumerates_negative_images_and_presence(self) -> None:
        self._image("positive.png")
        self._image("negative.png")
        (self.labels_dir / "positive.txt").write_text(
            "0 0.2 0.25 0.1 0.1\n4 0.7 0.6 0.1 0.1\n4 0.8 0.6 0.1 0.1\n",
            encoding="utf-8",
        )

        benchmark = load_yolo_benchmark(self.images_dir, self.labels_dir)

        self.assertEqual(benchmark.image_ids, ("negative", "positive"))
        self.assertFalse(any(benchmark.images["negative"].findings.values()))
        self.assertTrue(benchmark.images["positive"].findings["dental_implant"])
        self.assertTrue(benchmark.images["positive"].findings["carious_lesion"])
        self.assertEqual(len(benchmark.images["positive"].boxes), 3)

    def test_yolo_parser_rejects_invalid_class_and_yaml_order(self) -> None:
        self._image("sample.png")
        (self.labels_dir / "sample.txt").write_text("14 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unknown class ID 14"):
            load_yolo_benchmark(self.images_dir, self.labels_dir)

        yaml_path = self.root / "data.yaml"
        wrong_names = list(CONDITIONS)
        wrong_names[0], wrong_names[1] = wrong_names[1], wrong_names[0]
        yaml_path.write_text("names:\n" + "".join(f"  - {name}\n" for name in wrong_names), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "class order"):
            validate_dataset_yaml(yaml_path)

    def test_prediction_normalization_requires_complete_unique_ontology(self) -> None:
        statuses = _statuses()
        statuses["dental_implant"] = "PRESENT"
        normalized = normalize_prediction(_adaptation_prediction(statuses))
        self.assertEqual(normalized.statuses["dental_implant"], "PRESENT")

        incomplete = {"deterministic_atomic_statuses": {"dental_implant": "ABSENT"}}
        with self.assertRaisesRegex(ValueError, "missing="):
            normalize_prediction(incomplete)

        duplicate = _adaptation_prediction(statuses)
        duplicate["evaluation_adaptation_report"]["findings"].append(
            dict(duplicate["evaluation_adaptation_report"]["findings"][0])
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            normalize_prediction(duplicate)

    def test_prediction_directory_uses_image_identity_and_requires_exact_set(self) -> None:
        self._image("sample.png")
        benchmark = load_yolo_benchmark(self.images_dir, self.labels_dir)
        predictions_dir = self.root / "predictions"
        predictions_dir.mkdir()
        payload = {
            "image_path": "/different/runtime/path/sample.png",
            "deterministic_atomic_statuses": _statuses(),
        }
        (predictions_dir / "saved_output.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        (predictions_dir / "evaluation_results.json").write_text(
            json.dumps({"schema_version": 1}), encoding="utf-8"
        )

        result = evaluate_experiment(benchmark, predictions_dir)
        self.assertEqual(result["dataset"]["image_ids"], ["sample"])

        (predictions_dir / "saved_output.json").unlink()
        with self.assertRaisesRegex(ValueError, r"missing=\['sample'\]"):
            evaluate_experiment(benchmark, predictions_dir)

    def test_finding_metrics_and_uncertainty_are_hand_calculated(self) -> None:
        self._image("a.png")
        self._image("b.png")
        (self.labels_dir / "a.txt").write_text("0 0.2 0.25 0.1 0.1\n", encoding="utf-8")
        (self.labels_dir / "b.txt").write_text("1 0.8 0.75 0.1 0.1\n", encoding="utf-8")
        benchmark = load_yolo_benchmark(self.images_dir, self.labels_dir)

        a_statuses = _statuses()
        a_statuses["dental_implant"] = "PRESENT"
        a_statuses["prosthetic_restoration"] = "UNCERTAIN"
        b_statuses = _statuses()
        b_statuses["dental_implant"] = "UNCERTAIN"
        result = evaluate_experiment(
            benchmark,
            {
                "a": {"deterministic_atomic_statuses": a_statuses},
                "b": {"deterministic_atomic_statuses": b_statuses},
            },
        )

        overall = result["overall_metrics"]
        self.assertEqual((overall["total_tp"], overall["total_tn"]), (1, 24))
        self.assertEqual((overall["total_fp"], overall["total_fn"]), (2, 1))
        self.assertAlmostEqual(overall["micro_precision"], 1 / 3)
        self.assertAlmostEqual(overall["micro_recall"], 1 / 2)
        self.assertAlmostEqual(overall["micro_f1"], 0.4)
        self.assertAlmostEqual(overall["accuracy"], 25 / 28)
        self.assertEqual(overall["uncertain_prediction_count"], 2)
        self.assertAlmostEqual(overall["uncertainty_rate"], 2 / 28)
        self.assertEqual(overall["complete_image_count"], 1)
        self.assertAlmostEqual(overall["complete_image_rate"], 0.5)
        self.assertAlmostEqual(overall["mean_finding_coverage"], 0.5)

        implant = result["per_class_metrics"]["dental_implant"]
        self.assertEqual((implant["tp"], implant["fp"], implant["fn"]), (1, 1, 0))
        prosthetic = result["per_class_metrics"]["prosthetic_restoration"]
        self.assertEqual((prosthetic["tp"], prosthetic["fp"], prosthetic["fn"]), (0, 1, 1))
        no_positive_class = result["per_class_metrics"]["surgical_device"]
        self.assertEqual(no_positive_class["precision"], 0.0)
        self.assertEqual(no_positive_class["recall"], 0.0)
        self.assertEqual(no_positive_class["specificity"], 1.0)

    def test_geometry_levels_and_patient_orientation(self) -> None:
        self._image("sample.png")
        (self.labels_dir / "sample.txt").write_text(
            "0 0.2 0.25 0.1 0.1\n1 0.5 0.5 0.1 0.1\n", encoding="utf-8"
        )
        benchmark = load_yolo_benchmark(self.images_dir, self.labels_dir)
        left_box, center_box = benchmark.images["sample"].boxes

        self.assertEqual(
            geometry_region(left_box, GeometryThresholds(), 2),
            ("upper", "right", "posterior"),
        )
        self.assertEqual(
            geometry_region(
                left_box, GeometryThresholds(image_left_is_patient_right=False), 1
            ),
            ("upper", "left"),
        )
        self.assertEqual(
            geometry_region(center_box, GeometryThresholds(), 2),
            ("lower", "center", "anterior"),
        )

    def test_wrong_location_does_not_change_finding_true_positive(self) -> None:
        self._image("sample.png")
        (self.labels_dir / "sample.txt").write_text("0 0.2 0.25 0.1 0.1\n", encoding="utf-8")
        benchmark = load_yolo_benchmark(self.images_dir, self.labels_dir)
        statuses = _statuses()
        statuses["dental_implant"] = "PRESENT"
        prediction = _adaptation_prediction(
            statuses,
            {
                "dental_implant": {
                    "arch": "maxilla",
                    "side": "patient left",
                    "area": "posterior",
                }
            },
        )
        result = evaluate_experiment(
            benchmark,
            {"sample": prediction},
            config=EvaluationConfig(
                evaluate_location=True,
                location_level=2,
                location_adapters=("geometry",),
            ),
        )

        self.assertEqual(result["per_class_metrics"]["dental_implant"]["tp"], 1)
        location = result["location_metrics"]["geometry"]
        self.assertEqual((location["tp"], location["fp"], location["fn"]), (0, 1, 1))
        self.assertEqual(location["eligible_true_positive_findings"], 1)

    def test_vision_cache_is_strict_reusable_and_separate_from_geometry(self) -> None:
        self._image("sample.png")
        (self.labels_dir / "sample.txt").write_text("0 0.2 0.25 0.1 0.1\n", encoding="utf-8")
        benchmark = load_yolo_benchmark(self.images_dir, self.labels_dir)
        runner = FakeExpertRunner()
        resolver = VisionLocationResolver.from_expert_model(runner)
        cache_path = self.root / "vision_cache.json"
        annotated_dir = self.root / "annotated"

        cache = prepare_vision_location_cache(
            benchmark, resolver, cache_path, annotated_dir
        )
        prepare_vision_location_cache(benchmark, resolver, cache_path, annotated_dir)
        self.assertEqual(runner.calls, 1)
        self.assertTrue((annotated_dir / "sample_boxes.png").is_file())
        self.assertIn("box_0000", cache["images"]["sample"]["locations"])

        statuses = _statuses()
        statuses["dental_implant"] = "PRESENT"
        prediction = _adaptation_prediction(
            statuses,
            {
                "dental_implant": {
                    "arch": "upper",
                    "side": "right",
                    "area": "posterior",
                }
            },
        )
        result = evaluate_experiment(
            benchmark,
            {"sample": prediction},
            config=EvaluationConfig(
                evaluate_location=True,
                location_level=2,
                location_adapters=("geometry", "vision"),
                vision_backend="expert_model",
            ),
            vision_cache=cache,
        )
        self.assertEqual(set(result["location_metrics"]), {"geometry", "vision"})
        self.assertEqual(result["location_metrics"]["geometry"]["tp"], 1)
        self.assertEqual(result["location_metrics"]["vision"]["tp"], 1)

        mismatched_cache = json.loads(json.dumps(cache))
        mismatched_cache["resolver"]["backend"] = "llm"
        with self.assertRaisesRegex(ValueError, "backend does not match"):
            evaluate_experiment(
                benchmark,
                {"sample": prediction},
                config=EvaluationConfig(
                    evaluate_location=True,
                    location_level=2,
                    location_adapters=("vision",),
                    vision_backend="expert_model",
                ),
                vision_cache=mismatched_cache,
            )

    def test_comparison_requires_exact_same_dataset(self) -> None:
        base = {
            "dataset": {"fingerprint": "same", "image_ids": ["a"]},
            "experiment_metadata": {"name": "v1"},
            "overall_metrics": {
                "micro_recall": 1.0,
                "micro_precision": 1.0,
                "macro_f1": 1.0,
                "complete_image_rate": 1.0,
                "average_fp_per_image": 0.0,
                "average_fn_per_image": 0.0,
                "uncertainty_rate": 0.0,
            },
        }
        second = json.loads(json.dumps(base))
        second["experiment_metadata"]["name"] = "v2"
        rows = compare_experiments([base, second])
        self.assertEqual([row["experiment"] for row in rows], ["v1", "v2"])

        second["dataset"]["image_ids"] = ["b"]
        with self.assertRaisesRegex(ValueError, "exact same benchmark"):
            compare_experiments([base, second])


if __name__ == "__main__":
    unittest.main()
