from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import LLMOrchestrator  # noqa: E402
from prompts import CONDITIONS  # noqa: E402


class _Parsed:
    def __init__(self, value: dict):
        self.value = value

    def model_dump(self) -> dict:
        return self.value


class _QueuedCompletions:
    def __init__(self, outputs: list[dict]):
        self.outputs = list(outputs)
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        parsed = _Parsed(self.outputs.pop(0))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))]
        )


def _evaluation_report(overrides: dict[str, str] | None = None) -> dict:
    overrides = overrides or {}
    return {
        "findings": [
            {
                "condition": condition,
                "status": overrides.get(condition, "UNCERTAIN"),
                "evidence": "Not established in the report.",
                "location": {
                    "distribution": None,
                    "arch": None,
                    "side": None,
                    "area": None,
                    "relation": None,
                    "fdi_tooth": None,
                },
                "conflict": None,
            }
            for condition in reversed(CONDITIONS)
        ],
        "adaptation_notes": "",
    }


class OrchestratorPhaseTests(unittest.TestCase):
    def _orchestrator(self, outputs: list[dict]) -> tuple[LLMOrchestrator, _QueuedCompletions]:
        completions = _QueuedCompletions(outputs)
        orchestrator = object.__new__(LLMOrchestrator)
        orchestrator.client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )
        orchestrator.model = "test-model"
        orchestrator.provider = "test-provider"
        orchestrator.dentist_report_type = object()
        orchestrator.evaluation_adaptation_type = object()
        return orchestrator, completions

    def test_run_separates_dentist_report_and_evaluation_adaptation(self):
        observations = [
            {"question_id": "broad", "raw_answer": "Finding A."},
            {"question_id": "atomic", "raw_answer": "<answer>PRESENT</answer>"},
        ]
        statuses = {"dental_implant": "PRESENT"}
        dentist_report = {
            "report": "Finding A. Experimental output, not a clinical diagnosis.",
            "source_question_ids": ["atomic", "broad"],
        }
        orchestrator, completions = self._orchestrator(
            [dentist_report, _evaluation_report(statuses)]
        )

        result = orchestrator.run("DISEASE_HIERARCHY", observations, statuses)

        self.assertEqual(len(completions.calls), 2)
        phase_one_payload = json.loads(completions.calls[0]["messages"][1]["content"])
        phase_two_payload = json.loads(completions.calls[1]["messages"][1]["content"])
        self.assertIn("expert_model_observations", phase_one_payload)
        self.assertNotIn("closed_ontology", phase_one_payload)
        self.assertEqual(phase_two_payload["dentist_report"]["report"], dentist_report["report"])
        self.assertNotIn("expert_model_observations", phase_two_payload)
        self.assertEqual(
            result["dentist_report"]["source_question_ids"], ["broad", "atomic"]
        )
        self.assertEqual(
            [item["condition"] for item in result["evaluation_adaptation_report"]["findings"]],
            list(CONDITIONS),
        )

    def test_dentist_report_rejects_missing_source_coverage(self):
        orchestrator, _ = self._orchestrator(
            [{"report": "Incomplete.", "source_question_ids": ["one"]}]
        )
        observations = [
            {"question_id": "one", "raw_answer": "A"},
            {"question_id": "two", "raw_answer": "B"},
        ]

        with self.assertRaisesRegex(ValueError, "every expert-model observation"):
            orchestrator.create_dentist_report("BASIC", observations)

    def test_evaluation_adaptation_cannot_change_atomic_status(self):
        orchestrator, _ = self._orchestrator([_evaluation_report()])

        with self.assertRaisesRegex(ValueError, "changed deterministic status"):
            orchestrator.create_evaluation_adaptation_report(
                "DISEASE_HIERARCHY",
                {"report": "An implant is present.", "source_question_ids": ["atomic"]},
                {"dental_implant": "PRESENT"},
            )


if __name__ == "__main__":
    unittest.main()
