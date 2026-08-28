from __future__ import annotations

import json
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Literal

from prompts import (
    CONDITIONS,
    DENTIST_REPORT_SYSTEM_PROMPT,
    EVALUATION_ADAPTATION_SYSTEM_PROMPT,
    atomic_records,
    broad_records,
    family_records,
    location_records,
)


VALID_MODES = {"BASIC", "DISEASE_HIERARCHY", "DISEASE_AND_LOCATION"}


def print_pipeline_event(title: str, content: str) -> None:
    rule = "=" * 78
    print(f"\n{rule}\n{title}\n{'-' * 78}\n{content}\n{rule}")


def extract_answer_text(text: str) -> str | None:
    matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.I | re.S)
    return matches[-1].strip() if matches else None


def parse_atomic_status(text: str) -> str:
    """Conservatively normalize a dental expert-model atomic answer.

    Prefer the final <answer> block. If multiple incompatible labels appear inside it,
    return UNCERTAIN rather than choosing whichever token happened to occur last.
    """
    answer = extract_answer_text(text)
    searchable = answer if answer is not None else text[-500:]
    matches = re.findall(r"\b(PRESENT|ABSENT|UNCERTAIN)\b", searchable.upper())
    unique = set(matches)
    if len(unique) == 1:
        return next(iter(unique))
    return "UNCERTAIN"


class LLMOrchestrator:
    """Optional text-only model for report synthesis and evaluation adaptation."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        provider: str = "openai",
        timeout: float | None = 600,
        max_retries: int | None = 2,
    ):
        from openai import OpenAI
        from pydantic import BaseModel, ConfigDict

        class Location(BaseModel):
            model_config = ConfigDict(extra="forbid")
            distribution: str | None
            arch: str | None
            side: str | None
            area: str | None
            relation: str | None
            fdi_tooth: str | None

        class Finding(BaseModel):
            model_config = ConfigDict(extra="forbid")
            condition: Literal[
                "dental_implant",
                "prosthetic_restoration",
                "dental_filling",
                "endodontic_treatment",
                "carious_lesion",
                "periodontal_bone_loss",
                "impacted_tooth",
                "periapical_lesion",
                "root_fragment",
                "furcation_lesion",
                "apical_surgery",
                "root_resorption",
                "orthodontic_device",
                "surgical_device",
            ]
            status: Literal["PRESENT", "ABSENT", "UNCERTAIN"]
            evidence: str
            location: Location
            conflict: str | None

        class DentistReport(BaseModel):
            model_config = ConfigDict(extra="forbid")
            report: str
            source_question_ids: list[str]

        class EvaluationAdaptationReport(BaseModel):
            model_config = ConfigDict(extra="forbid")
            findings: list[Finding]
            adaptation_notes: str

        client_kwargs = {}
        if base_url:
            client_kwargs["base_url"] = base_url
        if api_key:
            client_kwargs["api_key"] = api_key
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        if max_retries is not None:
            client_kwargs["max_retries"] = max_retries

        self.client = OpenAI(**client_kwargs)
        self.model = model
        self.base_url = base_url
        self.provider = provider
        self.dentist_report_type = DentistReport
        self.evaluation_adaptation_type = EvaluationAdaptationReport

    def _parse(self, system_prompt: str, payload: dict, response_format):
        completion = self.client.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            response_format=response_format,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise RuntimeError("The orchestrator returned no parsed result.")
        return parsed.model_dump()

    def create_dentist_report(self, mode: str, observations: list[dict]) -> dict:
        payload = {
            "analysis_mode": mode,
            "expert_model_observations": observations,
        }
        print_pipeline_event(
            f"DENTIST REPORT REQUEST | provider={self.provider} | model={self.model}",
            "The orchestrator receives expert-model text only (no radiograph):\n"
            + json.dumps(payload, indent=2, ensure_ascii=False),
        )
        result = self._parse(
            DENTIST_REPORT_SYSTEM_PROMPT,
            payload,
            self.dentist_report_type,
        )
        returned_ids = result["source_question_ids"]
        expected_ids = [item["question_id"] for item in observations]
        if (
            len(returned_ids) != len(set(returned_ids))
            or set(returned_ids) != set(expected_ids)
        ):
            raise ValueError(
                "Dentist report must acknowledge every expert-model observation exactly once."
            )
        result["source_question_ids"] = expected_ids
        print_pipeline_event(
            f"DENTIST REPORT RESPONSE | provider={self.provider} | model={self.model}",
            result["report"],
        )
        return result

    def create_evaluation_adaptation_report(
        self,
        mode: str,
        dentist_report: dict,
        deterministic_atomic_statuses: dict[str, str],
    ) -> dict:
        payload = {
            "analysis_mode": mode,
            "closed_ontology": list(CONDITIONS),
            "deterministic_atomic_statuses": deterministic_atomic_statuses,
            "dentist_report": dentist_report,
        }
        print_pipeline_event(
            f"EVALUATION ADAPTATION REQUEST | provider={self.provider} | model={self.model}",
            "The orchestrator receives the dentist report and deterministic statuses only "
            "(no radiograph):\n"
            + json.dumps(payload, indent=2, ensure_ascii=False),
        )
        result = self._parse(
            EVALUATION_ADAPTATION_SYSTEM_PROMPT,
            payload,
            self.evaluation_adaptation_type,
        )

        returned = [item["condition"] for item in result["findings"]]
        expected = list(CONDITIONS)
        if len(returned) != len(set(returned)) or set(returned) != set(expected):
            raise ValueError("Orchestrator must return every ontology condition exactly once.")

        # Hard safety invariant: evaluation adaptation may format atomic statuses but
        # may not reinterpret them.
        by_condition = {item["condition"]: item for item in result["findings"]}
        for condition, status in deterministic_atomic_statuses.items():
            if by_condition[condition]["status"] != status:
                raise ValueError(
                    f"Orchestrator changed deterministic status for {condition}: "
                    f"{status} -> {by_condition[condition]['status']}"
                )

        result["findings"].sort(key=lambda x: expected.index(x["condition"]))
        print_pipeline_event(
            f"EVALUATION ADAPTATION RESPONSE | provider={self.provider} | model={self.model}",
            json.dumps(result, indent=2, ensure_ascii=False),
        )
        return result

    def run(
        self,
        mode: str,
        observations: list[dict],
        deterministic_atomic_statuses: dict[str, str],
    ) -> dict:
        """Run the two deliberately separate text-only orchestration phases."""
        dentist_report = self.create_dentist_report(mode, observations)
        evaluation_report = self.create_evaluation_adaptation_report(
            mode,
            dentist_report,
            deterministic_atomic_statuses,
        )
        return {
            "dentist_report": dentist_report,
            "evaluation_adaptation_report": evaluation_report,
        }


OpenAIOrchestrator = LLMOrchestrator


class DentalAnalysisPipeline:
    def __init__(
        self,
        expert_model_runner=None,
        orchestrator=None,
        locate_uncertain: bool = True,
        *,
        dental_runner=None,
    ):
        # dental_runner remains accepted so older notebooks keep working, but the
        # pipeline role is model-agnostic.
        if expert_model_runner is not None and dental_runner is not None:
            raise ValueError("Pass only expert_model_runner, not both runner arguments.")
        self.expert_model_runner = expert_model_runner or dental_runner
        if self.expert_model_runner is None:
            raise ValueError("expert_model_runner is required.")
        self.orchestrator = orchestrator
        self.locate_uncertain = locate_uncertain

    def _ask_records(self, image_path: str, records: list[dict]) -> list[dict]:
        completed = []
        for index, record in enumerate(records, start=1):
            item = deepcopy(record)
            print_pipeline_event(
                "EXPERT MODEL REQUEST "
                f"| {index}/{len(records)} | id={item['question_id']} "
                f"| layer={item['layer']} | model={self.expert_model_runner.model_id}",
                f"Question:\n{item['question']}",
            )

            item.update(
                self.expert_model_runner.ask(
                    image_path,
                    item["question"],
                    max_tokens=item.get("max_tokens"),
                )
            )

            print_pipeline_event(
                "EXPERT MODEL RESPONSE "
                f"| {index}/{len(records)} | id={item['question_id']} "
                f"| latency={item['latency_seconds']}s "
                f"| finish={item.get('finish_reason')}",
                f"Answer:\n{item['raw_answer']}",
            )

            if item.get("truncated"):
                print(f"WARNING: response {item['question_id']} hit the token limit.")

            if item["layer"] == "ATOMIC_FINDING":
                item["parsed_status"] = parse_atomic_status(item["raw_answer"])
            else:
                item["parsed_answer"] = extract_answer_text(item["raw_answer"])

            completed.append(item)
        return completed

    @staticmethod
    def _atomic_status_map(observations: list[dict]) -> dict[str, str]:
        return {
            item["target"]: item["parsed_status"]
            for item in observations
            if item["layer"] == "ATOMIC_FINDING"
        }

    def run(
        self,
        image_path: str,
        mode: str = "BASIC",
        output_dir: str = "outputs",
    ) -> dict:
        mode = mode.upper()
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(VALID_MODES)}")
        if not Path(image_path).is_file():
            raise FileNotFoundError(image_path)

        started = time.perf_counter()
        observations = self._ask_records(image_path, broad_records())

        if mode in {"DISEASE_HIERARCHY", "DISEASE_AND_LOCATION"}:
            observations += self._ask_records(image_path, family_records())
            atomic = self._ask_records(image_path, atomic_records())
            observations += atomic

            if mode == "DISEASE_AND_LOCATION":
                allowed = {"PRESENT", "UNCERTAIN"} if self.locate_uncertain else {"PRESENT"}
                candidates = [
                    item["target"]
                    for item in atomic
                    if item["parsed_status"] in allowed
                ]
                for condition in candidates:
                    print(f"Location follow-up: {condition}")
                    observations += self._ask_records(image_path, location_records(condition))

        atomic_statuses = self._atomic_status_map(observations)
        orchestration = (
            self.orchestrator.run(mode, observations, atomic_statuses)
            if self.orchestrator
            else None
        )

        result = {
            "analysis_mode": mode,
            "image_path": str(Path(image_path).resolve()),
            "expert_model": self.expert_model_runner.model_id,
            "expert_model_call_count": len(observations),
            "total_latency_seconds": round(time.perf_counter() - started, 3),
            "deterministic_atomic_statuses": atomic_statuses,
            "observations": observations,
            "dentist_report": orchestration["dentist_report"] if orchestration else None,
            "evaluation_adaptation_report": (
                orchestration["evaluation_adaptation_report"] if orchestration else None
            ),
        }

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{Path(image_path).stem}_{mode.lower()}.json"
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        result["saved_to"] = str(output_path.resolve())
        return result
