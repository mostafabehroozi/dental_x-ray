"""Three-mode question pipeline plus an optional capable-LLM orchestrator."""

from __future__ import annotations

import json
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Literal

from prompts import (
    CONDITIONS,
    ORCHESTRATOR_SYSTEM_PROMPT,
    atomic_records,
    broad_records,
    family_records,
    location_records,
)


VALID_MODES = {"BASIC", "DISEASE_HIERARCHY", "DISEASE_AND_LOCATION"}


def print_pipeline_event(title: str, content: str) -> None:
    """Print one readable request/response event for notebook execution."""
    rule = "=" * 78
    print(f"\n{rule}\n{title}\n{'-' * 78}\n{content}\n{rule}")


def parse_atomic_status(text: str) -> str:
    """Conservative parser used only to decide whether location questions run."""
    answer_tags = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.I | re.S)
    searchable = answer_tags[-1] if answer_tags else text[-400:]
    matches = re.findall(r"\b(PRESENT|ABSENT|UNCERTAIN)\b", searchable.upper())
    return matches[-1] if matches else "UNCERTAIN"


class LLMOrchestrator:
    """Optional text-only orchestrator. It never receives the radiograph."""

    def __init__(
        self,
        model: str = "gpt-5.6",
        base_url: str | None = None,
        api_key: str | None = None,
        provider: str = "openai",
        timeout: float | None = None,
        max_retries: int | None = None,
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

        class OrchestratedReport(BaseModel):
            model_config = ConfigDict(extra="forbid")
            findings: list[Finding]
            report: str

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
        self.output_type = OrchestratedReport

    def run(self, mode: str, observations: list[dict]) -> dict:
        payload = {
            "analysis_mode": mode,
            "closed_ontology": list(CONDITIONS),
            "observations": observations,
        }
        print_pipeline_event(
            f"ORCHESTRATOR REQUEST | provider={self.provider} | model={self.model}",
            "The orchestrator receives this text-only payload (no X-ray image):\n"
            + json.dumps(payload, indent=2, ensure_ascii=False),
        )
        completion = self.client.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            response_format=self.output_type,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise RuntimeError("The orchestrator returned no parsed result.")
        result = parsed.model_dump()

        returned = [item["condition"] for item in result["findings"]]
        expected = list(CONDITIONS)
        if len(returned) != len(set(returned)) or set(returned) != set(expected):
            raise ValueError("Orchestrator must return every ontology condition exactly once.")
        result["findings"].sort(key=lambda x: expected.index(x["condition"]))
        print_pipeline_event(
            f"ORCHESTRATOR RESPONSE | provider={self.provider} | model={self.model}",
            json.dumps(result, indent=2, ensure_ascii=False),
        )
        return result


OpenAIOrchestrator = LLMOrchestrator


class DentalAnalysisPipeline:
    def __init__(self, dental_runner, orchestrator=None):
        self.dental_runner = dental_runner
        self.orchestrator = orchestrator

    def _ask_records(self, image_path: str, records: list[dict]) -> list[dict]:
        completed = []
        for index, record in enumerate(records, start=1):
            item = deepcopy(record)
            print_pipeline_event(
                "DENTALGPT REQUEST "
                f"| {index}/{len(records)} | id={item['question_id']} "
                f"| layer={item['layer']} | model={self.dental_runner.model_id}",
                f"Question:\n{item['question']}",
            )
            item.update(self.dental_runner.ask(image_path, item["question"]))
            print_pipeline_event(
                "DENTALGPT RESPONSE "
                f"| {index}/{len(records)} | id={item['question_id']} "
                f"| latency={item['latency_seconds']}s",
                f"Answer:\n{item['raw_answer']}",
            )
            if item["layer"] == "ATOMIC_FINDING":
                item["parsed_status"] = parse_atomic_status(item["raw_answer"])
            completed.append(item)
        return completed

    def run(self, image_path: str, mode: str = "BASIC", output_dir: str = "outputs") -> dict:
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
                candidates = [
                    item["target"] for item in atomic
                    if item["parsed_status"] in {"PRESENT", "UNCERTAIN"}
                ]
                for condition in candidates:
                    print(f"Location follow-up: {condition}")
                    observations += self._ask_records(image_path, location_records(condition))

        orchestrated = self.orchestrator.run(mode, observations) if self.orchestrator else None
        result = {
            "analysis_mode": mode,
            "image_path": str(Path(image_path).resolve()),
            "dentalgpt_call_count": len(observations),
            "total_latency_seconds": round(time.perf_counter() - started, 3),
            "observations": observations,
            "orchestrated_output": orchestrated,
        }

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{Path(image_path).stem}_{mode.lower()}.json"
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        result["saved_to"] = str(output_path.resolve())
        return result
