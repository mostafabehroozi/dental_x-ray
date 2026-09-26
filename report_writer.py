"""A report planner over closed PAN facts, with deterministic clinical rendering.

The text model may organize findings. Clinical statements, regions and impressions
are rendered exclusively from validated IDs and normalized analyzer values.
"""
from __future__ import annotations
import hashlib
import json
import re
import time
from pathlib import Path
import dental_pipeline as dp
import llm_api
import run_monitor as mon

SCHEMA = "dentvlm-findings/2"
IDENTIFIERS = tuple(dp.TASKS)
TREATMENT = ("prosthetic_crown", "root_canal_therapy", "fillings", "prosthetic_bridge", "implant")
PATHOLOGY = tuple(k for k in IDENTIFIERS if k not in TREATMENT)
CATEGORIES = {"pathology": ("Findings", PATHOLOGY),
              "treatment": ("Existing treatments", TREATMENT)}
LABELS = dp.LABELS
LEGEND = {"present": "The analysis reports this finding.", "absent": "The analysis did not detect this finding.",
          "unresolved": "The response cannot establish presence or absence.",
          "not_assessed": "This supported task was not asked in the historical run."}
SYSTEM_PROMPT = "Organize the supplied PAN finding records. Copy IDs, statuses and regions exactly. Return JSON only."
OUTPUT_SCHEMA = '{"language":"English","sections":[{"category":"pathology","findings":[{"finding":"caries","status":"present","regions":[]}]}],"impression":["caries"]}'
USER_PROMPT = "Organize every supplied finding exactly once in its category. Include both categories. Copy every status and the complete permitted region list. Impression must contain every positive finding ID exactly once, and no other IDs. Do not write clinical prose or add keys."
REPAIR_PROMPT = "Correct these errors: {problems}. Return the complete JSON with all findings."
AGREEMENT_DATA = AGREEMENT_RULE = ""
AGREEMENT_BANDS = {}


def structured_findings(result, analyzer=None, include_rationale=False, vote_agreement=False, parser=None, counting=False):
    if include_rationale or vote_agreement:
        raise ValueError("PAN reports use normalized findings; raw rationales are audit evidence only")
    entries, evidence = [], []
    for key in IDENTIFIERS:
        task = result.get("tasks", {}).get(key)
        calls = [c for c in result.get("calls", []) if c.get("task") == key]
        evidence.append({"task": key, "calls": calls})
        answer = task.get("presence") if task else None
        status = "not_assessed" if task is None else "present" if answer == "yes" else "absent" if answer == "no" else "unresolved"
        source_regions = list(task.get("report_regions", [])) if task else []
        if task and "report_regions" not in task and answer == "yes":
            # Historical evidence is never reconstructed using today's question registry.
            for call in calls:
                if (call.get("stage") == "presence" and not call.get("truncated")
                        and dp.extract_answer(call.get("text", "")) == "yes"):
                    for match in dp.location_evidence(call["text"], key):
                        if match["reportable"]:
                            source_regions.extend(match["regions"])
        source_regions = [c for c in dp.CELLS if c in source_regions] if answer == "yes" else []
        permitted = list(dict.fromkeys(dp.report_location(c) for c in source_regions))
        entry = {"finding": key, "label": LABELS[key], "category": "treatment" if key in TREATMENT else "pathology",
                 "status": status, "regions": permitted, "location_status":
                 (task.get("location_status", "located" if permitted else "not_stated") if answer == "yes" else "not_applicable"),
                 "patient_laterality": "unresolved"}
        if counting and answer == "yes":
            entry["named_source_region_count"] = len(source_regions) if source_regions else None
        entries.append(entry)
    return {"schema": SCHEMA, "image": {"id": result.get("image_id", Path(result["image"]).stem),
            "file": Path(result["image"]).name, "sha256": result.get("image_sha256")},
            "analysis": {"analyzer": analyzer or "DentVLM", "questions_asked": result.get("call_count"),
                         "profile": result.get("profile", "legacy"), "patient_laterality": "unresolved",
                         "limitations": ["Patient laterality is unverified; source labels remain in audit evidence."]},
            "categories": [{"key": k, "label": v[0], "findings": list(v[1])} for k,v in CATEGORIES.items()],
            "findings": entries, "evidence": evidence,
            "summary": {state: [e["finding"] for e in entries if e["status"] == state]
                        for state in ("present", "absent", "unresolved", "not_assessed")}}


def user_prompt(structured, language):
    # No free analyzer text enters the reporter's factual input.
    payload = {"language": language, "categories": structured["categories"], "findings": structured["findings"]}
    return USER_PROMPT + "\nShape: " + OUTPUT_SCHEMA + "\n" + json.dumps(payload, ensure_ascii=False)


def verify_report(report, structured, parser=None):
    return verify_report_detailed(report, structured, parser)[0]


def verify_report_detailed(report, structured, parser=None, *, context=""):
    problems = []
    if not isinstance(report, dict) or set(report) != {"language", "sections", "impression"}:
        return ["Expected only language, sections and impression"], []
    if not isinstance(report["language"], str) or not report["language"].strip():
        problems.append("language must be a nonempty string")
    expected = {e["finding"]: e for e in structured["findings"]}
    seen, categories = [], []
    if not isinstance(report["sections"], list):
        return problems + ["sections must be a list"], []
    for section in report["sections"]:
        if not isinstance(section, dict) or set(section) != {"category", "findings"}:
            problems.append("invalid section shape"); continue
        category = section["category"]
        if not isinstance(category, str):
            problems.append("category must be a string"); continue
        categories.append(category)
        if category not in CATEGORIES or not isinstance(section["findings"], list):
            problems.append("unknown category or invalid findings"); continue
        for entry in section["findings"]:
            if not isinstance(entry, dict) or set(entry) != {"finding", "status", "regions"}:
                problems.append("finding must contain only finding, status and regions"); continue
            key = entry["finding"]
            if not isinstance(key, str) or key not in expected:
                problems.append("unknown finding"); continue
            seen.append(key); fact = expected[key]
            if fact["category"] != category or entry["status"] != fact["status"]:
                problems.append(f"{key}: category/status changed")
            if entry["regions"] != fact["regions"]:
                problems.append(f"{key}: permitted regions changed")
    if sorted(categories) != sorted(CATEGORIES):
        problems.append("each category must appear exactly once")
    if sorted(seen) != sorted(expected):
        problems.append("each supplied finding must appear exactly once")
    impression = report["impression"]
    if (not isinstance(impression, list) or any(not isinstance(i,str) for i in impression)
            or sorted(impression) != sorted(structured["summary"]["present"])):
        problems.append("impression must reference every positive finding exactly once")
    return problems, []


def render_facts(structured, report=None):
    expected = {e["finding"]: e for e in structured["findings"]}
    lines = ["# Panoramic radiograph: automated findings", "", "Image: " + structured["image"]["file"]]
    for category, (label, keys) in CATEGORIES.items():
        lines += ["", "## " + label, ""]
        for key in keys:
            fact = expected[key]
            status = fact["status"]
            phrase = {"present": "The analysis flags", "absent": "The analysis did not detect",
                      "unresolved": "Unresolved", "not_assessed": "Not assessed in this historical run"}[status]
            line = f"- {phrase}: {fact['label']}."
            if status == "present":
                line += " Location: " + (", ".join(fact["regions"]) if fact["regions"] else "unavailable") + "."
            lines.append(line)
    lines += ["", "## Impression", ""]
    ids = report["impression"] if report else structured["summary"]["present"]
    lines += ["- The analysis flags " + expected[key]["label"] + "." for key in ids]
    if not ids:
        lines.append("- No positive findings were reported within the assessed task set.")
    if structured["summary"]["unresolved"]:
        lines.append("- Unresolved: " + ", ".join(LABELS[k] for k in structured["summary"]["unresolved"]) + ".")
    lines += ["", "Patient laterality is unresolved. Unmentioned regions are not confirmed negative."]
    return "\n".join(lines)


def render_markdown(report, structured, writer_model=None):
    problems = verify_report(report, structured)
    if problems:
        raise ValueError("Invalid report: " + "; ".join(problems))
    return render_facts(structured, report)


def fallback_markdown(result, problems, counting=False):
    return render_facts(structured_findings(result, counting=counting))


def result_hash(result):
    return hashlib.sha256(json.dumps(result, sort_keys=True, default=str).encode()).hexdigest()



def extract_json(text: str) -> dict | None:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.S)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None

def read_report_json(text: str, parser=None, *, context: str = ""):
    """The report object in a reply, and the parser record behind it.

    The strict loader is the reader; a parser service may repair a reply it rejects, and may never
    write a value the reply does not contain (its prompt forbids it, and a repair that returns
    nothing usable leaves the report missing, which the verification then reports).
    """
    def code():
        payload = extract_json(text)
        return payload, None if payload is not None else "invalid_report_json"

    if parser is None or not parser.enabled("report_json"):
        return code()[0], None
    outcome = parser.report_json(text, code=code, keys=REPORT_KEYS, context=context)
    return outcome.value, outcome.record

def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "-", text).strip("-").lower() or "model"

class ReportWriter:
    """Structured findings of one image -> verified report JSON + Markdown, from a hosted text LLM.

    from_api() builds one from an llm_api spec. token_param "max_completion_tokens" and temperature
    None for OpenAI reasoning models; other request fields (reasoning_effort, response_format, ...)
    go through request_options. include_rationale adds DentVLM's own reply text per task to the
    input (off by default: the report then rests on the parsed answers alone). vote_agreement adds
    the vote counts behind those answers (off by default; see AGREEMENT_LEGEND). counting (on by
    default) gives the report each finding's multiplicity, the number of regions it was reported in.
    One repair turn is allowed: the reply's problems are sent back and the corrected JSON re-verified.
    """

    kind = "report"
    OPTIONS = ("token_param", "temperature", "max_output_tokens", "request_options", "language", "repairs",
               "include_rationale", "vote_agreement", "counting", "api_call_retries")

    def __init__(self, base_url: str | None, api_key: str, model: str, token_param: str = "max_tokens",
                 max_output_tokens: int = 4096, temperature: float | None = 0.0, language: str = "English",
                 repairs: int = 1, include_rationale: bool = False, vote_agreement: bool = False,
                 counting: bool = False, timeout: float = 600.0, request_options: dict | None = None,
                 api_call_retries: int = 2, call_log: str | None = None, client=None, parser=None) -> None:
        if include_rationale or vote_agreement:
            raise ValueError("Raw rationale and wording-vote report modes are retired for PAN reports")
        if token_param not in llm_api.TOKEN_PARAMS:
            raise ValueError(f"token_param must be one of {llm_api.TOKEN_PARAMS}")
        if not isinstance(language, str) or not language.strip():
            raise ValueError("language must be a non-empty string, e.g. 'English' or 'Persian'")
        llm_api.validate_api_retries(api_call_retries)
        self.client = client if client is not None else llm_api.connect(base_url, api_key, timeout)
        self.base_url, self.model = base_url, model
        self.token_param, self.max_output_tokens, self.temperature = token_param, max_output_tokens, temperature
        self.language, self.repairs, self.include_rationale = language.strip(), max(0, int(repairs)), bool(include_rationale)
        self.vote_agreement, self.counting = bool(vote_agreement), bool(counting)
        self._noted_single_wording = False
        self.request_options = dict(request_options or {})
        self.api_call_retries = api_call_retries
        # The reader for this writer's own replies (llm_parser.ParserService); None reads with code.
        self.parser = parser
        self.call_log = mon.CallLog("report", call_log)

    @classmethod
    def from_api(cls, spec: dict, language: str | None = None, counting: bool | None = None,
                 timeout: float = 600.0, client=None, parser=None) -> "ReportWriter":
        """Writer for a hosted model. spec = {"provider", "model", ...} as documented in llm_api, plus any
        of the constructor options named in OPTIONS; a language or counting argument wins over the spec's."""
        base_url, api_key = llm_api.resolve(spec)
        options = {k: spec[k] for k in cls.OPTIONS if k in spec}
        if language is not None:
            options["language"] = language
        if counting is not None:
            options["counting"] = counting
        return cls(base_url, api_key, spec["model"], timeout=timeout, client=client, parser=parser, **options)

    @property
    def calls(self) -> int:
        return self.call_log.calls

    @property
    def name(self) -> str:
        return "report-" + slug(self.model)

    @property
    def run_name(self) -> str:
        """Directory name of a run: the model and the language."""
        return f"{self.name}-{slug(self.language)}"

    def settings(self) -> dict:
        """Everything that shapes a report (prompts included), hashed into the run manifest."""
        return {"kind": self.kind, "model": self.model, "base_url": self.base_url, "token_param": self.token_param,
                "max_output_tokens": self.max_output_tokens, "temperature": self.temperature, "language": self.language,
                "repairs": self.repairs, "include_rationale": self.include_rationale,
                "vote_agreement": self.vote_agreement, "counting": self.counting,
                "api_call_retries": self.api_call_retries,
                "request_options": self.request_options, "schema": SCHEMA,
                "system_prompt": SYSTEM_PROMPT, "user_prompt": USER_PROMPT, "output_schema": OUTPUT_SCHEMA,
                "repair_prompt": REPAIR_PROMPT,
                **({"agreement_prompt": [AGREEMENT_DATA, AGREEMENT_RULE], "agreement_bands": AGREEMENT_BANDS}
                   if self.vote_agreement else {}),
                **({"parser": self.parser.settings()}
                   if self.parser is not None and self.parser.policy.uses_llm() else {})}

    def public(self) -> dict:
        """The settings without the prompt texts, for printouts."""
        return {k: v for k, v in self.settings().items()
                if k not in ("system_prompt", "user_prompt", "output_schema", "repair_prompt",
                             "agreement_prompt", "agreement_bands")}

    def _ask(self, messages: list[dict]) -> dict:
        request = {"model": self.model, "messages": messages,
                   **llm_api.generation_fields(self.token_param, self.max_output_tokens, self.temperature)}
        request.update(self.request_options)
        started = time.perf_counter()
        normalized = llm_api.call_with_retries(
            lambda: llm_api.chat_reply(self.client.chat.completions.create(**request)),
            self.api_call_retries, f"report model={self.model}")
        return self.call_log.live({**normalized, "latency_seconds": round(time.perf_counter() - started, 3)})

    def write(self, result: dict, analyzer: str | None = None) -> dict:
        """One image result -> {"structured", "report", "verified", "problems", "markdown", "attempts", ...}."""
        usage_at_start = self.parser.usage_snapshot() if self.parser is not None else None
        structured = structured_findings(result, analyzer, self.include_rationale, self.vote_agreement,
                                         self.parser, self.counting)
        if self.vote_agreement and not structured["analysis"]["vote_agreement"]["measured"] and not self._noted_single_wording:
            self._noted_single_wording = True
            llm_api.monitor("REPORT NOTE", "vote_agreement is on but this run asked one wording per task",
                            action="every finding will say agreement was not measured")
        prompt = user_prompt(structured, self.language)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        attempts, report, problems, parsing = [], None, ["no reply"], []
        context = f"image={structured['image']['id']}"
        for _attempt in range(1 + self.repairs):
            try:
                reply = self._ask(messages)
            except Exception as exc:
                reply = {"text": "", "truncated": False, "error": f"{type(exc).__name__}: {exc}"}
            report, extraction = read_report_json(reply["text"], self.parser, context=context)
            problems, checks = verify_report_detailed(report, structured, self.parser, context=context)
            if isinstance(report, dict) and report.get("language") != self.language:
                problems.append("language does not match requested report language")
            if reply["truncated"]:
                problems.append("the reply was cut off by max_output_tokens")
            records = ([extraction] if extraction else []) + checks
            parsing.append(records)
            attempts.append({**reply, "problems": problems, **({"parsing": records} if records else {})})
            if not problems:
                break
            llm_api.monitor("REPORT VERIFY WARNING", f"image={structured['image']['id']}",
                            attempt=f"{_attempt + 1}/{self.repairs + 1}", problems=len(problems))
            llm_api.failure_details(json.dumps(messages, ensure_ascii=False, indent=2), reply["text"], problems)
            if _attempt < self.repairs:
                llm_api.monitor("REPORT RETRY", f"image={structured['image']['id']}", action="verification repair")
            messages += [{"role": "assistant", "content": reply["text"]},
                         {"role": "user", "content": REPAIR_PROMPT.replace("{problems}", "\n".join(f"- {p}" for p in problems))}]
        verified = not problems
        if not verified:
            llm_api.monitor("REPORT FALLBACK", f"image={structured['image']['id']}", policy="deterministic summary")
        return {
            "image_id": structured["image"]["id"], "image": result["image"], "schema": SCHEMA,
            "language": self.language, "clinical_facts_language": "English", "source_result_hash": result_hash(result),
            "writer": self.public(), "analyzer": structured["analysis"]["analyzer"],
            "structured": structured, "prompt": prompt,
            "report": report if verified else None, "verified": verified, "problems": problems,
            "markdown": (render_markdown(report, structured, self.model) if verified
                         else fallback_markdown(result, problems, self.counting)),
            "attempts": attempts,
            **({"parser": self.parser.public(), "parser_fingerprint": self.parser.fingerprint(),
                "parser_usage": self.parser.usage_since(usage_at_start),
                "parsing": [row for rows in parsing for row in rows]}
               if self.parser is not None else {}),
        }

def report_dataset(writer: ReportWriter, results: dict[str, dict], out_dir: str | Path, analyzer: str | None = None,
                   resume: bool = True, limit: int | None = None, ledger: "mon.Ledger | None" = None,
                   stop_after: int = 3) -> dict[str, dict]:
    """Write a report for every result, one JSON and one .md per image under out_dir/reports; resumable.

    An image whose report call fails is recorded with its complete traceback and the
    loop continues, so one refused or unreachable request does not cost the rest of
    the reports; `stop_after` consecutive failures stop the loop.
    """
    out = Path(out_dir)
    reports_dir = out / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    config = {"writer": writer.settings(), "analyzer": analyzer, "categories": CATEGORIES, "legend": LEGEND}
    config["hash"] = hashlib.sha256(json.dumps(config, sort_keys=True, default=list).encode()).hexdigest()[:16]
    manifest_path = out / "manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("hash") != config["hash"]:
            raise ValueError(f"{out} holds reports from a different writer configuration or language; use a new directory.")
    else:
        manifest_path.write_text(json.dumps(config, indent=2, default=list), encoding="utf-8")

    todo = sorted(results.items())[:limit] if limit else sorted(results.items())
    failures = mon.Ledger(f"reports {out.name}")
    progress = mon.Progress(len(todo), label=f"reports {writer.run_name}", unit="report")
    for image_id, result in todo:
        target = reports_dir / f"{image_id}.json"
        if resume and target.is_file():
            saved = _load_report_file(target, image_id)
            if saved.get("source_result_hash") != result_hash(result):
                raise ValueError(f"{image_id}: analyzer result changed; use a new report directory")
            progress.skip(image_id)
            continue
        with mon.guard(f"{out.name}/{image_id}", failures) as step:
            payload = writer.write(result, analyzer)
            payload["image_id"] = image_id
            tmp = target.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
            tmp.replace(target)
            (reports_dir / f"{image_id}.md").write_text(payload["markdown"], encoding="utf-8")
        if not step.ok:
            if progress.failure(image_id) >= stop_after:
                progress.stop(f"{stop_after} reports in a row failed; fix the cause and rerun to resume")
                break
            continue
        detail = f"verified={payload['verified']} attempts={len(payload['attempts'])}"
        if not payload["verified"]:
            detail += f" | fell back, problems: {mon.clip('; '.join(payload['problems']), 120)}"
        progress.item(image_id, detail, repairs=len(payload["attempts"]) - 1 or None,
                      fallback=0 if payload["verified"] else 1)
    log = getattr(writer, "call_log", None)
    parser_log = getattr(getattr(writer, "parser", None), "model", None)
    detail = log.line(counts=False) if isinstance(log, mon.CallLog) else ""
    if parser_log is not None and parser_log.call_log.requests:
        detail = (detail + " | " if detail else "") + f"parser {parser_log.call_log.line()}"
    progress.done(detail=detail)
    if failures:
        failures.report(path=out / "failures.json")
        if ledger is not None:
            ledger.entries.extend(failures.entries)
    return load_reports(out)

def _load_report_file(path: Path, expected_id: str | None = None) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        llm_api.monitor("ARTIFACT ERROR", str(path), reason=str(exc))
        raise ValueError(f"invalid report artifact {path}: {exc}") from exc
    image_id = payload.get("image_id") if isinstance(payload, dict) else None
    if not isinstance(image_id, str) or image_id != path.stem or (expected_id and image_id != expected_id):
        raise llm_api.artifact_error(path, "report image_id does not match filename/expected id")
    if not isinstance(payload.get("verified"), bool) or not isinstance(payload.get("attempts"), list):
        raise llm_api.artifact_error(path, "report missing verified/attempts schema")
    return payload

def load_reports(out_dir: str | Path) -> dict[str, dict]:
    reports = {}
    for path in sorted(Path(out_dir, "reports").glob("*.json")):
        payload = _load_report_file(path)
        if payload["image_id"] in reports:
            raise llm_api.artifact_error(path, f"duplicate report image_id {payload['image_id']!r}")
        reports[payload["image_id"]] = payload
    return reports

def summarize_reports(reports: dict[str, dict]) -> dict:
    """How many reports verified at once, after a repair, or fell back to the deterministic summary."""
    verified = [r for r in reports.values() if r["verified"]]
    tokens = [a["completion_tokens"] for r in reports.values() for a in r["attempts"] if a.get("completion_tokens")]
    parsed: dict[str, int] = {}
    for report in reports.values():
        for stage, row in (report.get("parser_usage") or {}).items():
            for key, value in row.items():
                parsed[key] = parsed.get(key, 0) + value
    return {"images": len(reports), "verified": len(verified),
            "repaired": sum(len(r["attempts"]) > 1 for r in verified),
            "fallback": len(reports) - len(verified),
            "mean_completion_tokens": round(sum(tokens) / len(tokens)) if tokens else None,
            **({"parser": parsed} if parsed else {})}
