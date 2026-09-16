"""Dentist report: one text-LLM call turns the per-image findings into a classified report.

DentVLM answers one yes/no question per task on the whole image and names where it sees the
finding in its rationale; the pipeline turns that into 14 benchmark findings plus the model's
extra tasks, each with a presence, a set of dental-arch cells and a multiplicity. A dentist
wants one report. This module

* condenses a saved result into one dense, fixed-shape JSON (structured_findings): every
  finding of the benchmark vocabulary and every extra DentVLM task, each with an explicit
  status ("present", "absent", "unparseable", "not_assessed"), the task(s) that decided it with
  their verbatim question and answer, every cell with an explicit value, the multiplicity, and
  the optional out-of-distribution count. Nothing is implicit or null, so the report model
  never has to guess what a missing value means;
* sends that JSON to a text LLM (ReportWriter, described by an llm_api spec exactly like the
  analyzer's API backend and the location adapter) with a fixed prompt that asks for a
  radiology-style report in the dentist's language, returned as JSON with one entry per
  finding, classified into seven sections;
* verifies the reply against the input (every finding exactly once, statuses unchanged,
  nothing invented), asks once for a correction when it fails, and renders the verified JSON
  into Markdown deterministically. A reply that still fails is saved as such and the
  deterministic dentist_report takes its place, so the dentist always gets a report and the
  failure stays visible.

The report model never sees the image: it can only reword what DentVLM answered. Its input is
the parsed answers, not the rationale text, unless include_rationale is switched on.
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

SCHEMA = "dentvlm-findings/1"

# The seven sections of the report in reading order, each with its findings in reporting order.
# DentVLM's three extra tasks (no benchmark class) sit next to the findings they belong with.
CATEGORIES = {
    "restorations_and_prostheses": ("Restorations and prostheses",
                                    ("dental_filling", "prosthetic_restoration", "dental_implant")),
    "endodontic": ("Endodontic status", ("endodontic_treatment", "apical_surgery")),
    "caries": ("Caries", ("carious_lesion",)),
    "periodontal": ("Periodontal status", ("periodontal_bone_loss", "calculus", "furcation_lesion")),
    "periapical": ("Periapical pathology", ("periapical_lesion",)),
    "teeth_and_eruption": ("Teeth, roots and eruption",
                           ("impacted_tooth", "insufficient_eruption_space", "root_fragment", "residual_crown",
                            "root_resorption")),
    "appliances_and_hardware": ("Appliances and surgical hardware", ("orthodontic_device", "surgical_device")),
}
IDENTIFIERS = dp.CONDITIONS + dp.EXTRA_TASKS
FINDING_CATEGORY = {c: key for key, (_, conditions) in CATEGORIES.items() for c in conditions}
assert sorted(FINDING_CATEGORY) == sorted(IDENTIFIERS), "every finding and extra task belongs to exactly one section"
LABELS = {**dp.LABELS, **{task: dp.TASKS[task]["name"] for task in dp.EXTRA_TASKS}}

# Impression order: pathology before treatment history.
PATHOLOGY = ("carious_lesion", "periapical_lesion", "periodontal_bone_loss", "calculus", "furcation_lesion",
             "impacted_tooth", "insufficient_eruption_space", "root_fragment", "residual_crown", "root_resorption")
TREATMENT = tuple(c for c in IDENTIFIERS if c not in PATHOLOGY)

STATUSES = ("present", "absent", "unparseable", "not_assessed")
GLYPHS = {"present": "●", "absent": "○", "unparseable": "?", "not_assessed": "–"}

LEGEND = {
    "status": {
        "present": "the analyzer answered Yes (with several tasks: at least one answered Yes; with region questions: at least one region answered Yes)",
        "absent": "the analyzer answered No (every task, or every region, answered No)",
        "unparseable": "an answer could not be read as Yes or No; the finding is neither confirmed nor excluded",
        "not_assessed": "the analyzer has no question for this finding, so it was never asked",
    },
    "regions (location from the rationale)": {
        "named": "the model's rationale placed the finding in this region",
        "not_named": "the rationale did not name this region; this is NOT evidence that the region is free of the finding",
        "not_applicable": "the finding is not present, so no region applies",
    },
    "regions (location from region questions)": {
        "present": "Yes when the question named this region", "absent": "No when the question named this region",
        "unparseable": "the answer for this region could not be read", "not_asked": "no question named this region",
    },
    "multiplicity": "the number of distinct regions the model named (0 to 6); a lower bound on the number of occurrences, not a tooth count",
    "count": "an experimental tooth count from an out-of-distribution question, or 'not_asked', 'unparseable', 'not_countable'",
    "trained": "false when the analyzer was never trained on this question (zero-shot; the paper reports 52-64% accuracy on such diseases)",
    "detection": "whether the whole-image answer and the region answers agree; a region-only detection is a weaker signal",
}
LIMITATIONS = (
    "Experimental output of an automated model for review by a dentist; not a diagnosis.",
    "The report writer never saw the radiograph; every statement rewords the analyzer's answers.",
)
RATIONALE_LIMITATION = ("Locations are the regions the model named in its rationale: a region it did not name is not "
                        "evidence of absence there, and the number of regions is a lower bound on the number of occurrences.")


# ----------------------------------------------------------------------------
# Structured input: one dense JSON per image
# ----------------------------------------------------------------------------
def _status(answer) -> str:
    return {"yes": "present", "no": "absent"}.get(answer, "unparseable")


def patient_cell(cell: str, left_is_image_left: bool = dp.LEFT_IS_IMAGE_LEFT) -> str:
    """Patient-side name of a DentVLM cell: 'upper-right-posterior', 'lower-anterior', ..."""
    row, col = cell.split("-")
    if col == "anterior":
        return f"{row}-anterior"
    image_side = col if left_is_image_left else dp._FLIP[col]
    return f"{row}-{dp._FLIP[image_side]}-posterior"


PATIENT_ORDER = ("upper-right-posterior", "upper-anterior", "upper-left-posterior",
                 "lower-right-posterior", "lower-anterior", "lower-left-posterior")


def ordered_cells(left_is_image_left: bool = dp.LEFT_IS_IMAGE_LEFT) -> tuple[str, ...]:
    """DentVLM's cells in the dentist's reading order: patient's right, anterior, patient's left; upper, then lower."""
    return tuple(sorted(dp.CELLS, key=lambda c: PATIENT_ORDER.index(patient_cell(c, left_is_image_left))))


def cell_text(cell: str, left_is_image_left: bool = dp.LEFT_IS_IMAGE_LEFT) -> str:
    """Anatomical wording of a cell for the dentist, on the patient's side."""
    row, col = cell.split("-")
    jaw = "maxilla" if row == "upper" else "mandible"
    if col == "anterior":
        return f"{row} anterior region (incisors and canines, {jaw})"
    side = patient_cell(cell, left_is_image_left).split("-")[1]
    return f"{row} {side} posterior region (the patient's {side}: premolars and molars, {jaw})"


def detection_note(status: str, whole_image: str, regional: bool) -> str:
    """How the whole-image answer and the region answers relate, for the dentist's confidence."""
    if not regional:
        return "whole-image question"
    if status == "present":
        if whole_image == "present":
            return "whole-image and region answers agree"
        if whole_image == "absent":
            return "region questions only: the whole-image question answered No (weaker signal)"
        return "region questions only: the whole-image answer was unparseable"
    if status == "absent":
        if whole_image == "present":
            return "every region answered No although the whole-image question answered Yes (discordant; treated as absent)"
        if whole_image == "absent":
            return "whole-image and region answers agree"
        return "every region answered No; the whole-image answer was unparseable"
    return "no region answered Yes and at least one region answer was unparseable"


def method_text(result: dict) -> str:
    protocol, level = result["protocol"], result.get("location_level", "rationale")
    n_tasks = len(result.get("tasks") or {})
    parts = [f"one yes/no question per task on the whole panoramic radiograph ({n_tasks} tasks"
             + (f", {protocol['phrasings']} verbatim wordings each with a vote)" if protocol.get("phrasings", 1) > 1 else ")")]
    if level == "rationale":
        parts.append("the location read from the model's own rationale, which names regions with nine fixed descriptors "
                     "mapped onto six dental-arch cells (upper/lower x right posterior/anterior/left posterior)")
    elif level == "regions":
        parts.append("then every task again once per dental-arch region, on the same whole image, with the region "
                     "named inside the question in the model's own words, whatever the whole image answered")
    else:
        parts.append("presence only, no location")
    if protocol.get("count_question"):
        parts.append("an experimental, out-of-distribution tooth-count question for each positive countable finding")
    return "; ".join(parts)


def _task_entry(key: str, task: dict, flag: bool, model_text: str | None) -> dict:
    answers = task.get("answers") or []
    entry = {
        "task": key, "name": task.get("name", dp.task_name(key)),
        "question": dp.questions_for(key)[0],
        "answer": _status(task["presence"]),
        "regions_named": [patient_cell(c, flag) for c in ordered_cells(flag) if c in (task.get("regions") or [])],
        "phrasings": len(answers) or 1,
    }
    if model_text is not None:
        entry["model_text"] = model_text
    return entry


def _entry(identifier: str, result: dict, cell_answers: dict, flag: bool, include_rationale: bool) -> dict:
    """One dense entry for a benchmark finding or an extra DentVLM task."""
    protocol, level = result["protocol"], result.get("location_level", "rationale")
    tasks_out = result.get("tasks") or {}
    regional = level == "regions"
    if identifier in dp.CONDITIONS:
        finding = result["findings"][identifier]
        keys = list(finding["tasks"]) if finding["asked"] else []
        presence, whole_image = finding["presence"], finding.get("whole_image")
        regions, region_count, count = finding.get("regions"), finding.get("region_count"), finding.get("count")
        benchmark_class, countable = True, identifier in dp.COUNTABLE
        trained = identifier in dp.TRAINED
    else:
        task = tasks_out.get(identifier)
        keys = [identifier] if task else []
        presence, whole_image = (task["presence"], task.get("whole_image")) if task else (None, None)
        regions, count = (task.get("regions") if task else None), None
        region_count = len(regions) if regions is not None else None
        benchmark_class, countable, trained = False, False, True
    asked = bool(keys)

    texts = {}
    if include_rationale:
        for call in result.get("calls") or []:
            if call.get("parse_recovery", {}).get("error"):
                continue
            if call.get("stage") == "presence" and call.get("task") in keys and call["task"] not in texts:
                texts[call["task"]] = (call.get("text") or "")[:600]
    tasks = [_task_entry(k, tasks_out[k], flag, texts.get(k) if include_rationale else None) for k in keys if k in tasks_out]

    status = "not_assessed" if not asked else _status(presence)
    whole = "not_assessed" if not asked else _status(whole_image if whole_image is not None else presence)
    cells = ordered_cells(flag)
    if not asked or level == "none":
        region_map, region_source = {}, "none"
    elif regional:
        region_source = "region_questions"
        region_map = {}
        for cell in cells:
            answers = [cell_answers.get(k, {}).get(cell, "missing") for k in keys]
            if any(a == "missing" for a in answers):
                region_map[patient_cell(cell, flag)] = "not_asked"
            elif "yes" in answers:
                region_map[patient_cell(cell, flag)] = "present"
            elif all(a == "no" for a in answers):
                region_map[patient_cell(cell, flag)] = "absent"
            else:
                region_map[patient_cell(cell, flag)] = "unparseable"
    else:
        region_source = "rationale"
        named = set(regions or []) if status == "present" else set()
        region_map = {patient_cell(c, flag): ("not_applicable" if status != "present" else "named" if c in named else "not_named")
                      for c in cells}
    located_in = [r for r, v in region_map.items() if v in ("present", "named")] if status == "present" else []
    if status != "present":
        location_status = "not_applicable"
    elif region_source == "none":
        location_status = "not_asked"
    elif located_in:
        location_status = "located"
    elif region_source == "rationale":
        location_status = "not_stated: the model's rationale named no region"
    elif "unparseable" in region_map.values():
        location_status = "unresolved: a cell answer was unparseable"
    else:
        location_status = "not_localized: present on the whole image, no cell answered Yes"
    multiplicity = len(located_in) if status == "present" and region_source != "none" else "not_applicable"

    if not countable:
        count_value = "not_countable"
    elif not protocol.get("count_question"):
        count_value = "not_asked"
    elif status != "present":
        count_value = "not_asked"
    else:
        count_value = count if count is not None else "unparseable"

    return {
        "finding": identifier, "label": LABELS[identifier], "category": FINDING_CATEGORY[identifier],
        "benchmark_class": benchmark_class, "trained": trained,
        "status": status, "whole_image": whole,
        "detection": "not_assessed" if not asked else detection_note(status, whole, regional),
        "tasks": tasks,
        "regions": region_map, "region_source": region_source, "located_in": located_in,
        "location_status": location_status, "multiplicity": multiplicity,
        "countable": countable, "count": count_value,
    }


def structured_findings(result: dict, analyzer: str | None = None, include_rationale: bool = False) -> dict:
    """One dense JSON for the report model: every finding, task, cell and count with an explicit status."""
    flag = result.get("left_is_image_left", dp.LEFT_IS_IMAGE_LEFT)
    level = result.get("location_level", "rationale")
    cell_answers = dp.cell_answers(result) if level == "regions" else {}  # the evaluator reads the same answers
    findings = [_entry(i, result, cell_answers, flag, include_rationale) for i in IDENTIFIERS]
    status = {f["finding"]: f["status"] for f in findings}
    order = PATHOLOGY + TREATMENT
    limitations = list(LIMITATIONS)
    if level == "rationale":
        limitations.append(RATIONALE_LIMITATION)
    if any(s == "not_assessed" for s in status.values()):
        limitations.append("Findings the analyzer has no question for were not assessed.")
    region_legend = (LEGEND["regions (location from region questions)"] if level == "regions"
                     else LEGEND["regions (location from the rationale)"])
    return {
        "schema": SCHEMA,
        "image": {"id": result.get("image_id", Path(result["image"]).stem), "file": Path(result["image"]).name,
                  "sha256": result.get("image_sha256")},
        "analysis": {
            "analyzer": analyzer or "DentVLM",
            "questions_asked": result.get("call_count"),
            "protocol": result["protocol"],
            "location_source": level,
            "regions": [{"name": patient_cell(c, flag), "location": cell_text(c, flag)} for c in ordered_cells(flag)],
            "method": method_text(result),
            "limitations": limitations,
        },
        "legend": {"status": LEGEND["status"], "regions": region_legend, "multiplicity": LEGEND["multiplicity"],
                   "count": LEGEND["count"], "trained": LEGEND["trained"], "detection": LEGEND["detection"]},
        "categories": [{"key": key, "label": label, "findings": list(conditions)}
                       for key, (label, conditions) in CATEGORIES.items()],
        "findings": findings,
        "summary": {
            "present": [c for c in order if status[c] == "present"],
            "absent": [c for c in order if status[c] == "absent"],
            "unparseable": [c for c in order if status[c] == "unparseable"],
            "not_assessed": [c for c in order if status[c] == "not_assessed"],
            "regional_only": [f["finding"] for f in findings if f["status"] == "present" and f["whole_image"] != "present"
                              and level == "regions"],
        },
    }


# ----------------------------------------------------------------------------
# Prompts (the core of the report writer; edit wording here only)
# ----------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a dental radiology report writer. You turn the structured output of an automated analysis of a "
    "panoramic dental radiograph into a report for a dentist. You never see the image: the JSON you are given is "
    "the only source of facts. You reword and organise it; you never add, drop, soften or upgrade a finding. "
    "You answer with JSON only, no prose before or after it."
)

OUTPUT_SCHEMA = """{
 "language": "<the language the report is written in>",
 "title": "Panoramic radiograph: automated findings report",
 "headings": {"image": "Image", "findings": "Findings", "impression": "Impression",
              "not_assessable": "Not assessable", "limitations": "Limitations"},
 "sections": [
  {"category": "restorations_and_prostheses", "heading": "Restorations and prostheses",
   "findings": [
    {"finding": "dental_filling", "status": "present", "statement": "<one or two sentences>"},
    {"finding": "prosthetic_restoration", "status": "absent", "statement": "<one short sentence>"},
    {"finding": "dental_implant", "status": "absent", "statement": "<one short sentence>"}
   ]},
  {"category": "endodontic", "heading": "Endodontic status", "findings": ["... every finding of the category ..."]},
  {"category": "caries", "heading": "Caries", "findings": ["..."]},
  {"category": "periodontal", "heading": "Periodontal status", "findings": ["..."]},
  {"category": "periapical", "heading": "Periapical pathology", "findings": ["..."]},
  {"category": "teeth_and_eruption", "heading": "Teeth, roots and eruption", "findings": ["..."]},
  {"category": "appliances_and_hardware", "heading": "Appliances and surgical hardware", "findings": ["..."]}
 ],
 "impression": ["<1 to 6 short bullets, pathology first>"],
 "not_assessable": ["<one sentence per finding whose status is unparseable; an empty list when there is none>"],
 "limitations": ["<the limitations from the data, in the dentist's language, plus any caveat the data raises>"]
}"""

USER_PROMPT = """Write the dentist's report for the automated analysis below.

WHAT THE DATA IS
An automated analyzer ({analyzer}) was asked {method}. The JSON lists the 14 findings of the benchmark vocabulary and the analyzer's extra tasks, each with an explicit status, the task(s) that decided it with their verbatim question and answer, every dental-arch region with an explicit value, the multiplicity (number of regions the model named), and the optional count. Region names are on the PATIENT's side ("analysis.regions" spells them out). "unparseable" means an answer could not be read as Yes or No, so that finding is neither confirmed nor excluded; "not_assessed" means the analyzer has no question for that finding and was never asked. Every value is spelled out; there are no implicit defaults.

{findings_json}

HOW TO WRITE
1. Language: write every human-readable value (title, headings, statements, impression, not_assessable, limitations) in {language}, with the dental terminology a dentist reading that language expects. Keep the JSON keys and every "finding" and "category" identifier exactly as given, in English.
2. Fidelity: one entry per finding, in the section "categories" assigns it to, with "status" copied unchanged. State regions, multiplicity and counts exactly as given; never estimate a count, never name a tooth number, never add or remove a region, and never mention a finding that is not in the data. When a value is "not_asked", "not_stated" or "unparseable", say so in words. A finding with status "not_assessed" gets one sentence saying the analyzer does not assess it.
3. Wording: as a radiologist reports to a colleague. Short declarative sentences, present tense, attributed to the automated analysis ("The analysis flags ..."). Locate findings on the patient's side ("upper right posterior region"); never say image left or image right. A region the model did not name is never reported as free of the finding. An absent finding gets one short pertinent-negative sentence. When a finding was decided by several tasks (for example a prosthetic crown and a prosthetic bridge), say which task answered Yes. No diagnosis, no differential, no severity, no treatment advice.
4. Confidence: say when a finding comes from a question the analyzer was not trained on ("trained": false), when "detection" says it was flagged by region questions only, and that a count is experimental.
5. Impression: 1 to 6 short bullets. Pathology first (caries, periapical lesions, periodontal disease, calculus, furcation involvement, impacted teeth, insufficient eruption space, residual roots and crowns, root resorption), then existing treatment (fillings, crowns or bridges, root canal treatments, implants, appliances, surgical hardware), then what could not be assessed. Absent and not-assessed findings stay out of the impression, unless every assessed finding is absent: then say so in one bullet.
6. Limitations: the sentences in "analysis.limitations", in the dentist's language, plus any caveat the data raises (unparseable answers, untrained questions, region-only detections).

OUTPUT
JSON only, exactly this shape; the English values are placeholders to translate, the structure and the identifiers are fixed:
{output_schema}"""

REPAIR_PROMPT = """Your reply failed these checks against the data:
{problems}

Return the complete corrected JSON only: same shape, same language, every finding exactly once with its status unchanged."""


def user_prompt(structured: dict, language: str) -> str:
    """The user message for one image (placeholders are replaced, never str.format, because of the JSON braces)."""
    analysis = structured["analysis"]
    return (USER_PROMPT.replace("{analyzer}", str(analysis["analyzer"])).replace("{method}", analysis["method"])
            .replace("{findings_json}", json.dumps(structured, indent=1, ensure_ascii=False))
            .replace("{language}", language).replace("{output_schema}", OUTPUT_SCHEMA))


# ----------------------------------------------------------------------------
# Reply parsing and verification against the input
# ----------------------------------------------------------------------------
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


def _strings(value, minimum: int = 0, maximum: int | None = None) -> bool:
    return (isinstance(value, list) and all(isinstance(v, str) and v.strip() for v in value)
            and len(value) >= minimum and (maximum is None or len(value) <= maximum))


def verify_report(report: dict | None, structured: dict) -> list[str]:
    """Problems with a reply, empty when it is a faithful report of the structured findings."""
    if not isinstance(report, dict):
        return ["the reply is not a JSON object"]
    problems = []
    for key in ("title", "headings", "sections", "impression", "not_assessable", "limitations"):
        if key not in report:
            problems.append(f"missing key {key!r}")
    if problems:
        return problems
    if not isinstance(report["title"], str) or not report["title"].strip():
        problems.append("'title' must be a non-empty string")
    headings = report["headings"]
    if not isinstance(headings, dict) or any(not isinstance(headings.get(k), str) or not headings[k].strip()
                                             for k in ("image", "findings", "impression", "not_assessable", "limitations")):
        problems.append("'headings' must hold non-empty strings for image, findings, impression, not_assessable, limitations")
    expected = {f["finding"]: f for f in structured["findings"]}
    seen: dict[str, int] = {}
    if not isinstance(report["sections"], list):
        problems.append("'sections' must be a list")
    else:
        for section in report["sections"]:
            if not isinstance(section, dict) or section.get("category") not in CATEGORIES:
                problems.append(f"a section has an unknown category: {section.get('category') if isinstance(section, dict) else section!r}")
                continue
            if not isinstance(section.get("heading"), str) or not section["heading"].strip():
                problems.append(f"section {section['category']!r} needs a non-empty 'heading'")
            for entry in section.get("findings") or []:
                finding = entry.get("finding") if isinstance(entry, dict) else None
                if finding not in expected:
                    problems.append(f"unknown finding {finding!r}: only the findings in the data may appear")
                    continue
                seen[finding] = seen.get(finding, 0) + 1
                if entry.get("status") != expected[finding]["status"]:
                    problems.append(f"{finding}: status must stay {expected[finding]['status']!r}, got {entry.get('status')!r}")
                if section["category"] != expected[finding]["category"]:
                    problems.append(f"{finding} belongs in section {expected[finding]['category']!r}, not {section['category']!r}")
                if not isinstance(entry.get("statement"), str) or not entry["statement"].strip():
                    problems.append(f"{finding}: 'statement' must be a non-empty string")
        missing = [c for c in expected if c not in seen]
        duplicated = [c for c, n in seen.items() if n > 1]
        if missing:
            problems.append("missing findings: " + ", ".join(missing))
        if duplicated:
            problems.append("findings listed more than once: " + ", ".join(duplicated))
    if not _strings(report["impression"], 1, 8):
        problems.append("'impression' must be a list of 1 to 8 non-empty strings")
    unparseable = structured["summary"]["unparseable"]
    if not _strings(report["not_assessable"]):
        problems.append("'not_assessable' must be a list of strings")
    elif unparseable and not report["not_assessable"]:
        problems.append("'not_assessable' must name the unparseable findings: " + ", ".join(unparseable))
    elif not unparseable and report["not_assessable"]:
        problems.append("'not_assessable' must be empty: no finding was unparseable")
    if not _strings(report["limitations"], 1):
        problems.append("'limitations' must be a non-empty list of strings")
    return problems


# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------
def render_markdown(report: dict, structured: dict, writer_model: str | None = None) -> str:
    """Deterministic Markdown from a verified report: findings by section, impression, caveats."""
    headings = report["headings"]
    image, analysis = structured["image"], structured["analysis"]
    lines = [f"# {report['title']}", "", f"**{headings['image']}:** {image['file']}", "", f"## {headings['findings']}"]
    by_category: dict[str, dict] = {}
    for section in report["sections"]:
        slot = by_category.setdefault(section["category"], {"heading": section["heading"], "findings": []})
        slot["findings"].extend(section.get("findings") or [])
    for key, (_, conditions) in CATEGORIES.items():
        section = by_category.get(key)
        if not section:
            continue
        order = {c: i for i, c in enumerate(conditions)}
        lines += ["", f"### {section['heading']}"]
        for entry in sorted(section["findings"], key=lambda e: order.get(e["finding"], len(order))):
            lines.append(f"- {GLYPHS[entry['status']]} {entry['statement'].strip()}")
    lines += ["", f"## {headings['impression']}"] + [f"- {b.strip()}" for b in report["impression"]]
    if report["not_assessable"]:
        lines += ["", f"## {headings['not_assessable']}"] + [f"- {t.strip()}" for t in report["not_assessable"]]
    lines += ["", f"## {headings['limitations']}"] + [f"- {t.strip()}" for t in report["limitations"]]
    footer = f"{analysis['analyzer']} · {analysis['questions_asked']} questions"
    if writer_model:
        footer += f" → {writer_model}"
    lines += ["", "---", f"*{footer}*", ""]
    return "\n".join(lines)


def fallback_markdown(result: dict, problems: list[str]) -> str:
    """The deterministic summary, used when the report model's reply could not be verified."""
    lines = ["# Automatic summary (the report model's reply failed verification)", ""]
    lines += [f"- {p}" for p in problems]
    lines += ["", "```", dp.dentist_report(result), "```", ""]
    return "\n".join(lines)


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "-", text).strip("-").lower() or "model"


# ----------------------------------------------------------------------------
# Report writer (OpenAI-compatible chat completions; text only)
# ----------------------------------------------------------------------------
class ReportWriter:
    """Structured findings of one image -> verified report JSON + Markdown, from a hosted text LLM.

    from_api() builds one from an llm_api spec. token_param "max_completion_tokens" and temperature
    None for OpenAI reasoning models; other request fields (reasoning_effort, response_format, ...)
    go through request_options. include_rationale adds DentVLM's own reply text per task to the
    input (off by default: the report then rests on the parsed answers alone). One repair turn is
    allowed: the reply's problems are sent back and the corrected JSON re-verified.
    """

    kind = "report"
    OPTIONS = ("token_param", "temperature", "max_output_tokens", "request_options", "language", "repairs",
               "include_rationale", "api_call_retries")

    def __init__(self, base_url: str | None, api_key: str, model: str, token_param: str = "max_tokens",
                 max_output_tokens: int = 4096, temperature: float | None = 0.0, language: str = "English",
                 repairs: int = 1, include_rationale: bool = False, timeout: float = 600.0,
                 request_options: dict | None = None, api_call_retries: int = 2,
                 call_log: str | None = None, client=None) -> None:
        if token_param not in llm_api.TOKEN_PARAMS:
            raise ValueError(f"token_param must be one of {llm_api.TOKEN_PARAMS}")
        if not isinstance(language, str) or not language.strip():
            raise ValueError("language must be a non-empty string, e.g. 'English' or 'Persian'")
        llm_api.validate_api_retries(api_call_retries)
        self.client = client if client is not None else llm_api.connect(base_url, api_key, timeout)
        self.base_url, self.model = base_url, model
        self.token_param, self.max_output_tokens, self.temperature = token_param, max_output_tokens, temperature
        self.language, self.repairs, self.include_rationale = language.strip(), max(0, int(repairs)), bool(include_rationale)
        self.request_options = dict(request_options or {})
        self.api_call_retries = api_call_retries
        self.call_log = mon.CallLog("report", call_log)

    @classmethod
    def from_api(cls, spec: dict, language: str | None = None, timeout: float = 600.0, client=None) -> "ReportWriter":
        """Writer for a hosted model. spec = {"provider", "model", ...} as documented in llm_api, plus any
        of the constructor options named in OPTIONS; a language argument wins over the spec's."""
        base_url, api_key = llm_api.resolve(spec)
        options = {k: spec[k] for k in cls.OPTIONS if k in spec}
        if language is not None:
            options["language"] = language
        return cls(base_url, api_key, spec["model"], timeout=timeout, client=client, **options)

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
                "api_call_retries": self.api_call_retries,
                "request_options": self.request_options, "schema": SCHEMA,
                "system_prompt": SYSTEM_PROMPT, "user_prompt": USER_PROMPT, "output_schema": OUTPUT_SCHEMA,
                "repair_prompt": REPAIR_PROMPT}

    def public(self) -> dict:
        """The settings without the prompt texts, for printouts."""
        return {k: v for k, v in self.settings().items()
                if k not in ("system_prompt", "user_prompt", "output_schema", "repair_prompt")}

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
        structured = structured_findings(result, analyzer, self.include_rationale)
        prompt = user_prompt(structured, self.language)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        attempts, report, problems = [], None, ["no reply"]
        for _attempt in range(1 + self.repairs):
            reply = self._ask(messages)
            report = extract_json(reply["text"])
            problems = verify_report(report, structured)
            if reply["truncated"] and problems:
                problems.append("the reply was cut off by max_output_tokens")
            attempts.append({**reply, "problems": problems})
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
            "language": self.language, "writer": self.public(), "analyzer": structured["analysis"]["analyzer"],
            "structured": structured, "prompt": prompt,
            "report": report if verified else None, "verified": verified, "problems": problems,
            "markdown": render_markdown(report, structured, self.model) if verified else fallback_markdown(result, problems),
            "attempts": attempts,
        }


# ----------------------------------------------------------------------------
# Dataset loop with resume, loading, summary
# ----------------------------------------------------------------------------
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
            _load_report_file(target, image_id)  # a corrupt or foreign artifact stops the loop
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
    progress.done(detail=log.line(counts=False) if isinstance(log, mon.CallLog) else "")
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
    return {"images": len(reports), "verified": len(verified),
            "repaired": sum(len(r["attempts"]) > 1 for r in verified),
            "fallback": len(reports) - len(verified),
            "mean_completion_tokens": round(sum(tokens) / len(tokens)) if tokens else None}
