"""Dentist report: one text-LLM call turns the per-image findings into a classified report.

DentVLM answers one yes/no question per task on the whole image and names where it sees the
finding in its rationale; the pipeline turns that into 14 benchmark findings plus the model's
extra tasks, each with a presence, a set of dental-arch cells and a multiplicity. A dentist
wants one report. This module

* condenses a saved result into one dense, fixed-shape JSON (structured_findings): every
  finding of the benchmark vocabulary and every extra DentVLM task, each with an explicit
  status ("present", "absent", "unparseable", "not_assessed"), the task(s) that decided it with
  their verbatim question and answer, every cell with an explicit value, and the multiplicity.
  Nothing is implicit or null, so the report model never has to guess what a missing value means;
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

With vote_agreement switched on, the vote behind each answer reaches the report as well: how many
of the question's wordings reported the finding, and how many of them named each region, as counts
the report quotes rather than turns into a confidence. It is off by default, and while it is off
the structured input, the prompt, the report and the rendering are exactly what they were.

The multiplicity is the occupied-region count of dental_pipeline.count_block: the number of distinct
regions the model reported a finding in, said as "reported in two regions" and never as two lesions
or two teeth, with words instead of a number when it is partial, not stated or unresolved. With
counting switched off it leaves the structured input, the legend and the prompt altogether.
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

# The top-level keys of the report object, in reading order; the verification requires every one.
REPORT_KEYS = ("title", "headings", "sections", "impression", "not_assessable", "limitations")
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
    "multiplicity": "the number of distinct regions the model reported the finding in (0 to 6): regions, never lesions or "
                    "teeth, and a lower bound on the number of occurrences. Words instead of a number say why it is not "
                    "available: 'at least N' when some regions could not be read, 'not_stated' when the model named no "
                    "region, 'unresolved' when its location could not be read",
    "trained": "false when the analyzer was never trained on this question (zero-shot; the paper reports 52-64% accuracy on such diseases)",
    "detection": "whether the whole-image answer and the region answers agree; a region-only detection is a weaker signal",
}
LIMITATIONS = (
    "Experimental output of an automated model for review by a dentist; not a diagnosis.",
    "The report writer never saw the radiograph; every statement rewords the analyzer's answers.",
)
RATIONALE_LIMITATION = ("Locations are the regions the model named in its rationale: a region it did not name is not "
                        "evidence of absence there, and the number of regions is a lower bound on the number of occurrences.")
UNRESOLVED_LOCATION = ("the finding is present but the location in the model's rationale could not be read at all; this "
                       "is not evidence about any region, and no region may be reported for this finding")


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
    return "; ".join(parts)


# ----------------------------------------------------------------------------
# Agreement between prompt phrasings (optional, off by default)
# ----------------------------------------------------------------------------
# With protocol.phrasings > 1 every task is asked with several verbatim wordings of the same
# question and dental_pipeline.vote() turns their answers into one decision. The vote is saved
# per task under "answers"; the report only ever saw the decision. With vote_agreement switched
# on, those saved answers reach the report as explicit counts: how many wordings reported the
# finding, and how many of them named each region, so "lower-left 3/3" and "upper-right 1/3"
# stay distinguishable instead of arriving already merged by region_vote="union".
#
# Nothing here changes a prediction. The counts are read from the saved answers, the decision
# stays the one the pipeline made under its own union/majority policy, and the evaluation never
# sees this module. The counts measure how stable the model is under rewording, which is not a
# probability and not a confidence; every text below says so, because a reader will assume
# otherwise unless told.

AGREEMENT_BANDS = {
    "consistent": "consistently identified: every wording asked gave a readable answer and all of them agreed",
    "moderate": "moderately supported: most readable answers agreed, but not every wording agreed or not every wording was readable",
    "weak": "weakly supported / not consistent: only a minority of the readable answers agreed",
    "tie": "not consistent: the readable answers split evenly, so the analyzer recorded no decision",
    "single_source": "single source: only one answer reported the finding, so there is no agreement between wordings to measure",
    "none_named": "not named: no answer that reported the finding placed it in this region, which is not evidence that the region is free of it",
    "single": "not measured: this run asked one wording per task, so there is no agreement between wordings to measure",
    "none": "not measured: no answer to this question could be read as Yes or No",
}
AGREEMENT_DISCLAIMER = (
    "Vote counts say how many rewordings of the same question, put to the same model on the same image, agreed "
    "with each other. They are not a probability, not medical certainty and not diagnostic confidence: wordings "
    "that agree can all be wrong together, and a finding only one wording reported can still be real."
)
AGREEMENT_LIMITATION = (
    "Where this report gives vote counts, they say how many rewordings of the same question agreed with each "
    "other; they are not a probability, not medical certainty and not a diagnostic confidence."
)
AGREEMENT_LEGEND = {
    "what the counts are": AGREEMENT_DISCLAIMER,
    "measured": "false when this run asked one wording per task, when no answer was readable, or when the finding "
                "was never asked; the block then carries the reason in its 'wording' and there is no agreement to discuss",
    "presence": "how many readable answers reported the status the analyzer recorded, out of the readable answers "
                "('vote', e.g. '2/3'); 'wordings_requested' is how many wordings were asked, 'answers_received' how "
                "many came back and 'unreadable' how many could not be read as Yes or No, so a vote is never reported "
                "as 3/3 when only two answers were valid",
    "presence.tie": "true when the readable answers split evenly; the analyzer then recorded no decision and the "
                    "status of the finding is 'unparseable'",
    "regions": "per region, how many of the answers that reported the finding named that region; 'out_of' is that "
               "number of answers, never the number of wordings asked. Every region is listed with its own count and "
               "the counts are never added up or merged, so a region named by every reporting answer and a region "
               "named by one of them are never equally supported",
    "regions_basis": "which answers the region counts are taken over, or why there are none",
    "region_vote_policy": "how the run turned these region counts into the regions it reports: 'union' keeps every "
                          "named region, 'majority' keeps the regions a majority of the reporting answers named. "
                          "'located_in' and 'regions' on the finding are the decision; these counts are the evidence "
                          "behind it, and they can disagree with it",
    "band / wording": "the phrase to use for these counts, and no other confidence word. A presence count carries "
                      "its phrase in 'wording'; a region count carries its 'band' only, and 'bands' below gives the "
                      "phrase that belongs to every band",
    "bands": AGREEMENT_BANDS,
    "scope": "which question the presence counts describe",
}


def agreement_band(votes: int, readable: int, requested: int) -> str:
    """Band for a presence vote. 'consistent' only when every wording asked answered and agreed."""
    if requested < 2:
        return "single"
    if readable == 0:
        return "none"
    if votes * 2 == readable:
        return "tie"
    if votes == readable == requested:
        return "consistent"
    if votes * 2 > readable:
        return "moderate"
    return "weak"


def region_band(votes: int, out_of: int) -> str:
    """Band for a region vote, counted over the answers that reported the finding, not over the wordings."""
    if out_of == 0:
        return "none"
    if votes == 0:
        return "none_named"
    if out_of == 1:
        return "single_source"
    if votes == out_of:
        return "consistent"
    if votes * 2 > out_of:
        return "moderate"
    return "weak"


def _vote(votes: int, out_of: int, band: str, brief: bool = False, **extra) -> dict:
    """One count. brief leaves out the phrase, which the legend's "bands" spells out for every band,
    so six regions on one finding do not repeat the same sentence six times."""
    return {"vote": f"{votes}/{out_of}", "votes": votes, "out_of": out_of, "band": band,
            **({} if brief else {"wording": AGREEMENT_BANDS[band]}), **extra}


def task_agreement(task: dict, requested: int, level: str, flag: bool, region_vote: str) -> dict:
    """The saved per-wording answers of one task as counts. Reads only what the run saved; decides nothing.

    An answer that could not be read as Yes or No is counted as unreadable, never as a No, so the
    denominator is the readable answers and the wordings asked stay visible next to it.
    """
    answers = list(task.get("answers") or [])
    readable = [a.get("answer") for a in answers if a.get("answer") in ("yes", "no")]
    present, absent = readable.count("yes"), readable.count("no")
    decision = task.get("whole_image", task.get("presence"))
    votes = present if decision == "yes" else absent if decision == "no" else max(present, absent)
    measured = requested >= 2 and bool(readable)
    presence = _vote(votes, len(readable), agreement_band(votes, len(readable), requested),
                     wordings_requested=requested, answers_received=len(answers),
                     unreadable=len(answers) - len(readable),
                     present_votes=present, absent_votes=absent,
                     decision={"yes": "present", "no": "absent"}.get(decision, "no decision"),
                     tie=bool(readable) and present == absent)

    reporting = [a for a in answers if a.get("answer") == "yes"]
    if level != "rationale":
        regions, basis = {}, ("not measured: this run asked each region its own question, once, so there is no "
                              "vote between wordings for a region" if level == "regions" else
                              "not measured: this run asked for presence only, with no location")
    elif not reporting:
        regions, basis = {}, "no readable answer reported the finding, so no wording named a region"
    else:
        out_of = len(reporting)
        counts = {c: sum(c in (a.get("regions") or []) for a in reporting) for c in ordered_cells(flag)}
        if not any(counts.values()):  # nothing to keep apart: no wording that reported it named a region
            regions, basis = {}, (f"the {out_of} answer(s) that reported the finding named no region, which is "
                                  "not evidence that the finding is absent anywhere")
        else:
            regions = {patient_cell(c, flag): _vote(n, out_of, region_band(n, out_of), brief=True)
                       for c, n in counts.items()}
            basis = (f"counted over the {out_of} answer(s) that reported the finding; a region none of them named "
                     f"is 0/{out_of}, which is not evidence that the region is free of the finding")
    return {
        "measured": measured,
        "reason": "" if measured else ("this run asked one wording per task" if requested < 2
                                       else "no answer to this task could be read as Yes or No"),
        "scope": ("the whole-image question only; this run decided presence from the per-region questions"
                  if level == "regions" else "the presence of this finding"),
        "presence": presence, "regions": regions, "regions_basis": basis, "region_vote_policy": region_vote,
    }


def finding_agreement(blocks: dict[str, dict]) -> dict:
    """One agreement block for a finding, attributed to the task whose vote carries it.

    A finding can rest on several tasks (a prosthetic crown and a prosthetic bridge both make a
    prosthetic restoration) and is present when any of them answers Yes. Its presence vote is then
    the vote of the task that reported it, named in "from_task"; a region keeps the count of the
    task that named it most often. Nothing is summed across tasks, and every task's own block stays
    in the finding's "tasks" list, so no evidence is merged away here.
    """
    if not blocks:
        return {"measured": False, "reason": "the analyzer has no question for this finding, so nothing was voted",
                "wording": AGREEMENT_BANDS["none"], "presence": None, "regions": {},
                "regions_basis": "the finding was never asked", "tasks_voted": []}
    order = list(blocks)
    reporting = [k for k in order if blocks[k]["presence"]["decision"] == "present"]
    if reporting:  # any-yes aggregation: the task that reported it carries the finding
        decided_by = max(reporting, key=lambda k: (blocks[k]["presence"]["votes"], -order.index(k)))
    else:  # no task reported it: the least consistent answer is the honest one to show
        decided_by = min(order, key=lambda k: (blocks[k]["presence"]["votes"], order.index(k)))
    several = len(blocks) > 1
    presence = {**blocks[decided_by]["presence"], **({"from_task": decided_by} if several else {})}

    sources = reporting or order
    names = next((list(blocks[k]["regions"]) for k in sources if blocks[k]["regions"]), [])
    regions = {}
    for name in names:
        holders = [k for k in sources if name in blocks[k]["regions"]]
        best = max(holders, key=lambda k: (blocks[k]["regions"][name]["votes"], -order.index(k)))
        regions[name] = {**blocks[best]["regions"][name], **({"from_task": best} if several else {})}
    return {
        "measured": blocks[decided_by]["measured"],
        "reason": blocks[decided_by]["reason"],
        "scope": blocks[decided_by]["scope"],
        "presence": presence, "regions": regions,
        "regions_basis": blocks[decided_by]["regions_basis"],
        "region_vote_policy": blocks[decided_by]["region_vote_policy"],
        "tasks_voted": order,
    }


def _task_entry(key: str, task: dict, flag: bool, model_text: str | None,
                agreement: dict | None = None) -> dict:
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
    if agreement is not None:
        entry["agreement"] = agreement
    return entry


def _entry(identifier: str, result: dict, cell_answers: dict, flag: bool, include_rationale: bool,
           vote_agreement: bool = False, counting: bool = True) -> dict:
    """One dense entry for a benchmark finding or an extra DentVLM task."""
    level = result.get("location_level", "rationale")
    tasks_out = result.get("tasks") or {}
    regional = level == "regions"
    if identifier in dp.CONDITIONS:
        finding = result["findings"][identifier]
        keys = list(finding["tasks"]) if finding["asked"] else []
        presence, whole_image = finding["presence"], finding.get("whole_image")
        regions = finding.get("regions")
        benchmark_class = True
        trained = identifier in dp.TRAINED
    else:
        finding = None
        task = tasks_out.get(identifier)
        keys = [identifier] if task else []
        presence, whole_image = (task["presence"], task.get("whole_image")) if task else (None, None)
        regions = task.get("regions") if task else None
        benchmark_class, trained = False, True
    asked = bool(keys)

    texts = {}
    if include_rationale:
        for call in result.get("calls") or []:
            if call.get("parse_recovery", {}).get("error"):
                continue
            if call.get("stage") == "presence" and call.get("task") in keys and call["task"] not in texts:
                texts[call["task"]] = (call.get("text") or "")[:600]
    voted, blocks = [k for k in keys if k in tasks_out], {}
    if vote_agreement:
        protocol = result["protocol"]
        wordings, policy = int(protocol.get("phrasings", 1) or 1), protocol.get("region_vote", "union")
        blocks = {k: task_agreement(tasks_out[k], wordings, level, flag, policy) for k in voted}
    # Only a finding decided by several tasks needs its tasks' votes spelled out: with one task the
    # finding's own block is that task's block, and repeating it would double the prompt for nothing.
    per_task = len(voted) > 1
    tasks = [_task_entry(k, tasks_out[k], flag, texts.get(k) if include_rationale else None,
                         blocks.get(k) if per_task else None) for k in voted]

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
    elif status == "present" and regions is None:
        # Present, but the reader could not say where. "not_named" would claim the model named no
        # region, which is a different and stronger statement than "we could not read it".
        region_source = "rationale"
        region_map = {patient_cell(c, flag): "unresolved" for c in cells}
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
        location_status = ("unresolved: the location in the model's rationale could not be read"
                           if regions is None else "not_stated: the model's rationale named no region")
    elif "unparseable" in region_map.values():
        location_status = "unresolved: a cell answer was unparseable"
    else:
        location_status = "not_localized: present on the whole image, no cell answered Yes"

    entry = {
        "finding": identifier, "label": LABELS[identifier], "category": FINDING_CATEGORY[identifier],
        "benchmark_class": benchmark_class, "trained": trained,
        "status": status, "whole_image": whole,
        "detection": "not_assessed" if not asked else detection_note(status, whole, regional),
        "tasks": tasks,
        "regions": region_map, "region_source": region_source, "located_in": located_in,
        "location_status": location_status,
    }
    if counting:
        # The occupied-region count: the same block the evaluation scores, said in words when it is not
        # a number, so the report can never turn a partial or unlocated finding into a total.
        unresolved = [c for c in cells if region_map.get(patient_cell(c, flag)) == "unparseable"]
        if finding is not None:
            block = dp.finding_count(finding, level, unresolved)
        else:
            block = dp.count_block(presence, regions, unresolved, level)
        entry["multiplicity"] = multiplicity_text(block, len(located_in), unresolved, flag) if status == "present" \
            and region_source != "none" else "not_applicable"
    if vote_agreement:
        entry["agreement"] = finding_agreement(blocks)
    return entry


def multiplicity_text(block: dict, confirmed: int, unresolved: list[str], flag: bool):
    """The multiplicity of a present finding: the resolved count, or the words that say why there is none."""
    status = block["count_status"]
    if status == "resolved":
        return block["region_count"]
    if status == "partial":
        names = ", ".join(patient_cell(c, flag) for c in unresolved)
        return (f"at least {confirmed}: reported in {confirmed} region(s), and the answer for {len(unresolved)} "
                f"region(s) could not be read ({names})")
    if status == "unlocated":
        return "not_stated: the model reported the finding but named no region, so the number of regions is unknown"
    return "unresolved: the location in the model's rationale could not be read, so the number of regions is unknown"


def structured_findings(result: dict, analyzer: str | None = None, include_rationale: bool = False,
                       vote_agreement: bool = False, parser=None, counting: bool = True) -> dict:
    """One dense JSON for the report model: every finding, task and cell with an explicit status.

    vote_agreement adds one "agreement" block per finding and per task, and its legend; with it off
    the JSON is byte for byte the one the report writer has always been given. counting off leaves
    the multiplicity out of every finding, the legend and the prompt.
    """
    flag = result.get("left_is_image_left", dp.LEFT_IS_IMAGE_LEFT)
    level = result.get("location_level", "rationale")
    # The evaluator reads the same answers, through the same reader, so the report and the score can
    # never disagree about what a region call said.
    cell_answers = dp.cell_answers(result, parser) if level == "regions" else {}
    findings = [_entry(i, result, cell_answers, flag, include_rationale, vote_agreement, counting) for i in IDENTIFIERS]
    status = {f["finding"]: f["status"] for f in findings}
    order = PATHOLOGY + TREATMENT
    limitations = list(LIMITATIONS)
    if level == "rationale":
        limitations.append(RATIONALE_LIMITATION)
    if any(s == "not_assessed" for s in status.values()):
        limitations.append("Findings the analyzer has no question for were not assessed.")
    if vote_agreement:
        limitations.append(AGREEMENT_LIMITATION)
    region_legend = (LEGEND["regions (location from region questions)"] if level == "regions"
                     else LEGEND["regions (location from the rationale)"])
    if any("unresolved" in f["regions"].values() for f in findings):
        region_legend = {**region_legend, "unresolved": UNRESOLVED_LOCATION}
    phrasings = int(result["protocol"].get("phrasings", 1) or 1)
    agreement = {"wordings_per_task": phrasings, "measured": phrasings > 1,
                 "region_vote_policy": result["protocol"].get("region_vote", "union"),
                 "what_the_counts_are": AGREEMENT_DISCLAIMER} if vote_agreement else None
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
            **({"vote_agreement": agreement} if vote_agreement else {}),
        },
        "legend": {"status": LEGEND["status"], "regions": region_legend,
                   **({"multiplicity": LEGEND["multiplicity"]} if counting else {}),
                   "trained": LEGEND["trained"], "detection": LEGEND["detection"],
                   **({"agreement": AGREEMENT_LEGEND} if vote_agreement else {})},
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
An automated analyzer ({analyzer}) was asked {method}. The JSON lists the 14 findings of the benchmark vocabulary and the analyzer's extra tasks, each with an explicit status, the task(s) that decided it with their verbatim question and answer{multiplicity_data}, and every dental-arch region with an explicit value. Region names are on the PATIENT's side ("analysis.regions" spells them out). "unparseable" means an answer could not be read as Yes or No, so that finding is neither confirmed nor excluded; "not_assessed" means the analyzer has no question for that finding and was never asked. Every value is spelled out; there are no implicit defaults.

{findings_json}

HOW TO WRITE
1. Language: write every human-readable value (title, headings, statements, impression, not_assessable, limitations) in {language}, with the dental terminology a dentist reading that language expects. Keep the JSON keys and every "finding" and "category" identifier exactly as given, in English.
2. Fidelity: one entry per finding, in the section "categories" assigns it to, with "status" copied unchanged. State regions{multiplicity_rule} exactly as given; never estimate a number of teeth, never name a tooth number, never add or remove a region, and never mention a finding that is not in the data. When a value is "not_asked", "not_stated" or "unparseable", say so in words. A finding with status "not_assessed" gets one sentence saying the analyzer does not assess it.
3. Wording: as a radiologist reports to a colleague. Short declarative sentences, present tense, attributed to the automated analysis ("The analysis flags ..."). Locate findings on the patient's side ("upper right posterior region"); never say image left or image right. A region the model did not name is never reported as free of the finding. An absent finding gets one short pertinent-negative sentence. When a finding was decided by several tasks (for example a prosthetic crown and a prosthetic bridge), say which task answered Yes. No diagnosis, no differential, no severity, no treatment advice.
4. Confidence: say when a finding comes from a question the analyzer was not trained on ("trained": false) and when "detection" says it was flagged by region questions only.
5. Impression: 1 to 6 short bullets. Pathology first (caries, periapical lesions, periodontal disease, calculus, furcation involvement, impacted teeth, insufficient eruption space, residual roots and crowns, root resorption), then existing treatment (fillings, crowns or bridges, root canal treatments, implants, appliances, surgical hardware), then what could not be assessed. Absent and not-assessed findings stay out of the impression, unless every assessed finding is absent: then say so in one bullet.
6. Limitations: the sentences in "analysis.limitations", in the dentist's language, plus any caveat the data raises (unparseable answers, untrained questions, region-only detections).

OUTPUT
JSON only, exactly this shape; the English values are placeholders to translate, the structure and the identifiers are fixed:
{output_schema}"""

# The agreement passages. USER_PROMPT itself never changes: these are spliced in at two anchors when
# the structured input carries agreement blocks, so with the knob off the model sees the same prompt.
HOW_TO_WRITE_ANCHOR = "\nHOW TO WRITE\n"
OUTPUT_ANCHOR = "\nOUTPUT\n"

AGREEMENT_DATA = """
AGREEMENT BETWEEN WORDINGS
Every task was asked with several verbatim wordings of the same question, and the analyzer's answer is the vote of those wordings. Each finding and each of its tasks therefore carries an "agreement" block holding the counts of that vote, read from the saved answers. "presence" says how many of the readable answers reported the status that was recorded, out of the readable answers ("vote", for example "2/3"), and spells out next to it how many wordings were asked ("wordings_requested"), how many answers could not be read as Yes or No ("unreadable") and whether the readable answers split evenly ("tie"). "regions" says, region by region, how many of the answers that reported the finding named that region, counted over those answers only ("out_of"); the regions are listed separately and their counts are never added together. "band" and "wording" give the fixed phrase that belongs to those counts - a region count carries its "band" only, and the legend's "bands" gives the phrase for each one - and "measured": false means there is nothing to report for that finding, with the reason in its "wording". The "agreement" entry of the legend defines every field.
These counts measure agreement between rewordings of one question, put to one model, on one image. They are NOT a probability, NOT medical certainty and NOT diagnostic confidence: wordings that agree can be wrong together, and a finding that only one wording reported can still be real.
"""

AGREEMENT_RULE = """7. Agreement between wordings: report the agreement of each finding next to that finding, from its "agreement" block. Give the counts exactly as they stand (presence as "<votes>/<out_of>", each region as "<region> <votes>/<out_of>") and the fixed phrase from "wording", in {language}. Never invent a percentage, a probability, a confidence, a certainty or any confidence word the block does not give you; never re-derive, round, average or add up the counts; never quote a count that is not in the block, and in particular never report a vote out of the wordings asked when fewer answers than that were readable. Keep every region separate with its own count, so a region named by every reporting answer and a region named by one of them are never presented as equally supported. Say in words when answers were unreadable, when the wordings tied, and when fewer answers were readable than wordings asked. Where the counts and the recorded result differ, because the run's region_vote policy kept a region a minority named or dropped one, report the recorded finding and its regions first and the counts as the evidence behind them. When "measured" is false, give the reason from its "wording" and say nothing further about agreement for that finding. Say once, in the limitations, that these counts measure agreement between rewordings of the same question and are not a probability, medical certainty or diagnostic confidence.
"""

# The multiplicity passages, present exactly when the structured input carries a multiplicity (counting on).
MULTIPLICITY_DATA = (", the multiplicity (the number of distinct regions the model reported the finding in - regions, "
                     "never teeth or lesions - or the words saying why that number is not available)")
MULTIPLICITY_RULE = (' and multiplicity (say "reported in two regions", never "two lesions" or "two teeth"; a '
                     'multiplicity given as words - "at least", "not_stated", "unresolved" - is said in those words)')

REPAIR_PROMPT = """Your reply failed these checks against the data:
{problems}

Return the complete corrected JSON only: same shape, same language, every finding exactly once with its status unchanged."""


def with_agreement(template: str) -> str:
    """USER_PROMPT plus the two agreement passages. The anchors are checked, so a future edit of the
    prompt that loses one fails loudly instead of dropping the rules from the message."""
    for anchor, passage in ((HOW_TO_WRITE_ANCHOR, AGREEMENT_DATA), (OUTPUT_ANCHOR, AGREEMENT_RULE)):
        if template.count(anchor) != 1:
            raise ValueError(f"the report prompt no longer holds exactly one {anchor!r} anchor to splice into")
        template = template.replace(anchor, passage + anchor)
    return template


def user_prompt(structured: dict, language: str) -> str:
    """The user message for one image (placeholders are replaced, never str.format, because of the JSON braces).

    The agreement passages are added exactly when the structured input carries agreement blocks, so the
    prompt can never describe data the model was not given."""
    analysis = structured["analysis"]
    template = with_agreement(USER_PROMPT) if "agreement" in structured["legend"] else USER_PROMPT
    counting = "multiplicity" in structured["legend"]
    return (template.replace("{analyzer}", str(analysis["analyzer"])).replace("{method}", analysis["method"])
            .replace("{multiplicity_data}", MULTIPLICITY_DATA if counting else "")
            .replace("{multiplicity_rule}", MULTIPLICITY_RULE if counting else "")
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


_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_FRACTION = re.compile(r"(\d+)\s*/\s*(\d+)")
_ODD_SLASH = re.compile(r"\d\s*[\u2044\u2215\uff0f]\s*\d")
_SPELLED_PAIR = re.compile(r"(\d+)([^\d]{1,12}?)(\d+)")


def quoted_votes(text: str, limit: int) -> set[tuple[int, int]]:
    """The vote counts a sentence quotes ("2/3", and the same in Eastern Arabic digits).

    Only fractions whose denominator could be a vote are read, so a number that is not a vote is
    left alone rather than turned into a false failure.
    """
    return {(int(v), int(n)) for v, n in _FRACTION.findall(str(text).translate(_DIGITS)) if 1 <= int(n) <= limit}


def quoted_votes_checked(text: str, limit: int) -> tuple[set[tuple[int, int]], str | None]:
    """The counts this reader can read, plus an explicit failure when a vote claim is out of its reach.

    The strict reader only understands "<votes>/<out_of>". A report is written in the dentist's
    language, so the same vote can arrive as "2 out of 3" or with a fraction slash this reader does
    not know. Those are the cases where it cannot say it read the sentence, so it says so instead:
    two numbers that could be a vote separated by words rather than a slash, or a fraction written
    with a slash character it does not accept. Anything else - tooth numbers, dates, list positions -
    is left alone rather than turned into a failure.

    A vote written entirely in words ("two of the three wordings") carries no digits and is
    therefore invisible to any code reader, this one included. That is the case the "llm" mode of
    this stage exists for: it reads every statement rather than waiting to be told it failed.
    """
    votes = quoted_votes(text, limit)
    if limit < 1:
        return votes, None
    flat = str(text).translate(_DIGITS)
    if _ODD_SLASH.search(flat):
        return votes, "unreadable_vote_claim"
    for left, gap, right in _SPELLED_PAIR.findall(flat):
        if "/" in gap or not any(character.isalpha() for character in gap):
            continue  # a strict fraction this reader already has, or two plain numbers in a list
        if 1 <= int(right) <= limit and int(left) <= int(right):
            return votes, "unreadable_vote_claim"
    return votes, None


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


def entry_votes(entry: dict) -> set[tuple[int, int]]:
    """Every count the data offers for one finding: its own vote, its regions', and its tasks'."""
    pairs = set()
    for block in [entry.get("agreement")] + [t.get("agreement") for t in entry.get("tasks") or []]:
        if not isinstance(block, dict):
            continue
        for vote in [block.get("presence")] + list((block.get("regions") or {}).values()):
            if isinstance(vote, dict) and isinstance(vote.get("votes"), int):
                pairs.add((vote["votes"], vote["out_of"]))
    return pairs


def _strings(value, minimum: int = 0, maximum: int | None = None) -> bool:
    return (isinstance(value, list) and all(isinstance(v, str) and v.strip() for v in value)
            and len(value) >= minimum and (maximum is None or len(value) <= maximum))


def verify_report(report: dict | None, structured: dict, parser=None) -> list[str]:
    """Problems with a reply, empty when it is a faithful report of the structured findings."""
    return verify_report_detailed(report, structured, parser)[0]


def verify_report_detailed(report: dict | None, structured: dict, parser=None,
                           *, context: str = "") -> tuple[list[str], list[dict]]:
    """(problems, the parser records behind them).

    Every structural check below is code and stays code: they count findings, compare statuses and
    categories, and read quoted vote fractions, all of which have one correct answer. A parser
    service adds two readings that code cannot do: a vote claim written in words rather than as a
    fraction, and the semantic fidelity check. A reading that stays unresolved is recorded as
    unresolved - it never invents a problem and never silences one the structural checks found.
    """
    if not isinstance(report, dict):
        return ["the reply is not a JSON object"], []
    problems, records = [], []
    for key in REPORT_KEYS:
        if key not in report:
            problems.append(f"missing key {key!r}")
    if problems:
        return problems, records
    if not isinstance(report["title"], str) or not report["title"].strip():
        problems.append("'title' must be a non-empty string")
    headings = report["headings"]
    if not isinstance(headings, dict) or any(not isinstance(headings.get(k), str) or not headings[k].strip()
                                             for k in ("image", "findings", "impression", "not_assessable", "limitations")):
        problems.append("'headings' must hold non-empty strings for image, findings, impression, not_assessable, limitations")
    expected = {f["finding"]: f for f in structured["findings"]}
    # With vote agreement on, a count the report quotes must be one the data holds: the model may not
    # claim 3/3 where two answers were readable, nor merge two regions' counts into one.
    agreement = structured["analysis"].get("vote_agreement") if "agreement" in structured["legend"] else None
    limit = max(1, int(agreement["wordings_per_task"])) if agreement else 0
    votes = {c: entry_votes(f) for c, f in expected.items()} if limit else {}
    seen: dict[str, int] = {}

    def quoted(text: str, where: str) -> set[tuple[int, int]] | None:
        """The counts one sentence quotes, or None when neither reader could read its vote claim."""
        if parser is None or not parser.enabled("vote_fraction"):
            return quoted_votes(text, limit)
        outcome = parser.vote_fraction(text, limit, code=lambda: quoted_votes_checked(text, limit),
                                       context=f"{context} | {where}".strip(" |"))
        records.append(outcome.record)
        return set(outcome.value or ()) if outcome.resolved else None
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
                elif limit:
                    quotes = quoted(entry["statement"], f"finding={finding}")
                    invented = (quotes - votes[finding]) if quotes is not None else set()
                    if invented:
                        problems.append(f"{finding}: the statement quotes vote counts the data does not hold: "
                                        + ", ".join(f"{a}/{b}" for a, b in sorted(invented))
                                        + "; quote only the counts in this finding's 'agreement' block")
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
    if limit:
        known = set().union(*votes.values()) if votes else set()
        for key in ("impression", "not_assessable", "limitations"):
            if not _strings(report[key]):
                continue
            read = [quoted(text, key) for text in report[key]]
            invented = set().union(*(q for q in read if q is not None)) - known if read else set()
            if invented:
                problems.append(f"{key}: quotes vote counts the data does not hold: "
                                + ", ".join(f"{a}/{b}" for a, b in sorted(invented)))
    # Meaning, once the structure holds: a report can pass every count above and still say something
    # the data does not support. The check only ever adds what it can point to; unresolved adds nothing.
    if parser is not None and parser.enabled("report_fidelity"):
        outcome = parser.report_fidelity(report, structured, context=context)
        records.append(outcome.record)
        if outcome.resolved and not outcome.value["faithful"]:
            problems.extend(outcome.value["problems"])
        elif not outcome.resolved:
            llm_api.monitor("REPORT FIDELITY UNRESOLVED", context or "report",
                            reason=outcome.error, policy="structural verdict stands")
    return problems, records


# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------
def agreement_note(entry: dict) -> str:
    """The vote counts behind one finding, as one line, regions kept apart.

    Rendered from the data, not from the model's sentence, so the evidence survives whatever the
    report model chose to say about it.
    """
    block = entry.get("agreement")
    if not isinstance(block, dict) or not block.get("measured"):
        return ""
    presence = block["presence"]
    if presence["decision"] == "no decision":
        head = (f"the readable answers tied, {presence['present_votes']} present to {presence['absent_votes']} absent, "
                f"so no decision was recorded")
    else:
        head = f"{presence['vote']} readable answers said {presence['decision']}"
    if presence["unreadable"]:
        head += f" ({presence['wordings_requested']} wordings asked, {presence['unreadable']} unreadable)"
    parts = [head]
    named = [f"{name} {vote['vote']}" for name, vote in (block.get("regions") or {}).items() if vote["votes"]]
    if named:
        parts.append("regions " + ", ".join(named))
    return "; ".join(parts)


def render_markdown(report: dict, structured: dict, writer_model: str | None = None) -> str:
    """Deterministic Markdown from a verified report: findings by section, impression, caveats."""
    headings = report["headings"]
    image, analysis = structured["image"], structured["analysis"]
    lines = [f"# {report['title']}", "", f"**{headings['image']}:** {image['file']}", "", f"## {headings['findings']}"]
    by_finding = {f["finding"]: f for f in structured["findings"]}
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
            note = agreement_note(by_finding.get(entry["finding"], {}))
            if note:
                lines.append(f"  - *agreement between wordings - {note}*")
    lines += ["", f"## {headings['impression']}"] + [f"- {b.strip()}" for b in report["impression"]]
    if report["not_assessable"]:
        lines += ["", f"## {headings['not_assessable']}"] + [f"- {t.strip()}" for t in report["not_assessable"]]
    lines += ["", f"## {headings['limitations']}"] + [f"- {t.strip()}" for t in report["limitations"]]
    footer = f"{analysis['analyzer']} · {analysis['questions_asked']} questions"
    if "agreement" in structured["legend"]:
        footer += " · vote counts are agreement between question wordings, not probability or certainty"
    if writer_model:
        footer += f" → {writer_model}"
    lines += ["", "---", f"*{footer}*", ""]
    return "\n".join(lines)


def fallback_markdown(result: dict, problems: list[str], counting: bool = True) -> str:
    """The deterministic summary, used when the report model's reply could not be verified."""
    lines = ["# Automatic summary (the report model's reply failed verification)", ""]
    lines += [f"- {p}" for p in problems]
    lines += ["", "```", dp.dentist_report(result, counting), "```", ""]
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
                 counting: bool = True, timeout: float = 600.0, request_options: dict | None = None,
                 api_call_retries: int = 2, call_log: str | None = None, client=None, parser=None) -> None:
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
            reply = self._ask(messages)
            report, extraction = read_report_json(reply["text"], self.parser, context=context)
            problems, checks = verify_report_detailed(report, structured, self.parser, context=context)
            if reply["truncated"] and problems:
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
            "language": self.language, "writer": self.public(), "analyzer": structured["analysis"]["analyzer"],
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
