"""Read model-generated text with a language model when the code parser cannot.

Every decision this project makes about a radiograph is made by *reading text a model
wrote*. The readers are strict regular expressions and strict JSON loaders: DentVLM's
"Yes" on line 1, the nine location descriptors matched verbatim, the location adapter's
box JSON, the report writer's report JSON, the vote fractions a report quotes. They are
exact, cheap and reproducible, and they fail in two ways:

* they reject a reply a person would understand at once ("Caries is evident in the lower
  left quadrant", a rationale that says "both lower posterior regions", JSON with a
  trailing comma), and
* they accept text whose meaning is not what the matched string suggests (a rationale
  that names a region only to rule it out, a report sentence that quotes a count the data
  never held).

This module adds a second reader: a text LLM whose only job is to say what an existing
reply means. It never sees a radiograph, never sees ground truth, and never decides
anything clinical - it reads one piece of text and reports what it says, or reports that
it cannot tell. Nothing it cannot resolve becomes a clinical value: an unreadable decision
stays unresolved, an unreadable location stays unresolved, and both readers failing leaves
exactly the unresolved state the strict reader would have left.

Modes
-----
Every eligible stage has its own mode:

* "code"          - the strict reader only; the LLM is never called (today's behaviour),
* "llm"           - the LLM only; the strict reader is not consulted,
* "code_then_llm" - the strict reader first, and the LLM only after it reports a real
                    failure (ambiguous or missing decision, invalid or incomplete JSON,
                    schema mismatch, missing entries, truncation, or another explicitly
                    represented unresolved state). A reader that succeeds never costs a
                    call.

One global mode overrides every stage when it is not None; None is manual mode, in which
each stage follows its own setting, whose default is the one documented in STAGES below.

What is deliberately NOT routed through a model: OpenAI response envelopes, YOLO/DENTEX
annotations, run manifests, response-cache artifacts, configuration dictionaries and
saved-artifact integrity checks. Those are machine formats with one correct reading; a
model could only make them less reliable.
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
from response_cache import ResponseCache

MODES = ("code", "llm", "code_then_llm")
PROMPT_VERSION = 1

# ----------------------------------------------------------------------------
# The eligible stages, their default manual-mode mode, and why it is that one
# ----------------------------------------------------------------------------
# "code_then_llm" is the default wherever the strict reader is trustworthy when it
# succeeds, so the LLM is a repair for an explicit failure and costs nothing otherwise.
# "llm" is the default wherever the strict reader can appear to succeed while losing
# meaning, so consulting it first would hide the very thing the LLM is there to catch.
STAGES = {
    "whole_image_decision": {
        "default_mode": "code_then_llm",
        "what": "the Yes/No decision in a whole-image task reply",
        "why": "line 1 of a DentVLM reply is a reliable decision when it can be read at all; the "
               "LLM is only needed when it is missing, ambiguous or cut off",
        "input": "the reply text and the question it answers",
    },
    "region_decision": {
        "default_mode": "code_then_llm",
        "what": "the Yes/No decision in a reply to a question that named one region",
        "why": "same reader, same reliability as the whole-image decision",
        "input": "the reply text and the question it answers",
    },
    "rationale_location": {
        "default_mode": "llm",
        "what": "the dental-arch regions a rationale names",
        "why": "the strict reader matches nine fixed descriptors verbatim, so a rationale that "
               "names regions in any other words looks like a rationale that named none - a "
               "silent loss the reader reports as success",
        "input": "the reply text, the six cell names and the nine descriptors",
    },
    "saved_answer_reconstruction": {
        "default_mode": "code_then_llm",
        "what": "the Yes/No answer of a saved region call that carries no parsed value",
        "why": "reconstruction re-reads a reply the strict reader already handled once; the LLM "
               "is only needed for the replies it could not read",
        "input": "the saved reply text and the question it answers",
    },
    "spotlight_decision": {
        "default_mode": "code_then_llm",
        "what": "the Yes/No decision in a spotlight-adapter reply",
        "why": "the same line-1 reader on the same model's replies",
        "input": "the reply text and the question it answers",
    },
    "spotlight_location": {
        "default_mode": "llm",
        "what": "the dental-arch region a spotlight-adapter reply places the box in",
        "why": "the same verbatim-descriptor loss as the rationale, and a box that is not placed "
               "falls back to fixed windows, so a missed descriptor silently changes the truth",
        "input": "the reply text, the six cell names and the nine descriptors",
    },
    "location_json": {
        "default_mode": "code_then_llm",
        "what": "the location adapter's box JSON: ids, anatomical units and tooth numbers",
        "why": "valid JSON that passes the schema check is exactly right; the LLM is the repair "
               "for invalid JSON, a missing box, a duplicate id, an unknown unit or a cut-off reply",
        "input": "the reply text, how many boxes were asked about, and the valid unit names",
    },
    "report_json": {
        "default_mode": "code_then_llm",
        "what": "the report object in the report writer's reply",
        "why": "a reply that loads as JSON is the report; the LLM is the repair for a fence, a "
               "stray sentence, a trailing comma or a cut-off object",
        "input": "the reply text and the report's top-level key names",
    },
    "report_fidelity": {
        "default_mode": "llm",
        "what": "whether the report's sentences say what the data says, beyond the structural checks",
        "why": "the structural check counts findings, statuses, categories and quoted fractions; "
               "a report can pass all of it and still upgrade a hedge to a diagnosis, move a "
               "finding to a region the data never named, or drop a limitation - none of which a "
               "code check can see",
        "input": "the report JSON and the structured findings it must reword (both model output, "
                 "never ground truth)",
    },
    "vote_fraction": {
        "default_mode": "code_then_llm",
        "what": "the vote counts a report sentence quotes",
        "why": "the strict reader reads '2/3' in Western and Eastern Arabic digits exactly; the "
               "LLM is needed when a sentence states a vote in words or in a numeral form the "
               "reader does not cover, which the reader reports as an unread vote claim",
        "input": "the sentence and how many wordings were asked",
    },
}
DEFAULT_MODES = {stage: spec["default_mode"] for stage, spec in STAGES.items()}
DECISION_STAGES = ("whole_image_decision", "region_decision", "saved_answer_reconstruction",
                   "spotlight_decision")
LOCATION_STAGES = ("rationale_location", "spotlight_location")


# ----------------------------------------------------------------------------
# Prompts: one per task, each naming its schema, its values and its unresolved case
# ----------------------------------------------------------------------------
# Every prompt ends the same way on purpose: JSON only, one fixed shape, and "unresolved"
# rather than a guess. The models used here are capable, so the prompts are explicit
# rather than terse; what they must never do is add information the text does not carry.
_NO_GUESSING = (
    "You are reading one piece of text. You never see an image, you are never told what is "
    "true of the patient, and you must never use what is likely or common in dental "
    "radiology to fill a gap. If the text does not settle the question, the honest answer "
    "is \"unresolved\", and \"unresolved\" is always better than a plausible guess."
)

DECISION_SYSTEM = (
    "You report what another model's reply to a single yes/no question says. You are a reader, not "
    "an examiner: you never look at a radiograph and you never form your own opinion about the "
    "patient. " + _NO_GUESSING + " You answer with JSON only, no prose before or after it."
)

DECISION_USER = """A model was asked one yes/no question about a dental panoramic radiograph and replied in words. Report which of the two answers its reply gives.

THE QUESTION IT WAS ASKED
{question}

ITS COMPLETE REPLY
<<<REPLY
{text}
REPLY>>>

RULES
1. Report only what the reply says. The reply is the whole of the evidence: you are not deciding whether the finding is there, only what this reply answered.
2. "yes" when the reply affirms the thing asked about (it says Yes, or states the finding is present, visible, observed, evident, noted, or describes where it is). "no" when the reply denies it (it says No, or states the finding is absent, not seen, not observed, none identified, unremarkable in that respect).
3. A reply may answer without the words yes or no. Read the meaning of the sentences, not the presence of a keyword.
4. Answer "unresolved" when: the reply gives both answers or contradicts itself; it hedges without settling ("possible", "cannot be excluded", "further evaluation needed") and never commits; it refuses or says it cannot tell; it answers a different question than the one above; it describes the image without ever answering; or it contains no decision at all.
5. A negated finding is still an answer: "There is no evidence of caries" is "no". A mention of the finding inside a denial is not a "yes".
6. TRUNCATION: {truncation}. A reply that stops mid-sentence still counts as an answer when the decision itself is complete and unambiguous before the cut. If the cut removes the decision, or leaves it ambiguous, answer "unresolved".
7. Never infer the answer from how the question is worded, from what would usually be found on a panoramic radiograph, or from other findings. Only this reply decides.

OUTPUT
JSON only, exactly this shape:
{"answer": "yes" | "no" | "unresolved", "evidence": "<the words of the reply that carry the answer, copied verbatim, or an empty string>", "reason": "<one short sentence; for unresolved, say what is missing or conflicting>"}"""

LOCATION_SYSTEM = (
    "You report which regions of the dental arch a model's own reply places a finding in. You are a "
    "reader, not an examiner: you never look at a radiograph, and you never place a finding yourself. "
    + _NO_GUESSING + " You answer with JSON only, no prose before or after it."
)

LOCATION_USER = """A model was asked about a finding on a dental panoramic radiograph and wrote a rationale. Report which regions of the dental arch ITS REPLY places the finding in.

{question_block}ITS COMPLETE REPLY
<<<REPLY
{text}
REPLY>>>

THE SIX REGIONS
Answer with these six identifiers and no others:
{cell_lines}

The model was trained to write its locations with these fixed phrases; they map onto the six identifiers like this:
{descriptor_lines}

RULES
1. "left" and "right" are the reply's own words and the identifiers above use the same convention. Copy the side the reply states; never flip it, never convert it to the patient's side, never reason about how a radiograph is displayed.
2. List every region the reply places THIS finding in, in any wording. A reply may name a region with the fixed phrases above, with a paraphrase ("the lower left back teeth", "the upper front region", "both posterior segments of the mandible"), with a quadrant name, with tooth numbers, or by describing the site in a sentence. Read the meaning and map it onto the identifiers.
3. Several regions are normal. A phrase naming two regions at once ("the right posterior region of both the upper and lower dentition") is two identifiers. "Both posterior regions of the lower dentition" is lower-right and lower-left. A finding stated in several places is every one of them. Never collapse several regions into one and never add a region the reply does not place the finding in.
4. Only regions where the reply places THIS finding count. A region the reply mentions in order to rule the finding out, or to describe some other structure or some other finding, is NOT a region for this finding.
5. Tooth numbers, when the reply gives them, use the FDI system: quadrant 1 = upper right, 2 = upper left, 3 = lower left, 4 = lower right (the reply's own left and right); positions 1-3 of a quadrant are its anterior region and positions 4-8 its posterior region. The upper-anterior identifier covers quadrants 1 and 2 anterior, and lower-anterior covers quadrants 3 and 4 anterior.
6. An empty list is a real answer and the right one when the reply states the finding but never says where. Do NOT use "unresolved" for that.
7. Use "unresolved": true only when the reply does place the finding somewhere but you cannot tell where - the wording is contradictory, the site is named in a way that does not map onto these six regions, or the text is cut off in the middle of the location.
8. TRUNCATION: {truncation}. Regions that are complete before the cut still count; if the cut interrupts a location, mark the answer unresolved.

OUTPUT
JSON only, exactly this shape:
{"regions": ["<identifier>", "..."], "unresolved": false, "evidence": "<the words of the reply that name the regions, copied verbatim, or an empty string>", "reason": "<one short sentence>"}"""

LOCATION_JSON_SYSTEM = (
    "You recover the JSON answer a model was asked to produce when its reply is not valid, not complete "
    "or not in the shape that was asked for. You only rearrange and repair what the reply already says. "
    + _NO_GUESSING + " You answer with JSON only, no prose before or after it."
)

LOCATION_JSON_USER = """A model was asked to classify {n_boxes} numbered bounding boxes on a dental panoramic radiograph into dental-arch units, and to answer with JSON. Its reply could not be read by a strict JSON reader. Recover the answer it actually gave.

THE STRICT READER'S COMPLAINT
{error}

ITS COMPLETE REPLY
<<<REPLY
{text}
REPLY>>>

WHAT THE ANSWER MUST LOOK LIKE
One entry per box, with the box numbers 1 to {n_boxes}. Each entry has "id" (the box number), "units" (a list of the dental-arch units the box occupies) and "teeth" (a list of FDI tooth numbers, possibly empty).

The only valid unit names are exactly:
{unit_names}

RULES
1. Recover, never invent. Every unit and every tooth number you return must be one this reply gives for that box. If the reply says nothing about a box, leave that box out of your answer entirely - do not guess a unit for it, and do not shift another box's answer onto it.
2. Map the reply's own words onto the valid unit names when it clearly means one of them (for example "upper right posterior" or "Q1 posterior" or "quadrant 1, molars" all mean Q1-posterior). Drop a unit you cannot map; never replace it with a neighbouring one.
3. An empty "units" list is a real answer: it means the reply placed that box outside the dental arches or explicitly could not place it. Keep it.
4. "teeth" holds whole numbers only. Drop anything that is not a tooth number; an empty list is fine.
5. Ids must be whole numbers between 1 and {n_boxes}, each at most once. If the reply gives the same id twice with different answers, leave that box out rather than choosing between them.
6. TRUNCATION: the reply may be cut off. Entries that are complete before the cut are recoverable; an entry that is cut off mid-way is not - leave it out.
7. If the reply carries no recoverable box answer at all, return an empty "boxes" list and set "unresolved" to true.

OUTPUT
JSON only, exactly this shape:
{"boxes": [{"id": 1, "units": ["Q1-posterior"], "teeth": [16, 17]}], "unresolved": false, "reason": "<one short sentence saying what you recovered and what you left out>"}"""

REPORT_JSON_SYSTEM = (
    "You recover the JSON document a model was asked to produce when its reply is not valid JSON or is "
    "wrapped in other text. You only recover what the reply already contains: you never write a value "
    "the reply does not have, and you never translate, shorten, correct or improve its text. "
    + _NO_GUESSING + " You answer with JSON only, no prose before or after it."
)

REPORT_JSON_USER = """A model was asked to return a dental report as a single JSON object. Its reply could not be read by a strict JSON reader. Recover the JSON object it gave.

ITS COMPLETE REPLY
<<<REPLY
{text}
REPLY>>>

THE OBJECT IT WAS ASKED FOR
A single JSON object with these top-level keys: {keys}. Their values are the report's own text, in whatever language the model wrote it.

RULES
1. Copy every value exactly as the reply wrote it, character for character, in its own language. Do not translate, rephrase, summarise, reorder, correct spelling, fix grammar or "improve" anything.
2. Repair only the JSON itself: a code fence, a sentence before or after the object, a trailing comma, a smart quote, an unescaped quote, a missing closing bracket whose content is nonetheless complete.
3. Never write a value the reply does not contain. If a key is missing, leave it out; do not invent a title, a heading, a statement, an impression bullet or a limitation.
4. TRUNCATION: if the reply is cut off, keep every element that is complete and drop the one that is cut off mid-way. If the cut leaves the object so incomplete that its findings cannot be recovered, return "unresolved": true and an empty object.
5. If the reply contains no recoverable JSON object at all, return "unresolved": true and an empty object. A missing report is a visible failure; an invented report is a dangerous one.

OUTPUT
JSON only, exactly this shape:
{"report": {"<the recovered object, or empty>": "..."}, "unresolved": false, "reason": "<one short sentence saying what you recovered and what you dropped>"}"""

REPORT_FIDELITY_SYSTEM = (
    "You check that a written dental report says what its source data says, and nothing more. You never "
    "see a radiograph and you never judge whether the data is clinically right: the data is the only "
    "truth here, and your single question is whether the report is faithful to it. " + _NO_GUESSING
    + " You answer with JSON only, no prose before or after it."
)

REPORT_FIDELITY_USER = """A report was written from structured findings by a model that never saw the radiograph. Its only job was to reword and organise those findings. Check that it did, and list where it did not.

THE DATA THE REPORT MUST REWORD
{structured_json}

THE REPORT
{report_json}

The report may be written in any language; the data is in English. Judge the meaning, not the words, and never report a difference that is only translation.

CHECK, AND REPORT A PROBLEM WHEN
1. A finding's sentence contradicts its status. "present", "absent", "unparseable" (the answer could not be read, so the finding is neither confirmed nor excluded) and "not_assessed" (never asked) must each be said as what they are. An unparseable finding stated as absent, or a not_assessed finding stated as absent, is a problem.
2. A finding is reported in a region the data does not place it in, or one of its regions is dropped, or the number of regions is changed, or a region the data marks "not_named" / "not_asked" is reported as free of the finding.
3. A finding sits in a section other than the category the data assigns it.
4. The report states a tooth number, a count of teeth, a size, a severity, a stage, a grade, a probability, a percentage or a certainty that the data does not contain, or turns a vote count into a confidence.
5. The report adds a diagnosis, a differential, a prognosis or treatment advice, or attributes a finding to a cause.
6. The report drops a limitation the data lists, or states the analysis saw the image, or presents the analysis as a clinician's reading.
7. The report quotes a vote count ("2/3") that is not in that finding's own agreement data, or merges two regions' counts, or reports a vote out of the wordings asked when fewer answers were readable.
8. The report claims a finding the data does not list, or leaves out one it does.

DO NOT report a problem for: wording, style, tone, ordering inside a section, a translation choice, a sentence that is shorter or longer than you would write, or anything you merely dislike. Do not re-diagnose the patient. Do not repeat a problem that is only a consequence of another one.

Every problem must name the finding identifier (or the report section) and quote the words that are wrong, so the problem can be checked without you.

If you cannot compare the two documents at all - the report is unreadable, or the data is missing - set "unresolved" to true instead of listing problems. An unresolved check is recorded as unresolved; it never becomes a failure and never becomes a pass.

OUTPUT
JSON only, exactly this shape:
{"faithful": true, "problems": ["<finding or section>: <what is wrong>, quoting the report's words"], "unresolved": false, "reason": "<one short sentence>"}"""

VOTE_FRACTION_SYSTEM = (
    "You report the vote counts a sentence quotes. You are a reader: you never compute a count, never "
    "correct one, and never decide whether a count is right. " + _NO_GUESSING
    + " You answer with JSON only, no prose before or after it."
)

VOTE_FRACTION_USER = """A sentence from a dental report may quote vote counts: how many rewordings of one question agreed, of how many that could be read. Report every count the sentence quotes.

THE SENTENCE
<<<TEXT
{text}
TEXT>>>

WHY THE STRICT READER STOPPED
{error}

RULES
1. A count is a pair of whole numbers, "how many" out of "out of". Report it as [how_many, out_of].
2. Read the count in whatever form the sentence gives it: "2/3", "2 of 3", "2 out of 3", "two of the three wordings", the same in any language, and in any numeral system (Western, Eastern Arabic, Persian, Devanagari, ...). Convert the numerals to ordinary integers; never change the numbers themselves.
3. "out_of" must be between 1 and {limit}; a pair whose "out_of" is outside that is not a vote count and is left out. Report "how_many" exactly as the sentence states it, even when it is larger than "out_of": a sentence that claims more agreeing answers than there were is a claim the checker must see, not one for you to correct.
4. Numbers that are not vote counts are left out entirely: tooth numbers, region names, dates, measurements, list positions, and any number that is not presented as "some of some".
5. An empty list is a real answer and the right one when the sentence quotes no vote count at all. Do NOT use "unresolved" for that.
6. Use "unresolved": true only when the sentence clearly claims a vote but you cannot recover its two numbers.

OUTPUT
JSON only, exactly this shape:
{"votes": [[2, 3]], "unresolved": false, "reason": "<one short sentence>"}"""

FORMAT_REMINDER = ("\n\nYour previous reply could not be read: {error}. Reply again with the JSON object "
                   "described above and nothing else: no explanation, no code fence, no text before or "
                   "after it. Keep the same shape and the same allowed values.")

PROMPTS = {
    "decision": {"system": DECISION_SYSTEM, "user": DECISION_USER},
    "location": {"system": LOCATION_SYSTEM, "user": LOCATION_USER},
    "location_json": {"system": LOCATION_JSON_SYSTEM, "user": LOCATION_JSON_USER},
    "report_json": {"system": REPORT_JSON_SYSTEM, "user": REPORT_JSON_USER},
    "report_fidelity": {"system": REPORT_FIDELITY_SYSTEM, "user": REPORT_FIDELITY_USER},
    "vote_fraction": {"system": VOTE_FRACTION_SYSTEM, "user": VOTE_FRACTION_USER},
    "format_reminder": FORMAT_REMINDER,
}

TRUNCATED_NOTE = "this reply was cut off by the model's output limit, so its end is missing"
COMPLETE_NOTE = "this reply is complete; it was not cut off"


def _fill(template: str, **values) -> str:
    """Replace {name} placeholders one at a time; str.format would choke on the JSON braces."""
    for key, value in values.items():
        template = template.replace("{" + key + "}", str(value))
    return template


def _cell_lines() -> str:
    return "\n".join(f"- {cell}: {dp.CELL_DESCRIPTORS[cell]}" for cell in dp.CELLS)


def _descriptor_lines() -> str:
    return "\n".join(f'- "{text}" -> {", ".join(cells)}' for text, cells in dp.DESCRIPTORS.items())


# ----------------------------------------------------------------------------
# Reading the parser's own reply (strict, by code: no second model behind this one)
# ----------------------------------------------------------------------------
def parser_json(text: str) -> dict | None:
    """The JSON object in the parser's reply, or None. A code fence or stray prose is tolerated."""
    text = (text or "").strip()
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


def _read_decision(text: str):
    payload = parser_json(text)
    if payload is None:
        return None, "parser_reply_not_json"
    answer = payload.get("answer")
    if answer == "unresolved":
        return None, "unresolved_decision"
    if answer in ("yes", "no"):
        return answer, None
    return None, "parser_reply_invalid_answer"


def _read_location(text: str):
    payload = parser_json(text)
    if payload is None:
        return None, "parser_reply_not_json"
    if payload.get("unresolved") is True:
        return None, "unresolved_location"
    regions = payload.get("regions")
    if not isinstance(regions, list) or any(not isinstance(cell, str) for cell in regions):
        return None, "parser_reply_invalid_regions"
    unknown = [cell for cell in regions if cell not in dp.CELLS]
    if unknown:
        return None, "parser_reply_unknown_region"
    return [cell for cell in dp.CELLS if cell in set(regions)], None


def _read_location_json(n_boxes: int):
    def read(text: str):
        payload = parser_json(text)
        if payload is None:
            return None, "parser_reply_not_json"
        entries = payload.get("boxes")
        if not isinstance(entries, list):
            return None, "parser_reply_invalid_boxes"
        parsed: dict[int, dict] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                return None, "parser_reply_invalid_box_entry"
            try:
                box_id = int(entry.get("id"))
            except (TypeError, ValueError):
                return None, "parser_reply_invalid_box_id"
            if not 1 <= box_id <= n_boxes or box_id in parsed:
                return None, "parser_reply_invalid_box_id"
            units, teeth = entry.get("units"), entry.get("teeth")
            if not isinstance(units, list) or any(u not in dp.UNITS for u in units):
                return None, "parser_reply_unknown_unit"
            if not isinstance(teeth, list) or any(not isinstance(t, (int, str)) or not str(t).strip().isdigit()
                                                  for t in teeth):
                return None, "parser_reply_invalid_tooth_number"
            parsed[box_id] = {"units": [u for u in dp.UNITS if u in set(units)], "teeth": [int(t) for t in teeth]}
        if payload.get("unresolved") is True and not parsed:
            return None, "unresolved_location_json"
        if not parsed:
            return None, "parser_reply_no_boxes"
        return parsed, None
    return read


def _read_report_json(text: str):
    payload = parser_json(text)
    if payload is None:
        return None, "parser_reply_not_json"
    if payload.get("unresolved") is True:
        return None, "unresolved_report_json"
    report = payload.get("report")
    if not isinstance(report, dict) or not report:
        return None, "parser_reply_no_report"
    return report, None


def _read_report_fidelity(text: str):
    payload = parser_json(text)
    if payload is None:
        return None, "parser_reply_not_json"
    if payload.get("unresolved") is True:
        return None, "unresolved_fidelity"
    problems = payload.get("problems")
    if problems is None:
        problems = []
    if not isinstance(problems, list) or any(not isinstance(p, str) for p in problems):
        return None, "parser_reply_invalid_problems"
    problems = [p.strip() for p in problems if p.strip()]
    faithful = payload.get("faithful")
    if not isinstance(faithful, bool):
        return None, "parser_reply_invalid_faithful"
    if faithful is bool(problems):  # a verdict that argues with itself is not a verdict
        return None, "parser_reply_contradictory_verdict"
    return {"faithful": faithful, "problems": problems}, None


def _read_vote_fraction(limit: int):
    def read(text: str):
        payload = parser_json(text)
        if payload is None:
            return None, "parser_reply_not_json"
        if payload.get("unresolved") is True:
            return None, "unresolved_vote_claim"
        votes = payload.get("votes")
        if votes is None:
            votes = []
        if not isinstance(votes, list):
            return None, "parser_reply_invalid_votes"
        pairs = set()
        for pair in votes:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                return None, "parser_reply_invalid_vote_pair"
            try:
                a, b = int(pair[0]), int(pair[1])
            except (TypeError, ValueError):
                return None, "parser_reply_invalid_vote_pair"
            if 1 <= b <= limit and a >= 0:  # the same range the strict reader accepts
                pairs.add((a, b))
        return pairs, None
    return read


# ----------------------------------------------------------------------------
# Policy: the global mode, the per-stage modes, and how they resolve
# ----------------------------------------------------------------------------
def validate_mode(mode, *, allow_none: bool = False, where: str = "parser mode") -> None:
    if mode is None and allow_none:
        return
    if mode not in MODES:
        allowed = list(MODES) + ([None] if allow_none else [])
        raise ValueError(f"{where} must be one of {allowed}, got {mode!r}")


class ParserPolicy:
    """Which reader each stage uses. A global mode that is not None overrides every stage."""

    def __init__(self, global_mode: str | None = None, modes: dict | None = None) -> None:
        validate_mode(global_mode, allow_none=True, where="parser_mode (the global setting)")
        self.global_mode = global_mode
        self.modes = dict(DEFAULT_MODES)
        for stage, mode in (modes or {}).items():
            if stage not in STAGES:
                raise ValueError(f"unknown parser stage {stage!r}; expected one of {sorted(STAGES)}")
            validate_mode(mode, where=f"parser mode for {stage!r}")
            self.modes[stage] = mode

    def selected(self, stage: str) -> str:
        if stage not in STAGES:
            raise ValueError(f"unknown parser stage {stage!r}; expected one of {sorted(STAGES)}")
        return self.modes[stage]

    def mode(self, stage: str) -> tuple[str, str]:
        """(what this stage was set to, what it actually runs as after the global override)."""
        selected = self.selected(stage)
        return selected, self.global_mode or selected

    def resolved(self) -> dict[str, str]:
        return {stage: self.mode(stage)[1] for stage in STAGES}

    def uses_llm(self) -> bool:
        return any(mode != "code" for mode in self.resolved().values())

    def llm_stages(self) -> list[str]:
        return [stage for stage, mode in self.resolved().items() if mode != "code"]

    def settings(self) -> dict:
        return {"global_mode": self.global_mode, "selected_modes": dict(self.modes),
                "resolved_modes": self.resolved(), "defaults": dict(DEFAULT_MODES)}

    def summary_lines(self) -> list[str]:
        head = ("manual mode: every stage follows its own setting" if self.global_mode is None
                else f"global mode {self.global_mode!r} overrides every stage")
        lines = [head]
        for stage in STAGES:
            selected, resolved = self.mode(stage)
            note = "" if selected == resolved else f" (set to {selected!r}, overridden)"
            default = "" if selected == DEFAULT_MODES[stage] else f" [default {DEFAULT_MODES[stage]!r}]"
            lines.append(f"  {stage:<28} {resolved}{note}{default}")
        return lines


# ----------------------------------------------------------------------------
# The parser model: its own role, its own retries, its own cache, its own counters
# ----------------------------------------------------------------------------
class ParserModel:
    """The parser role: a hosted text model that only ever reads text another model wrote.

    from_api() builds one from an llm_api spec exactly like the analyzer, the location adapter
    and the reporter. token_param "max_completion_tokens" and temperature None for OpenAI
    reasoning models; other request fields go through request_options. Its calls are counted
    under their own CallLog("parser"), never mixed with the other roles.
    """

    kind = "parser"
    OPTIONS = ("token_param", "temperature", "max_output_tokens", "request_options",
               "api_call_retries", "parse_retries")

    def __init__(self, base_url: str | None, api_key: str, model: str, token_param: str = "max_tokens",
                 max_output_tokens: int = 2048, temperature: float | None = 0.0, timeout: float = 600.0,
                 request_options: dict | None = None, api_call_retries: int = 2, parse_retries: int = 1,
                 response_cache: ResponseCache | None = None, call_log: str | None = None,
                 client=None) -> None:
        if token_param not in llm_api.TOKEN_PARAMS:
            raise ValueError(f"token_param must be one of {llm_api.TOKEN_PARAMS}")
        llm_api.validate_api_retries(api_call_retries)
        llm_api.validate_parse_retries(parse_retries)
        self.client = client if client is not None else llm_api.connect(base_url, api_key, timeout)
        self.base_url, self.model = base_url, model
        self.token_param, self.max_output_tokens, self.temperature = token_param, max_output_tokens, temperature
        self.request_options = dict(request_options or {})
        self.api_call_retries, self.parse_retries = api_call_retries, parse_retries
        self.response_cache = response_cache
        self.call_log = mon.CallLog("parser", call_log)

    @classmethod
    def from_api(cls, spec: dict, timeout: float = 600.0, response_cache: ResponseCache | None = None,
                 client=None) -> "ParserModel":
        base_url, api_key = llm_api.resolve(spec)
        options = {k: spec[k] for k in cls.OPTIONS if k in spec}
        return cls(base_url, api_key, spec["model"], timeout=timeout, response_cache=response_cache,
                   client=client, **options)

    @property
    def calls(self) -> int:
        return self.call_log.calls

    @property
    def requests(self) -> int:
        return self.call_log.requests

    @property
    def cache_hits(self) -> int:
        return self.call_log.cache_hits

    def settings(self) -> dict:
        """Everything that shapes a parser reply, including the prompts, so a prompt edit is a
        different configuration and a resumed run says so instead of mixing two readings."""
        return {"kind": self.kind, "model": self.model, "base_url": self.base_url,
                "token_param": self.token_param, "max_output_tokens": self.max_output_tokens,
                "temperature": self.temperature, "request_options": self.request_options,
                "api_call_retries": self.api_call_retries, "parse_retries": self.parse_retries,
                "prompt_version": PROMPT_VERSION, "prompts": PROMPTS}

    def public(self) -> dict:
        """The settings without the prompt texts, plus the cache, for printouts and summaries.

        The cache is deliberately not part of settings(): reusing an identical reply from disk is
        the same reading as asking for it again, so turning it on must not invalidate a resume.
        """
        return {k: v for k, v in self.settings().items() if k != "prompts"} | {
            "response_cache": self.response_cache is not None}

    def ask(self, system: str, user: str, *, context: str = "") -> dict:
        request = {"model": self.model,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                   **llm_api.generation_fields(self.token_param, self.max_output_tokens, self.temperature)}
        request.update(self.request_options)
        cache_key = self.response_cache.key(request) if self.response_cache is not None else None
        cached = self.response_cache.get(cache_key) if self.response_cache is not None else None
        if cached is not None:
            mon.tally("parser_cache_hit")
            return self.call_log.cached({**cached, "cache_hit": True, "cache_key": cache_key})
        started = time.perf_counter()
        normalized = llm_api.call_with_retries(
            lambda: llm_api.chat_reply(self.client.chat.completions.create(**request)),
            self.api_call_retries, f"parser model={self.model} | {context}".rstrip(" |"))
        result = {**normalized, "latency_seconds": round(time.perf_counter() - started, 3),
                  "cache_hit": False, "cache_key": cache_key}
        if self.response_cache is not None:
            self.response_cache.put(cache_key, result)
        return self.call_log.live(result)


# ----------------------------------------------------------------------------
# One parse: the outcome, its complete record, and the engine that produces both
# ----------------------------------------------------------------------------
class ParseOutcome:
    """The value a stage produced, whether it is resolved, and everything about how it got there."""

    __slots__ = ("stage", "value", "error", "record")

    def __init__(self, stage: str, value, error: str | None, record: dict) -> None:
        self.stage, self.value, self.error, self.record = stage, value, error, record

    @property
    def resolved(self) -> bool:
        return self.error is None

    @property
    def used_llm(self) -> bool:
        return bool(self.record.get("llm_used"))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<ParseOutcome {self.stage} value={self.value!r} error={self.error!r}>"


def _jsonable(value):
    """A record field that survives json.dumps: sets and tuples become sorted lists."""
    if isinstance(value, set):
        return sorted(list(v) if isinstance(v, tuple) else v for v in value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


class ParserService:
    """The two readers, the policy that chooses between them, and the record of every parse.

    One service is shared by the analyzer run, the location adapter and the report writer, so
    every parser call of a run is counted once, in one place, next to its own model.
    """

    def __init__(self, policy: ParserPolicy | None = None, model: ParserModel | None = None) -> None:
        self.policy = policy or ParserPolicy(global_mode="code")
        self.model = model
        needed = self.policy.llm_stages()
        if needed and model is None:
            raise ValueError("a parser model is required: these stages resolve to an LLM mode: "
                             + ", ".join(needed) + " (set parser_mode='code' to read with code only)")
        self.usage: dict[str, dict[str, int]] = {}

    def enabled(self, stage: str) -> bool:
        """True when this stage may call the parser model, i.e. its resolved mode is not "code"."""
        return self.policy.mode(stage)[1] != "code"

    # -- bookkeeping ---------------------------------------------------------
    def _count(self, stage: str, record: dict) -> None:
        row = self.usage.setdefault(stage, {"parses": 0, "code_ok": 0, "code_failed": 0, "llm_calls": 0,
                                            "llm_ok": 0, "fallbacks": 0, "cache_hits": 0, "retries": 0,
                                            "unresolved": 0})
        row["parses"] += 1
        row["code_ok"] += bool(record.get("code_ok"))
        row["code_failed"] += record.get("code_attempted", False) and not record.get("code_ok")
        row["llm_calls"] += len(record.get("parser_attempts") or [])
        row["llm_ok"] += bool(record.get("llm_used"))
        row["fallbacks"] += bool(record.get("fallback_reason"))
        row["cache_hits"] += sum(bool(a.get("cache_hit")) for a in record.get("parser_attempts") or [])
        row["retries"] += record.get("retries", 0)
        row["unresolved"] += record.get("error") is not None

    def usage_snapshot(self) -> dict:
        return {stage: dict(row) for stage, row in self.usage.items()}

    def usage_since(self, snapshot: dict) -> dict:
        """What this service has parsed since `snapshot`, so one image's record holds only its own."""
        delta = {}
        for stage, row in self.usage.items():
            before = snapshot.get(stage, {})
            changed = {k: v - before.get(k, 0) for k, v in row.items() if v - before.get(k, 0)}
            if changed:
                delta[stage] = changed
        return delta

    # -- configuration -------------------------------------------------------
    def settings(self) -> dict:
        """What every manifest records, so a resumed run cannot read saved text a second way."""
        return {"policy": self.policy.settings(), "model": self.model.settings() if self.model else None,
                "stages": {stage: {k: spec[k] for k in ("default_mode", "what", "why", "input")}
                           for stage, spec in STAGES.items()}}

    def public(self) -> dict:
        """The settings without the prompt texts, for configuration summaries and printouts."""
        return {"policy": self.policy.settings(), "model": self.model.public() if self.model else None}

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.settings(), sort_keys=True, default=str).encode()).hexdigest()[:16]

    def summary_lines(self) -> list[str]:
        model = self.model.public() if self.model else None
        head = (f"parser model: {model['model']} (max_output_tokens={model['max_output_tokens']}, "
                f"temperature={model['temperature']}, parse_retries={model['parse_retries']}, "
                f"api_call_retries={model['api_call_retries']}, cache={model['response_cache']})"
                if model else "parser model: none (every stage reads with code)")
        return [head] + self.policy.summary_lines()

    # -- the engine ----------------------------------------------------------
    def _run(self, stage: str, *, code, prompt, read, original_text, context: str) -> ParseOutcome:
        selected, resolved_mode = self.policy.mode(stage)
        # The text is copied into the record whenever the parser model saw it, so that record can be
        # checked on its own next to the input and the reply it produced. A stage that only ran the
        # strict reader leaves it out: the text it read is the call the record is stored on, and
        # copying every reply a second time would double every artifact for nothing.
        record = {
            "stage": stage, "selected_mode": selected, "resolved_mode": resolved_mode,
            "global_mode": self.policy.global_mode,
            "original_text": None if resolved_mode == "code" else original_text,
            "code_attempted": False, "code_ok": None, "code_error": None, "code_value": None,
            "llm_attempted": False, "llm_used": False, "fallback_reason": None,
            "parser_input": None, "parser_response": None, "parser_attempts": [],
            "model": self.model.model if self.model else None, "latency_seconds": None,
            "prompt_tokens": None, "completion_tokens": None, "retries": 0,
            "cache_hit": None, "cache_key": None,
            "value": None, "error": None, "failure_reason": None,
        }
        code_value, code_error = None, None
        if resolved_mode in ("code", "code_then_llm"):
            record["code_attempted"] = True
            code_value, code_error = code()
            record["code_ok"] = code_error is None
            record["code_error"] = code_error
            record["code_value"] = _jsonable(code_value)
            if code_error is None or resolved_mode == "code":
                return self._finish(record, code_value, code_error)
            # A real failure of the strict reader, and only that, pays for a model call.
            record["fallback_reason"] = code_error
            mon.tally("parser_fallback")
            mon.monitor("PARSER FALLBACK", f"{stage} | {context}".rstrip(" |"), reason=code_error,
                        action="reading with the parser model")

        record["llm_attempted"] = True
        system, user = prompt(code_error)
        record["parser_input"] = {"system": system, "user": user}
        message, value, error = user, None, "parser_not_called"
        for attempt in range(self.model.parse_retries + 1):
            reply = self.model.ask(system, message, context=f"{stage} | {context}".rstrip(" |"))
            mon.tally("parser_llm_call")
            value, error = read(reply.get("text", ""))
            if error:
                if not (reply.get("text") or "").strip():
                    error = "empty_parser_response"
                elif reply.get("truncated") and error.startswith("parser_reply"):
                    error = "truncated_parser_output"
            record["parser_attempts"].append({
                "attempt": attempt + 1, "max_attempts": self.model.parse_retries + 1,
                "text": reply.get("text", ""), "finish_reason": reply.get("finish_reason"),
                "truncated": reply.get("truncated"), "prompt_tokens": reply.get("prompt_tokens"),
                "completion_tokens": reply.get("completion_tokens"),
                "latency_seconds": reply.get("latency_seconds"), "cache_hit": reply.get("cache_hit"),
                "cache_key": reply.get("cache_key"), "error": error,
            })
            record["parser_response"] = reply.get("text", "")
            record["latency_seconds"] = reply.get("latency_seconds")
            record["prompt_tokens"] = reply.get("prompt_tokens")
            record["completion_tokens"] = reply.get("completion_tokens")
            record["cache_hit"] = reply.get("cache_hit")
            record["cache_key"] = reply.get("cache_key")
            if not error:
                if attempt:
                    mon.tally("parser_recovered")
                    mon.monitor("PARSER RECOVERED", f"{stage} | {context}".rstrip(" |"),
                                attempt=f"{attempt + 1}/{self.model.parse_retries + 1}")
                break
            if error.startswith("unresolved"):
                break  # the parser answered, and its answer is "I cannot tell": retrying asks nothing new
            mon.tally("parser_warning")
            mon.monitor("PARSER WARNING", f"{stage} | {context}".rstrip(" |"),
                        attempt=f"{attempt + 1}/{self.model.parse_retries + 1}", reason=error,
                        finish=reply.get("finish_reason"))
            llm_api.failure_details("SYSTEM:\n" + system + "\n\nUSER:\n" + message, reply.get("text", ""))
            if attempt < self.model.parse_retries:
                record["retries"] += 1
                mon.tally("parser_retry")
                mon.monitor("PARSER RETRY", f"{stage} | {context}".rstrip(" |"), action="format reminder")
                message = user + _fill(FORMAT_REMINDER, error=error)
        if not error:
            record["llm_used"] = True
            return self._finish(record, value, None)
        # Both readers failed. Keep whatever the strict reader recovered (it may be partially
        # usable) and stay unresolved; nothing here may become a clinical value.
        mon.tally("parser_unresolved")
        mon.monitor("PARSER UNRESOLVED", f"{stage} | {context}".rstrip(" |"),
                    code=code_error, parser=error, policy="unresolved")
        return self._finish(record, code_value, code_error or error,
                            failure_reason=f"code:{code_error or 'not attempted'} | parser:{error}")

    def _finish(self, record: dict, value, error: str | None, failure_reason: str | None = None):
        record["value"] = _jsonable(value)
        record["error"] = error
        record["failure_reason"] = failure_reason if failure_reason is not None else error
        self._count(record["stage"], record)
        return ParseOutcome(record["stage"], value, error, record)

    # -- one method per stage ------------------------------------------------
    def decision(self, stage: str, text: str, question: str, *, truncated: bool = False,
                 context: str = "") -> ParseOutcome:
        """'yes' / 'no' / unresolved from one reply to one yes/no question."""
        if stage not in DECISION_STAGES:
            raise ValueError(f"{stage!r} is not a decision stage; expected one of {DECISION_STAGES}")

        def code():
            # A cut-off reply can hold an incomplete rationale even when line 1 reads, so the strict
            # reader refuses it; that refusal is what buys the model call.
            value = None if truncated else dp.extract_answer(text)
            return value, None if value is not None else ("truncated_output" if truncated
                                                          else "missing_or_ambiguous_decision")

        def prompt(_code_error):
            return DECISION_SYSTEM, _fill(DECISION_USER, question=question, text=text,
                                          truncation=TRUNCATED_NOTE if truncated else COMPLETE_NOTE)

        return self._run(stage, code=code, prompt=prompt, read=_read_decision,
                         original_text=text, context=context)

    def location(self, stage: str, text: str, *, question: str | None = None, truncated: bool = False,
                 expects_location: bool = True, context: str = "") -> ParseOutcome:
        """The cells a reply places the finding in; [] when it names none, unresolved when unreadable.

        `expects_location` is True when the reply reported the finding, so a reader that finds no
        descriptor has hit the explicit "missing_location" state rather than read a real "nowhere".
        """
        if stage not in LOCATION_STAGES:
            raise ValueError(f"{stage!r} is not a location stage; expected one of {LOCATION_STAGES}")
        repairable = self.policy.mode(stage)[1] == "code_then_llm"

        def code():
            # The strict reader matches nine phrases verbatim, so "no phrase" and "no location" look
            # the same to it. On its own it therefore has no failure at all: silence is its answer,
            # exactly as it always has been. Only when a model stands behind it does that silence
            # become the two states worth repairing - a reply that reported the finding without a
            # readable location, and a reply cut off inside one.
            cells = dp.extract_regions(text)
            if cells or not repairable:
                return cells, None
            if truncated:
                return [], "truncated_output"
            if expects_location:
                return [], "missing_location"
            return [], None

        def prompt(_code_error):
            block = f"THE QUESTION IT WAS ASKED\n{question}\n\n" if question else ""
            return LOCATION_SYSTEM, _fill(LOCATION_USER, question_block=block, text=text,
                                          cell_lines=_cell_lines(), descriptor_lines=_descriptor_lines(),
                                          truncation=TRUNCATED_NOTE if truncated else COMPLETE_NOTE)

        return self._run(stage, code=code, prompt=prompt, read=_read_location,
                         original_text=text, context=context)

    def location_json(self, text: str, n_boxes: int, *, code, truncated: bool = False,
                      context: str = "") -> ParseOutcome:
        """{box id: {"units", "teeth"}} from the location adapter's reply.

        `code` is the adapter's own strict reader as a callable returning (parsed, error), so the
        adapter keeps owning its schema check and the reader runs only when the mode asks for it.
        The model is given the reply, how many boxes were asked about and the valid unit names -
        never the ground-truth labels, conditions or coordinates of those boxes.
        """
        def prompt(code_error):
            return LOCATION_JSON_SYSTEM, _fill(
                LOCATION_JSON_USER, n_boxes=n_boxes, text=text,
                error=code_error or "the reply was not in the shape that was asked for",
                unit_names="\n".join(f"- {unit}" for unit in dp.UNITS))

        return self._run("location_json", code=code, prompt=prompt, read=_read_location_json(n_boxes),
                         original_text=text, context=context)

    def report_json(self, text: str, *, code, keys=(), context: str = "") -> ParseOutcome:
        """The report object in the report writer's reply. `code` is the strict reader, as a callable."""
        def prompt(_code_error):
            return REPORT_JSON_SYSTEM, _fill(REPORT_JSON_USER, text=text, keys=", ".join(keys))

        return self._run("report_json", code=code, prompt=prompt, read=_read_report_json,
                         original_text=text, context=context)

    def report_fidelity(self, report: dict, structured: dict, *, context: str = "") -> ParseOutcome:
        """Semantic check of a report against the findings it rewords, beyond the structural checks.

        Unresolved means unresolved: it never becomes a verification failure, and it never
        becomes a pass. The caller records it and the structural verdict stands.
        """
        def code():
            # There is no code reader for meaning: the structural checks are the caller's and run
            # either way, so in "code" mode this stage simply has nothing of its own to add.
            return {"faithful": None, "problems": []}, "no_code_reader_for_semantic_fidelity"

        def prompt(_code_error):
            return REPORT_FIDELITY_SYSTEM, _fill(
                REPORT_FIDELITY_USER,
                structured_json=json.dumps(structured, indent=1, ensure_ascii=False),
                report_json=json.dumps(report, indent=1, ensure_ascii=False))

        return self._run("report_fidelity", code=code, prompt=prompt, read=_read_report_fidelity,
                         original_text=json.dumps(report, ensure_ascii=False), context=context)

    def vote_fraction(self, text: str, limit: int, *, code, context: str = "") -> ParseOutcome:
        """The vote counts one report sentence quotes, as a set of (votes, out_of) pairs."""
        def prompt(code_error):
            return VOTE_FRACTION_SYSTEM, _fill(
                VOTE_FRACTION_USER, text=text, limit=limit,
                error=code_error or "a vote claim could not be read as two numbers")

        return self._run("vote_fraction", code=code, prompt=prompt, read=_read_vote_fraction(limit),
                         original_text=text, context=context)


# ----------------------------------------------------------------------------
# Building a service from a resolved configuration
# ----------------------------------------------------------------------------
def response_cache(root: str | Path, model_settings: dict) -> ResponseCache:
    """One shared exact-response cache for parser replies, namespaced by the parser model."""
    namespace = {"role": "parser", "prompt_version": PROMPT_VERSION,
                 **{k: model_settings[k] for k in ("model", "base_url", "token_param",
                                                   "max_output_tokens", "temperature", "request_options")}}
    return ResponseCache(Path(root) / "_parser_cache", namespace)


def build(spec: dict | None, global_mode: str | None, modes: dict | None = None, *,
          timeout: float = 600.0, cache_root: str | Path | None = None, client=None) -> ParserService:
    """The parser service of one experiment: the policy, and the model only when a stage needs it."""
    policy = ParserPolicy(global_mode, modes)
    if not policy.uses_llm():
        return ParserService(policy, None)
    if not isinstance(spec, dict) or not spec.get("model"):
        raise ValueError("these parser stages need a parser model: " + ", ".join(policy.llm_stages())
                         + "; give a 'parser' spec with a 'model' (see llm_api) or set parser_mode='code'")
    model = ParserModel.from_api(spec, timeout=timeout, client=client)
    if cache_root:  # built from the resolved model, so two providers of one model name never share
        model.response_cache = response_cache(cache_root, model.settings())
    return ParserService(policy, model)


def code_only() -> ParserService:
    """A service that reads with code alone: what every stage does when no parser is configured."""
    return ParserService(ParserPolicy(global_mode="code"), None)


def usage_rows(usage: dict) -> list[dict]:
    """One row per stage for the diagnostics tables; empty when nothing was parsed."""
    return [{"stage": stage, **{k: row.get(k, 0) for k in
                                ("parses", "code_ok", "code_failed", "llm_calls", "llm_ok", "fallbacks",
                                 "cache_hits", "retries", "unresolved")}}
            for stage, row in sorted((usage or {}).items())]
