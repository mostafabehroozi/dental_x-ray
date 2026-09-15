"""In-distribution DentalGPT pipeline: presence, tooth counts, and region location.

Every text sent to the model is one of the shapes with evidence in the DentalGPT
paper (arXiv 2512.11558) or in the panoramic benchmark it was scored on
(MMOral-OPG-Bench, arXiv 2509.09254):

* Presence: the Panorama-Classification question of Figure 7, one condition per
  call, answered A (True) or B (False). This is the only panoramic skill the
  paper measures (84% accuracy).
* Count: the Figure 1/9 filling-count question, and the same tooth-anchored
  "How many ..." shape for the other countable findings. The model counts
  teeth, so findings whose boxes are regions or devices are presence-only.
* Region: the same two questions restricted to one region of the mouth. The
  regions are the two jaws (the benchmark asks "in which jaw" and even counts
  "in the lower jaw") and the four FDI quadrants (the model walks them by name
  in its own reasoning, Figure 9). The region is either named in the question
  on the whole image ("words") or implied by sending a crop ("crop").

Two levels are set independently (Protocol): where presence is resolved (the
whole image only, or every region for every finding, with the whole-image
answers kept as a separate result) and where counts are taken (one whole-image
count, or one count per region). The whole-image answers never decide which
regional questions are asked, so a finding missed with the model's attention
spread over the whole image can be recovered in a region. Counting can also be
switched off altogether (Protocol.counting False): the model then only decides
presence, on the whole image and in every region, and no count question is
asked. Nothing else (JSON contracts or forced zeros) is used. Optional parse
retries append a format reminder.

A hosted vision-language model (Protocol.question_form "combined") is asked presence
and count in one question for the nine countable findings: the Figure 7 sentence and
the Figure 9-shaped count sentence stay verbatim inside a longer prompt that defines
the finding, fixes the scope and the display convention, and asks for two fixed reply
lines ("Answer: A. True" / "Count: 3", or "Answer: B. False" / "Count: 0"). One reply
then yields the presence and the count of one scope, so the separate count calls go
away while the results keep the same fields. DentalGPT keeps the separate questions.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import mimetypes
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import llm_api
import run_monitor as mon
from response_cache import ResponseCache

# Stable ontology in YOLO class order of the UMFIH 14-class dataset.
CONDITIONS = (
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
)

# Condition names substituted into the presence question. Paper vocabulary is
# used where the paper has the category (Root canal treatment, Periodontal
# disease, Impacted tooth, Periapical lesion, Dental caries); MMOral/PMC
# vocabulary elsewhere. Edit here only.
LABELS = {
    "dental_implant": "Dental implant",
    "prosthetic_restoration": "Dental crown or bridge",
    "dental_filling": "Dental filling",
    "endodontic_treatment": "Root canal treatment",
    "carious_lesion": "Dental caries",
    "periodontal_bone_loss": "Periodontal disease",
    "impacted_tooth": "Impacted tooth",
    "periapical_lesion": "Periapical lesion",
    "root_fragment": "Residual root",
    "furcation_lesion": "Furcation involvement",
    "apical_surgery": "Apical surgery",
    "root_resorption": "Root resorption",
    "orthodontic_device": "Orthodontic appliance",
    "surgical_device": "Surgical fixation plate or screws",
}

# Figure 7 (Panorama-Classification) wording, verbatim except the label.
PRESENCE_QUESTION = "Kindly evaluate if the condition '{label}' is present in this image.\nA. True\nB. False"
# The same wording with the region named; used when Protocol.region_prompt is "words".
REGION_PRESENCE_QUESTION = "Kindly evaluate if the condition '{label}' is present in {region} of this image.\nA. True\nB. False"

# Tooth-anchored count questions as (template, whole-image scope). The whole-image wording is
# the template with its scope filled in (the filling question is Figure 1/9 verbatim); a region
# replaces the scope ("How many teeth in the upper right quadrant have ...").
COUNT_TEMPLATES = {
    "dental_implant": ("How many dental implants are visualized in {scope}?", "the panoramic radiograph"),
    "prosthetic_restoration": ("How many teeth in {scope} have a dental crown or bridge?", "the image"),
    "dental_filling": ("How many visible teeth in {scope} appear to have dental fillings based on their radiopaque characteristics?", "the image"),
    "endodontic_treatment": ("How many teeth in {scope} have root canal treatment?", "the image"),
    "carious_lesion": ("How many teeth in {scope} are suspected to have caries?", "the image"),
    "impacted_tooth": ("How many impacted teeth are visualized in {scope}?", "the panoramic radiograph"),
    "periapical_lesion": ("How many teeth in {scope} show signs of a periapical lesion?", "the image"),
    "root_fragment": ("How many residual roots are visualized in {scope}?", "the panoramic radiograph"),
    "root_resorption": ("How many teeth in {scope} show root resorption?", "the image"),
}
COUNT_QUESTIONS = {c: template.format(scope=scope) for c, (template, scope) in COUNT_TEMPLATES.items()}  # whole image
COUNTABLE = tuple(c for c in CONDITIONS if c in COUNT_TEMPLATES)  # the other five are presence-only

# ----------------------------------------------------------------------------
# Combined presence-and-count question (Protocol.question_form "combined", hosted models)
# ----------------------------------------------------------------------------
# A capable vision-language model can decide whether a finding is present and count the affected
# teeth in one reply, so each countable finding is asked one question per scope instead of a presence
# question followed by a count question. The Figure 7 sentence and the Figure 9-shaped count sentence
# are kept verbatim inside a longer prompt that defines the finding, fixes the scope and the display
# convention, and ends in two fixed reply lines the pipeline reads. The five presence-only findings
# keep the bare Figure 7 question. DentalGPT keeps the separate questions ("separate"): the combined
# prompt is outside its training distribution.
QUESTION_FORMS = ("separate", "combined")

# One-line radiographic definition and counting rule per countable finding. The counting unit is the
# one the count sentence names (teeth, implants, residual roots). Edit here only.
DEFINITIONS = {
    "dental_implant": (
        "A dental implant is a screw-shaped, uniformly radiopaque endosseous fixture placed in the alveolar "
        "bone, with or without an abutment or a crown on it.",
        "Count each implant fixture once; an implant-supported bridge counts one per fixture, not per pontic. "
        "A root canal filling, a post and a fixation screw are not implants."),
    "prosthetic_restoration": (
        "A dental crown or bridge is a full-coverage artificial crown (metal, ceramic or metal-ceramic, "
        "radiopaque with a smooth outline that follows the shape of the crown) on a natural tooth or an "
        "implant, or a fixed bridge of such crowns joined by pontics.",
        "Count each tooth that carries a crown or serves as a bridge abutment once; pontics replace missing "
        "teeth and are not counted. An intracoronal filling is not a crown."),
    "dental_filling": (
        "A dental filling is intracoronal restorative material (radiopaque amalgam or composite, or an inlay) "
        "inside the crown of a tooth.",
        "Count each tooth with at least one filling once, however many fillings it carries. A full-coverage "
        "crown and root canal filling material are not dental fillings."),
    "endodontic_treatment": (
        "Root canal treatment shows as radiopaque obturation material (gutta-percha, sealer, a post) filling "
        "one or more root canals of a tooth.",
        "Count each root-canal-treated tooth once, however many of its canals are filled."),
    "carious_lesion": (
        "Dental caries shows as a radiolucent (dark) defect in the enamel or dentine of a tooth crown or "
        "root, including recurrent caries at the margin of a restoration.",
        "Count each tooth with at least one carious lesion once. A radiolucent composite restoration, "
        "cervical burn-out, the pulp chamber and an extraction socket are not caries."),
    "impacted_tooth": (
        "An impacted tooth has failed to erupt into its position in the arch and remains fully or partly in "
        "bone or soft tissue, typically a third molar or a canine, often tilted or blocked by a neighbouring "
        "tooth.",
        "Count each impacted tooth once. A normally developing, not yet erupted tooth of a growing patient "
        "is not impacted."),
    "periapical_lesion": (
        "A periapical lesion is a radiolucency centred on the apex of a root (periapical granuloma, cyst or "
        "abscess), usually with loss of the lamina dura around the apex.",
        "Count each tooth with a periapical lesion once, however many of its roots are involved. The mental "
        "foramen, the incisive canal, the maxillary sinus and marrow spaces are not lesions."),
    "root_fragment": (
        "A residual root is a retained root or root remnant whose crown has been lost or removed, left in the "
        "alveolar bone, usually in an otherwise edentulous space.",
        "Count each residual root once. A tooth with an intact crown, and a root-canal-treated tooth under a "
        "crown, are not residual roots."),
    "root_resorption": (
        "Root resorption is loss of root structure of a permanent tooth: shortening or blunting of the apex "
        "or an irregular root outline (external resorption), or a widening of the canal (internal "
        "resorption), often next to an impacted tooth or after orthodontic treatment.",
        "Count each tooth with resorption once. The physiological resorption of the roots of primary "
        "(deciduous) teeth is not counted."),
}

# The radiograph and its display convention, stated once so that "right" and "left" in a question are
# the patient's sides. The combined form does not use QUADRANT_WORDS_ARE_PATIENT_SIDE: its quadrant words
# are always the patient's, and the scope note names the image half as well.
COMBINED_CONTEXT = ("The image is a panoramic dental radiograph (orthopantomogram) in the standard display "
                    "orientation: the patient's right side is on the left of the image and the patient's left side "
                    "on the right; the maxilla (upper jaw) is at the top and the mandible (lower jaw) at the bottom.")

# What the model may look at, per window (words) or for the whole image and a crop.
SCOPE_NOTES = {
    "UR": "the patient's upper right quadrant (FDI quadrant 1): the upper teeth on the LEFT half of the image as "
          "displayed, from the upper right central incisor back to the upper right third molar",
    "UL": "the patient's upper left quadrant (FDI quadrant 2): the upper teeth on the RIGHT half of the image as "
          "displayed, from the upper left central incisor back to the upper left third molar",
    "LL": "the patient's lower left quadrant (FDI quadrant 3): the lower teeth on the RIGHT half of the image as "
          "displayed, from the lower left central incisor back to the lower left third molar",
    "LR": "the patient's lower right quadrant (FDI quadrant 4): the lower teeth on the LEFT half of the image as "
          "displayed, from the lower right central incisor back to the lower right third molar",
    "upper": "the upper jaw (maxilla): the upper teeth and the maxillary bone across the whole width of the image",
    "lower": "the lower jaw (mandible): the lower teeth and the mandibular bone across the whole width of the image",
}
WHOLE_SCOPE = "the whole radiograph."
CROP_SCOPE = "this image, which is a cropped region of the radiograph; consider only the teeth visible in it."

# The template. {presence_question} is the Figure 7 question (whole image, or with the region named) and
# {count_question} the Figure 9-shaped count sentence with the same scope, both verbatim; the reply ends
# in the two lines extract_combined reads.
COMBINED_QUESTION = """{context}

Finding under review: {label}. {definition}

Scope: {scope}

{presence_question}

If the answer is A, also answer: {count_question} {counting_rule} If the answer is B, the count is 0.

Decide from what is visible in this image, not from what is usual for a patient. An answer of A requires a count of at least 1, and a count of 0 requires the answer B.

You may reason briefly first, then end your reply with exactly these two lines and nothing after them:
Answer: <exactly "A. True" or "B. False">
Count: <a whole number written in digits; 0 when the answer is B>"""

# The paper says a fixed sentence was appended during RL to request <think> and
# <answer> tags but does not publish it. This is the common VLM-R1 wording and is
# a reconstruction; probe() decides whether it is needed at all.
THINK_SUFFIX = "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
MODES = ("plain", "tagged")

# Region windows as normalized (left, top, right, bottom). Quadrants use patient-side
# names in FDI order (UR, UL, LL, LR); image left is the patient's right. Windows
# overlap by 10% of the width and 20% of the height so a finding on the midline or
# the occlusal plane is whole in at least one crop; location truth is scored
# against these same windows.
CROPS = {
    "quadrant": {
        "UR": (0.00, 0.00, 0.55, 0.60),
        "UL": (0.45, 0.00, 1.00, 0.60),
        "LL": (0.45, 0.40, 1.00, 1.00),
        "LR": (0.00, 0.40, 0.55, 1.00),
    },
    "arch": {
        "upper": (0.00, 0.00, 1.00, 0.60),
        "lower": (0.00, 0.40, 1.00, 1.00),
    },
}
REGION_SCHEMES = tuple(CROPS)

# The regions as dental text, used when the region is named in the question. Quadrants carry
# the patient's sides, as dentists and DentalGPT itself (Figure 9) name them, so "the upper
# right quadrant" is the image-left window UR. QUADRANT_WORDS_ARE_PATIENT_SIDE records that the
# model reads the words that way; the DENTEX side check (dental_eval.side_agreement) confirms
# it, and setting it False attaches the mirrored words to the windows instead.
REGION_PHRASES = {
    "quadrant": {
        "UR": "the upper right quadrant",
        "UL": "the upper left quadrant",
        "LL": "the lower left quadrant",
        "LR": "the lower right quadrant",
    },
    "arch": {
        "upper": "the upper jaw",
        "lower": "the lower jaw",
    },
}
QUADRANT_WORDS_ARE_PATIENT_SIDE = True
_MIRROR = {"UR": "UL", "UL": "UR", "LL": "LR", "LR": "LL"}

# Dental-arch units: FDI quadrant x {anterior, posterior}, the finest division the quadrant and
# arch windows are made of (anterior = incisors and canine, positions 1-3). Ground-truth boxes
# translated into units (location_adapter) map onto the window names deterministically; the same
# unit output also serves a six-cell vocabulary (DentVLM branch). Quadrant names are patient-side
# in FDI order; UR and LR are the image-left windows.
UNITS = ("Q1-posterior", "Q1-anterior", "Q2-anterior", "Q2-posterior",
         "Q3-posterior", "Q3-anterior", "Q4-anterior", "Q4-posterior")
UNIT_QUADRANT = {"Q1": "UR", "Q2": "UL", "Q3": "LL", "Q4": "LR"}
QUADRANT_ARCH = {"UR": "upper", "UL": "upper", "LL": "lower", "LR": "lower"}


def fdi_unit(quadrant: int, tooth: int) -> str:
    """Unit of an FDI tooth position; primary-dentition quadrants 5-8 fold onto 1-4."""
    quadrant = quadrant - 4 if quadrant > 4 else quadrant
    return f"Q{quadrant}-{'anterior' if tooth <= 3 else 'posterior'}"


def unit_region(unit: str, level: str = "quadrant") -> str:
    """Window name of a unit at the given level."""
    if unit not in UNITS:
        raise ValueError(f"unknown unit {unit!r}")
    quadrant = UNIT_QUADRANT[unit.split("-")[0]]
    return quadrant if level == "quadrant" else QUADRANT_ARCH[quadrant]


def units_to_regions(units, level: str = "quadrant") -> list[str]:
    names = {unit_region(u, level) for u in units}
    return [r for r in CROPS[level] if r in names]


def quadrants_to_regions(quadrants, level: str = "quadrant") -> list[str]:
    """Quadrant names (UR, UL, LL, LR) at the given level, in window order."""
    names = set(quadrants) if level == "quadrant" else {QUADRANT_ARCH[q] for q in quadrants}
    return [r for r in CROPS[level] if r in names]


def region_phrase(region: str, scheme: str = "quadrant", patient_side: bool | None = None) -> str:
    """The words for a window. With patient_side False the quadrant words are mirrored."""
    if scheme not in REGION_PHRASES or region not in REGION_PHRASES[scheme]:
        raise ValueError(f"unknown region {region!r} for scheme {scheme!r}")
    if patient_side is None:
        patient_side = QUADRANT_WORDS_ARE_PATIENT_SIDE
    if scheme == "quadrant" and not patient_side:
        region = _MIRROR[region]
    return REGION_PHRASES[scheme][region]


# ----------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------
def with_mode(question: str, mode: str) -> str:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    return question if mode == "plain" else f"{question}\n\n{THINK_SUFFIX}"


def presence_question(condition: str, mode: str = "plain", region: str | None = None,
                      scheme: str = "quadrant") -> str:
    """Figure 7 question for the whole image, or for one named region."""
    if region is None:
        text = PRESENCE_QUESTION.format(label=LABELS[condition])
    else:
        text = REGION_PRESENCE_QUESTION.format(label=LABELS[condition], region=region_phrase(region, scheme))
    return with_mode(text, mode)


def count_scope(condition: str, region: str | None = None, scheme: str = "quadrant",
                patient_side: bool | None = None) -> str:
    """Scope words of a count question: the whole image, or a region ("... of the panoramic radiograph")."""
    _, whole = COUNT_TEMPLATES[condition]
    if region is None:
        return whole
    phrase = region_phrase(region, scheme, patient_side)
    return phrase if whole == "the image" else f"{phrase} of {whole}"


def count_question(condition: str, mode: str = "plain", region: str | None = None,
                   scheme: str = "quadrant") -> str:
    template, _ = COUNT_TEMPLATES[condition]
    return with_mode(template.format(scope=count_scope(condition, region, scheme)), mode)


def combined_question(condition: str, mode: str = "plain", region: str | None = None,
                      scheme: str = "quadrant", crop: bool = False) -> str:
    """Presence-and-count question for one countable finding: the whole image, one named region
    (words), or the whole-image wording on a crop (crop=True). Quadrant words are the patient's sides
    and the scope note names the image half, so QUADRANT_WORDS_ARE_PATIENT_SIDE does not apply."""
    if condition not in COUNTABLE:
        raise ValueError(f"{condition!r} is presence-only; use presence_question")
    if crop and region is not None:
        raise ValueError("a crop carries the whole-image wording: give region or crop, not both")
    label = LABELS[condition]
    definition, counting_rule = DEFINITIONS[condition]
    template, _ = COUNT_TEMPLATES[condition]
    if region is None:
        presence = PRESENCE_QUESTION.format(label=label)
        scope = CROP_SCOPE if crop else WHOLE_SCOPE
    else:
        presence = REGION_PRESENCE_QUESTION.format(label=label, region=region_phrase(region, scheme, patient_side=True))
        scope = f"only {SCOPE_NOTES[region]}; ignore every tooth outside it."
    count = template.format(scope=count_scope(condition, region, scheme, patient_side=True))
    text = COMBINED_QUESTION.format(context=COMBINED_CONTEXT, label=label, definition=definition, scope=scope,
                                    presence_question=presence, count_question=count, counting_rule=counting_rule)
    return with_mode(text, mode)


# ----------------------------------------------------------------------------
# Answer extraction (lenient, like the paper's rule-based reward)
# ----------------------------------------------------------------------------
_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}
_FDI_NUMBERS = {q * 10 + t for q in (1, 2, 3, 4) for t in range(1, 9)}
_TOOTH_REFERENCE = re.compile(r"\b(?:tooth|teeth)[\s:(]*#?\d{1,2}(?:\s*(?:,|and|&)\s*#?\d{1,2})*|#\d{1,2}", re.I)


def answer_body(text: str) -> str:
    """The graded part of a response: last <answer> block, else text after </think>.

    An opened but unclosed <think> block (cut off by max_tokens) is not an answer.
    """
    blocks = re.findall(r"<answer>(.*?)</answer>", text, flags=re.I | re.S)
    if blocks:
        return blocks[-1].strip()
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip()
    if "<think>" in text:
        return ""
    return text.strip()


def graded(reply: dict, extract):
    """Grade a model reply; a reply cut off by max_tokens without a closed <answer> is unparseable."""
    if reply.get("truncated") and "</answer>" not in reply["text"].lower():
        return None
    return extract(reply["text"])


def extract_choice(text: str) -> str | None:
    """Return 'A', 'B', or None (unparseable). Never guesses."""
    body = answer_body(text)
    a_option, b_option = r"\bA\s*[.)]\s*True\b", r"\bB\s*[.)]\s*False\b"
    if re.search(a_option, body, re.I) and re.search(b_option, body, re.I):
        # The options were restated before answering; drop that first copy only.
        body = re.sub(a_option, " ", body, count=1, flags=re.I)
        body = re.sub(b_option, " ", body, count=1, flags=re.I)
    match = re.search(r"\b([AB])\s*[.):]?\s*(True|False)\b", body, re.I)
    if match:
        return match.group(1).upper()
    # "Answer: A", "The answer is A because ...", "(B)", "**A**", "B." (letter stays case-sensitive
    # so the article "a" is never read as option A).
    match = (re.search(r"(?i:answer|option|choice)\s*(?:is|:)?\s*[\"'*(]*([AB])\b", body)
             or re.search(r"(?:^|[\s(\[*\"'>])([AB])(?=[.),:\]*\"'\n]|\s+(?:is|because)\b|\s*$)", body))
    if match:
        return match.group(1)
    truthy = re.search(r"\b(true|yes)\b", body, re.I)
    falsy = re.search(r"\b(false|no)\b", body, re.I)
    if truthy and not falsy:
        return "A"
    if falsy and not truthy:
        return "B"
    if truthy and falsy:
        return "A" if truthy.start() < falsy.start() else "B"
    return None


def extract_count(text: str) -> int | None:
    """Count of teeth/instances in the answer body; None if absent."""
    body = answer_body(text)
    # Prefer "10 teeth" / "3 implants" style totals (Figure 9 ends with "demonstrates 10 teeth").
    unit_counts = re.findall(
        r"(?<![\d#])\b(\d{1,3})\s+(?:visible\s+|distinct\s+)?(?:teeth|tooth|dental|implants?|roots?|residual|impacted|lesions?|crowns?|fillings?)\b",
        body, flags=re.I)
    if unit_counts:
        return int(unit_counts[-1])
    body = _TOOTH_REFERENCE.sub(" ", body)  # "teeth 16, 26 and 36", "tooth 36", "#16" are not counts
    digits = [int(d) for d in re.findall(r"(?<![\d#])\b(\d{1,3})\b", body)]
    words = [w.lower() for w in re.findall(r"[A-Za-z]+", body)]
    numbers = [_NUMBER_WORDS[w] for w in words if w in _NUMBER_WORDS]
    if digits and all(d in _FDI_NUMBERS for d in digits) and any(n > 0 for n in numbers):
        return [n for n in numbers if n > 0][-1]  # "three teeth ..., on 16, 26 and 36" -> 3
    if digits:
        return digits[-1]
    if numbers:
        return numbers[-1]
    if {"no", "none"} & set(words):
        return 0
    return None


# The two closing lines of a combined reply ("Answer: A. True", "Count: 3"), also accepted mid-line,
# bold, or with a full-width colon; the last occurrence of each wins.
_ANSWER_LINE = re.compile(r"\banswer\b[\s*_]*[:=\uff1a]\s*([^\n;,]+)", re.I)
_COUNT_LINE = re.compile(r"\bcount\b[\s*_]*[:=\uff1a]\s*([^\n;,.]+)", re.I)


def extract_combined(text: str) -> tuple[str | None, int | None]:
    """(choice, count) of a combined reply: the last "Answer:" and "Count:" lines, else the lenient
    presence and count rules over the whole body. The count is what the reply says; whether it agrees
    with the choice is judged by consistent_pair."""
    body = answer_body(text)
    answers, counts = _ANSWER_LINE.findall(body), _COUNT_LINE.findall(body)
    choice = extract_choice(answers[-1]) if answers else extract_choice(body)
    count = extract_count(counts[-1]) if counts else extract_count(_ANSWER_LINE.sub(" ", body))
    return choice, count


def consistent_pair(choice: str | None, count: int | None) -> tuple[str | None, int | None]:
    """The (presence, count) kept from a combined reply. Never guesses: with the letter unreadable
    nothing is kept, and a count that contradicts the letter (A with 0, B with more than 0) is
    unparseable while the letter stands."""
    if choice is None:
        return None, None
    if count is not None and (choice == "A") != (count > 0):
        return choice, None
    return choice, count


def graded_pair(reply: dict) -> tuple[str | None, int | None]:
    """Graded (presence, count) of a combined reply; truncated replies are unparseable as in graded()."""
    return consistent_pair(*(graded(reply, extract_combined) or (None, None)))


# ----------------------------------------------------------------------------
# Images and crops
# ----------------------------------------------------------------------------
def image_data_uri(image: str | Path | bytes, mime: str = "image/png") -> str:
    if isinstance(image, (bytes, bytearray)):
        payload = bytes(image)
    else:
        path = Path(image)
        guessed, _ = mimetypes.guess_type(path.name)
        if guessed and guessed.startswith("image/"):
            mime = guessed
        payload = path.read_bytes()
    return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"


def make_crops(image_path: str | Path, level: str, cache_dir: str | Path | None = None) -> dict[str, bytes]:
    """Return {region: png_bytes} for the requested scheme, optionally cached on disk."""
    if level not in CROPS:
        raise ValueError(f"level must be one of {tuple(CROPS)}")
    from PIL import Image

    crops: dict[str, bytes] = {}
    with Image.open(image_path) as source:
        width, height = source.size
        for region, (left, top, right, bottom) in CROPS[level].items():
            target = Path(cache_dir) / f"{Path(image_path).stem}_{region}.png" if cache_dir else None
            if target is not None and target.is_file():
                crops[region] = target.read_bytes()
                continue
            box = (round(left * width), round(top * height), round(right * width), round(bottom * height))
            buffer = io.BytesIO()
            source.crop(box).save(buffer, format="PNG")
            crops[region] = buffer.getvalue()
            if target is not None:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(crops[region])
    return crops


# ----------------------------------------------------------------------------
# Model runner (OpenAI-compatible chat completions; llama.cpp or a hosted API)
# ----------------------------------------------------------------------------
class VisionRunner:
    """One image + one question -> one answer. No paraphrase or sampling retries.

    Local llama.cpp by default; from_api() builds one for a hosted model from an llm_api spec.
    temperature None leaves the field out and token_param "max_completion_tokens" replaces
    max_tokens, as OpenAI reasoning models require. Transport errors, rate limits and 5xx are
    retried by the client with backoff; a bad request or key fails at once.
    """

    def __init__(
        self,
        base_url: str | None = "http://127.0.0.1:8080/v1",
        api_key: str = "local-llama-cpp",
        model: str = "dentalgpt",
        max_tokens: int = 4096,
        temperature: float | None = 0.0,
        timeout: float = 600.0,
        local: bool = True,
        cache_prompt: bool = True,
        request_options: dict | None = None,
        token_param: str = "max_tokens",
        api_call_retries: int = 2,
        response_cache: ResponseCache | None = None,
        call_log: str | None = None,
        client=None,
    ) -> None:
        if token_param not in llm_api.TOKEN_PARAMS:
            raise ValueError(f"token_param must be one of {llm_api.TOKEN_PARAMS}")
        llm_api.validate_api_retries(api_call_retries)
        self.client = client if client is not None else llm_api.connect(base_url, api_key, timeout)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.token_param = token_param
        self.local = local
        self.cache_prompt = cache_prompt
        self.request_options = dict(request_options or {})
        self.api_call_retries = api_call_retries
        self.response_cache = response_cache
        # Counters for every question asked, and the policy for printing single calls: one line per
        # image is denser than one per question, so a call prints only when it is worth reading.
        self.call_log = mon.CallLog("analyzer", call_log)

    @classmethod
    def from_api(cls, spec: dict, max_tokens: int = 4096, temperature: float | None = 0.0,
                 timeout: float = 600.0, api_call_retries: int = 2, client=None) -> "VisionRunner":
        """Runner for a hosted model. spec = {"provider", "model", ...} as documented in llm_api.

        A "temperature" or "token_param" in the spec wins over the arguments, so the spec of a
        reasoning model can say that it rejects a temperature.
        """
        base_url, api_key = llm_api.resolve(spec)
        return cls(base_url=base_url, api_key=api_key, model=spec["model"], max_tokens=max_tokens,
                   temperature=spec.get("temperature", temperature), timeout=timeout, local=False,
                   cache_prompt=False, request_options=spec.get("request_options"),
                   token_param=spec.get("token_param", "max_tokens"),
                   api_call_retries=spec.get("api_call_retries", api_call_retries), client=client)

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
        return {
            "model": self.model, "max_tokens": self.max_tokens, "token_param": self.token_param,
            "temperature": self.temperature, "local": self.local, "cache_prompt": self.cache_prompt,
            "request_options": self.request_options,
            "api_call_retries": self.api_call_retries,
        }

    def ask(self, image: str | Path | bytes, question: str) -> dict:
        request = {
            "model": self.model,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_data_uri(image)}},
                {"type": "text", "text": question},
            ]}],
            **llm_api.generation_fields(self.token_param, self.max_tokens, self.temperature),
        }
        if self.local:
            # Keep <think>/<answer> text intact, reuse the image KV prefix, and use the same
            # repetition-penalty value as the backbone's generation_config (llama.cpp applies it
            # over the last 64 tokens rather than the whole sequence).
            request["extra_body"] = {"reasoning_format": "none", "cache_prompt": self.cache_prompt,
                                     "repeat_penalty": 1.05, "seed": 0}
        request.update(self.request_options)
        cache_key = self.response_cache.key(request) if self.response_cache is not None else None
        cached = self.response_cache.get(cache_key) if self.response_cache is not None else None
        if cached is not None:
            return self.call_log.cached({**cached, "cache_hit": True, "cache_key": cache_key})
        started = time.perf_counter()
        normalized = llm_api.call_with_retries(
            lambda: llm_api.chat_reply(self.client.chat.completions.create(**request)), self.api_call_retries,
            f"analyzer model={self.model}")
        result = {**normalized, "latency_seconds": round(time.perf_counter() - started, 3),
                  "cache_hit": False, "cache_key": cache_key}
        if self.response_cache is not None:
            self.response_cache.put(cache_key, result)
        return self.call_log.live(result)


# ----------------------------------------------------------------------------
# Probe: does this checkpoint emit <think> tags on its own?
# ----------------------------------------------------------------------------
def probe(runner, image_paths, n: int = 10) -> dict:
    """Send the bare presence and count questions, with and without the suffix.

    Recommends "plain" when the model already reasons in tags on its own, and
    "tagged" only when the suffix actually produces the tagged format without
    mostly running past max_tokens.
    """
    paths = list(image_paths)[:n]
    stats = {}
    for mode in MODES:
        rows = []
        for path in paths:
            for kind, question in (("presence", presence_question("endodontic_treatment", mode)),
                                   ("count", count_question("dental_filling", mode))):
                reply = runner.ask(path, question)
                text = reply["text"]
                rows.append({
                    "image": str(path), "kind": kind, "mode": mode, "text": text,
                    "has_think": "<think>" in text, "has_answer": "<answer>" in text,
                    "parsed": graded(reply, extract_choice if kind == "presence" else extract_count),
                    "truncated": reply["truncated"], "completion_tokens": reply["completion_tokens"],
                })
        total = len(rows) or 1
        stats[mode] = {
            "think_tag_rate": sum(r["has_think"] for r in rows) / total,
            "answer_tag_rate": sum(r["has_answer"] for r in rows) / total,
            "parse_rate": sum(r["parsed"] is not None for r in rows) / total,
            "truncation_rate": sum(r["truncated"] for r in rows) / total,
            "samples": rows,
        }
    plain, tagged = stats["plain"], stats["tagged"]
    if plain["think_tag_rate"] >= 0.5 or tagged["think_tag_rate"] < 0.5 or tagged["truncation_rate"] > 0.5:
        recommended = "plain"
    else:
        recommended = "tagged"
    return {"recommended_mode": recommended, "images": len(paths), **stats}


# ----------------------------------------------------------------------------
# Protocol: the two levels and how a region is put to the model
# ----------------------------------------------------------------------------
PRESENCE_LEVELS = ("overall", "region")
COUNT_LEVELS = ("overall", "region")
REGION_PROMPTS = ("words", "crop")


@dataclass(frozen=True)
class Protocol:
    """What the wrapper may vary. Defaults are the recommended run.

    counting        True: counts are taken where count_level says. False: no count question of
                    any kind; the model only decides presence, on the whole image and, with
                    presence_level "region", in every region, so a finding's result is its
                    presence and its region set. count_level and question_form then have no
                    effect: every question is the bare Figure 7 question, on both backends.
    presence_level  "overall": presence from the whole-image question only.
                    "region":  the same question for every finding in every region, region by
                               region, independent of the whole-image answers (kept as a separate
                               result). A finding is present when any region answers A and absent
                               only when every region answers B; the region set is the regions
                               answering A.
    count_level     "overall": one whole-image count per positive countable finding.
                    "region":  one count per region: right after a region answers A when
                               presence_level is "region", else in every region for every
                               countable finding. The finding's count is the sum. A region count
                               of 0 is a valid answer.
    region_scheme   "quadrant" (UR, UL, LL, LR) or "arch" (upper, lower).
    region_prompt   "words": the region is named in the question and the whole image is sent.
                    "crop":  the whole-image question is sent with the region crop.
    question_form   "separate": the Figure 7 presence question, then the Figure 9-shaped count question
                                for a positive countable finding (DentalGPT's shapes).
                    "combined": for hosted models. Wherever the separate form would ask a presence and
                                then a count question in the same scope, one presence-and-count question
                                is asked instead (the five presence-only findings keep the bare presence
                                question); one reply gives both. A count that contradicts the letter is
                                unparseable. With presence_level "region" and count_level "overall", a
                                finding the whole image did not answer A but a region did still gets
                                the whole-image count question.
    """

    presence_level: str = "region"
    count_level: str = "region"
    region_scheme: str = "quadrant"
    region_prompt: str = "words"
    question_form: str = "separate"
    parse_retries: int = 0  # extra attempts per failed question; notebook defaults to 1
    counting: bool = True

    def __post_init__(self) -> None:
        llm_api.validate_parse_retries(self.parse_retries)
        if type(self.counting) is not bool:
            raise ValueError(f"counting must be True or False, got {self.counting!r}")
        for value, allowed, name in ((self.presence_level, PRESENCE_LEVELS, "presence_level"),
                                     (self.count_level, COUNT_LEVELS, "count_level"),
                                     (self.region_scheme, REGION_SCHEMES, "region_scheme"),
                                     (self.region_prompt, REGION_PROMPTS, "region_prompt"),
                                     (self.question_form, QUESTION_FORMS, "question_form")):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {value!r}")

    @property
    def counts_per_region(self) -> bool:
        """Counts are taken in every region (never with counting off)."""
        return self.counting and self.count_level == "region"

    @property
    def combined(self) -> bool:
        """The presence-and-count question is in use (never with counting off)."""
        return self.counting and self.question_form == "combined"

    @property
    def uses_regions(self) -> bool:
        return self.presence_level == "region" or self.counts_per_region

    @property
    def regions(self) -> tuple[str, ...]:
        """Region names in window order, empty when both levels are "overall"."""
        return tuple(CROPS[self.region_scheme]) if self.uses_regions else ()


# ----------------------------------------------------------------------------
# Per-image analysis and dataset runs with resume
# ----------------------------------------------------------------------------
def _record(calls: list, stage: str, condition: str, region: str | None, question: str, reply: dict,
            parsed: dict | None = None) -> None:
    call = {"stage": stage, "condition": condition, "region": region, "question": question, **reply}
    if parsed is not None:
        call["parsed"] = parsed
    calls.append(call)


def analyze_image(runner, image_path: str | Path, mode: str = "plain", protocol: Protocol = Protocol(),
                  crop_dir: str | Path | None = None) -> dict:
    """Whole-image presence for all 14 findings (kept under "whole_image"), then, as the protocol
    says, every region for every finding: presence, and a count as soon as a region answers A. The
    whole-image answers never decide which regional questions are asked. Deterministic order:
    region-major, so with crops the crop's image prefix stays cached; with words every call shares
    the whole image. With question_form "combined", a slot that would take a presence question and
    then a count question takes one presence-and-count question instead (countable findings only);
    the call record keeps the raw pair, so a count rejected as contradictory stays visible. With
    counting off no count question is asked at all: only the presence answers are filled in."""
    path = Path(image_path)
    calls: list[dict] = []
    findings = {c: {"presence": None, "whole_image": None, "count": None, "regions": None, "region_counts": None}
                for c in CONDITIONS}
    scheme, regions = protocol.region_scheme, protocol.regions
    by_crop = protocol.region_prompt == "crop"
    region_counts = protocol.counts_per_region
    combined = protocol.combined

    def ask(stage: str, condition: str, region: str | None, image, question: str):
        counting = stage in {"count", "region_count"}
        extract = extract_count if counting else extract_choice
        hint = ("Return only the whole-number count in digits." if counting else
                "Return only Answer: A. True or Answer: B. False, choosing one option.")
        if mode == "tagged":
            hint += " Put the final answer inside a closed <answer>...</answer> block."

        def parse(reply):
            value = graded(reply, extract)
            return value, None if value is not None else "missing_count" if counting else "missing_decision"

        value, _ = llm_api.ask_parsed(
            runner, image, question, parse=parse, retries=protocol.parse_retries,
            context=f"image={path.name} | condition={condition} | stage={stage} | region={region or 'whole'}",
            record=lambda q, r: _record(calls, stage, condition, region, q, r),
            fallback=lambda _: question + "\n\n" + hint)
        return value

    def ask_pair(stage: str, condition: str, region: str | None, image, question: str) -> tuple[str | None, int | None]:
        locked_choice = None
        count_only = False

        def parse(reply):
            nonlocal locked_choice
            reply["parse_target"] = "count" if count_only else "combined"
            raw = ((locked_choice, graded(reply, extract_count)) if count_only else
                   graded(reply, extract_combined) or (None, None))
            consistent = None if None in raw else (raw[0] == "A") == (raw[1] > 0)
            reply["parsed"] = {"choice": raw[0], "count": raw[1], "consistent": consistent}
            value = consistent_pair(*raw)
            if value[0] is not None:
                locked_choice = value[0]
            error = ("missing_decision" if value[0] is None else
                     "invalid_count_pair" if consistent is False else
                     "missing_count" if value[1] is None else None)
            return value, error

        def fallback(value):
            nonlocal count_only
            count_only = value[0] is not None
            if count_only:
                template, _ = COUNT_TEMPLATES[condition]
                # Preserve the combined prompt's patient-side scope, including crop mode.
                scope_region = region if region is not None and not by_crop else None
                text = template.format(scope=count_scope(condition, scope_region, scheme, patient_side=True))
                definition, counting_rule = DEFINITIONS[condition]
                text += f"\n\n{definition}\n{counting_rule}"
                hint = "Return only the whole-number count in digits."
                if mode == "tagged":
                    hint += " Put it inside a closed <answer>...</answer> block."
                return with_mode(text, mode) + "\n\n" + hint
            return question + ("\n\nReturn exactly two lines: Answer: A. True or Answer: B. False "
                               "(choose one), then Count: <integer>. No explanation.")

        value, _ = llm_api.ask_parsed(
            runner, image, question, parse=parse, retries=protocol.parse_retries,
            context=f"image={path.name} | condition={condition} | stage={stage} | region={region or 'whole'}",
            record=lambda q, r: _record(calls, stage, condition, region, q, r), fallback=fallback)
        return value

    # Whole image. With overall counts the combined form takes the count in the same reply.
    whole_counts: dict[str, int | None] = {}
    for condition in CONDITIONS:
        if combined and condition in COUNTABLE and not region_counts:
            answer, whole_counts[condition] = ask_pair("presence", condition, None, path, combined_question(condition, mode))
        else:
            answer = ask("presence", condition, None, path, presence_question(condition, mode))
        findings[condition]["whole_image"] = findings[condition]["presence"] = answer

    crops = make_crops(path, scheme, crop_dir) if (regions and by_crop) else {}

    def image_for(region: str):
        return crops[region] if by_crop else path

    def named(region: str) -> str | None:
        return None if by_crop else region

    if protocol.presence_level == "region":
        for condition in CONDITIONS:
            findings[condition]["regions"] = {}
            if region_counts and condition in COUNTABLE:
                findings[condition]["region_counts"] = {}
        for region in regions:
            for condition in CONDITIONS:
                if combined and region_counts and condition in COUNTABLE:
                    answer, count = ask_pair("region", condition, region, image_for(region),
                                             combined_question(condition, mode, named(region), scheme, crop=by_crop))
                    if answer == "A":
                        findings[condition]["region_counts"][region] = count
                else:
                    answer = ask("region", condition, region, image_for(region),
                                 presence_question(condition, mode, named(region), scheme))
                    if answer == "A" and region_counts and condition in COUNTABLE:
                        findings[condition]["region_counts"][region] = ask(
                            "region_count", condition, region, image_for(region),
                            count_question(condition, mode, named(region), scheme))
                findings[condition]["regions"][region] = answer
        for condition in CONDITIONS:
            answers = findings[condition]["regions"].values()
            # Present when any region answers A; absent only when every region answers B.
            findings[condition]["presence"] = "A" if "A" in answers else "B" if all(a == "B" for a in answers) else None
    elif region_counts:
        # Whole-image presence with region counts: every countable finding is counted in every region. The
        # combined form asks the region's presence-and-count question and keeps the count (0 with a B).
        for condition in COUNTABLE:
            findings[condition]["region_counts"] = {}
        for region in regions:
            for condition in COUNTABLE:
                if combined:
                    _, count = ask_pair("region_count", condition, region, image_for(region),
                                        combined_question(condition, mode, named(region), scheme, crop=by_crop))
                else:
                    count = ask("region_count", condition, region, image_for(region),
                                count_question(condition, mode, named(region), scheme))
                findings[condition]["region_counts"][region] = count

    for condition in COUNTABLE if protocol.counting else ():
        finding = findings[condition]
        if region_counts:
            counts = finding["region_counts"]
            complete = bool(counts) and all(n is not None for n in counts.values())
            finding["count"] = sum(counts.values()) if complete else None
        elif finding["presence"] == "A":
            if combined and finding["whole_image"] == "A":
                finding["count"] = whole_counts[condition]  # from the same reply; None when it contradicted the A
            else:
                # The separate form, or a finding the whole image did not answer A (found by the regions).
                finding["count"] = ask("count", condition, None, path, count_question(condition, mode))

    return {
        "image": str(path.resolve()),
        "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "mode": mode,
        "protocol": asdict(protocol),
        "region_scheme": scheme if regions else None,
        "findings": findings,
        "calls": calls,
        "call_count": len(calls),
        "inference_call_count": sum(not call.get("cache_hit", False) for call in calls),
        "cache_hit_count": sum(call.get("cache_hit", False) for call in calls),
        "parse_recovery": llm_api.parse_recovery_summary(calls),
    }


def run_config(mode: str, protocol: Protocol, runner_settings: dict, provenance: dict | None = None) -> dict:
    """Everything that defines a run; its hash guards resume. provenance = checkpoint/server facts."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}; run the probe or set MODE to one of them")
    words = protocol.uses_regions and protocol.region_prompt == "words"
    combined = protocol.combined
    config = {
        "mode": mode, "protocol": asdict(protocol), "labels": LABELS,
        "presence_question": PRESENCE_QUESTION,
        "region_presence_question": REGION_PRESENCE_QUESTION if words and protocol.presence_level == "region" else None,
        "region_phrases": ({r: region_phrase(r, protocol.region_scheme, patient_side=True if combined else None)
                            for r in protocol.regions} if words else None),
        "count_questions": COUNT_QUESTIONS if protocol.counting else None,
        "count_templates": {c: t for c, (t, _) in COUNT_TEMPLATES.items()} if protocol.counts_per_region else None,
        "combined": ({"question": COMBINED_QUESTION, "context": COMBINED_CONTEXT, "definitions": DEFINITIONS,
                      "whole_scope": WHOLE_SCOPE, "crop_scope": CROP_SCOPE,
                      "scope_notes": {r: SCOPE_NOTES[r] for r in protocol.regions} if words else None}
                     if combined else None),
        "think_suffix": THINK_SUFFIX if mode == "tagged" else None,
        "crops": CROPS[protocol.region_scheme] if protocol.uses_regions and protocol.region_prompt == "crop" else None,
        "runner": runner_settings, "provenance": provenance or {},
        "parse_recovery_version": 1,
    }
    config["hash"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    return config


def result_line(result: dict) -> str:
    """Dense one-line state of one image: what was found (with counts), and what stayed unresolved."""
    findings = result["findings"]
    present = [c for c in CONDITIONS if findings[c]["presence"] == "A"]
    absent = sum(findings[c]["presence"] == "B" for c in CONDITIONS)
    unclear = [c for c in CONDITIONS if findings[c]["presence"] is None]
    parts = [f"present={len(present)} absent={absent} unclear={len(unclear)}"]
    if present:
        named = [c + ("" if findings[c]["count"] is None else f" x{findings[c]['count']}") for c in present[:3]]
        parts.append(", ".join(named) + (f", +{len(present) - 3}" if len(present) > 3 else ""))
    if unclear:
        parts.append("unresolved: " + ", ".join(unclear[:3]) + (f", +{len(unclear) - 3}" if len(unclear) > 3 else ""))
    return " | ".join(parts)


def run_dataset(runner, images: dict[str, str | Path], out_dir: str | Path, mode: str = "plain",
                protocol: Protocol = Protocol(), resume: bool = True, provenance: dict | None = None,
                ledger: "mon.Ledger | None" = None, stop_after: int = 3) -> Path:
    """Analyze every image, one JSON per image, skipping finished ones on resume.

    An image that fails (a rejected request, an unreadable file, a model that never
    answers) is recorded with its complete traceback and the run moves to the next
    image, so one bad image never costs the rest of the dataset; the failures are
    saved as failures.json next to the results. `stop_after` consecutive failures
    stop the run instead, because that is a dead server or a rejected key rather
    than a bad image. A saved result that does not match its own run directory is
    still fatal: that is a mixed-up output directory, not a bad image.
    """
    out = Path(out_dir)
    results_dir = out / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    config = run_config(mode, protocol, runner.settings(), provenance)
    manifest_path = out / "manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("hash") != config["hash"]:
            raise ValueError(f"{out} holds a run with a different configuration; use a new directory.")
    else:
        manifest_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    todo = [(image_id, Path(p)) for image_id, p in images.items()]
    failures = mon.Ledger(f"{out.parent.name}/{out.name}")
    progress = mon.Progress(len(todo), label=f"{out.parent.name}/{out.name}", unit="image")
    for image_id, path in todo:
        target = results_dir / f"{image_id}.json"
        if resume and target.is_file():
            _load_result_file(target, image_id)  # a corrupt or foreign artifact stops the run
            progress.skip(image_id)
            continue
        with mon.guard(f"{out.name}/{image_id}", failures) as step:
            result = analyze_image(runner, path, mode=mode, protocol=protocol, crop_dir=out / "crops")
            result["image_id"] = image_id
            tmp = target.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
            tmp.replace(target)
        if not step.ok:
            if progress.failure(image_id) >= stop_after:
                progress.stop(f"{stop_after} images in a row failed; fix the cause and rerun to resume")
                break
            continue
        recovery = result["parse_recovery"]
        progress.item(image_id, result_line(result), calls=result["inference_call_count"],
                      cache=result["cache_hit_count"] or None, retries=recovery["retry_calls"] or None,
                      unparsed=recovery["unresolved_checks"] or None)
    log = getattr(runner, "call_log", None)
    progress.done(detail=log.line(counts=False) if isinstance(log, mon.CallLog) else "")
    if failures:
        failures.report(path=out / "failures.json")
        if ledger is not None:  # the sweep's own ledger keeps every stage's failures together
            ledger.entries.extend(failures.entries)
    return out


def _load_result_file(path: Path, expected_id: str | None = None) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        llm_api.monitor("ARTIFACT ERROR", str(path), reason=str(exc))
        raise ValueError(f"invalid result artifact {path}: {exc}") from exc
    image_id = payload.get("image_id") if isinstance(payload, dict) else None
    if not isinstance(image_id, str) or not image_id:
        raise llm_api.artifact_error(path, "missing non-empty image_id")
    if image_id != path.stem or (expected_id is not None and image_id != expected_id):
        raise llm_api.artifact_error(path, f"image_id {image_id!r} does not match filename/expected id")
    findings = payload.get("findings")
    if not isinstance(findings, dict) or any(c not in findings for c in CONDITIONS):
        raise llm_api.artifact_error(path, "incomplete findings schema")
    return payload


def load_results(out_dir: str | Path) -> dict[str, dict]:
    results = {}
    for path in sorted(Path(out_dir, "results").glob("*.json")):
        payload = _load_result_file(path)
        image_id = payload["image_id"]
        if image_id in results:
            raise llm_api.artifact_error(path, f"duplicate result image_id {image_id!r}")
        results[image_id] = payload
    return results


def dentist_report(result: dict) -> str:
    """Deterministic plain-text summary of one image result for a dentist."""
    present, absent, unclear = [], [], []
    for condition in CONDITIONS:
        finding = result["findings"][condition]
        label = LABELS[condition]
        if finding["presence"] == "A":
            parts = [label]
            region_counts = finding.get("region_counts")
            if finding["count"] is not None:
                text = f"count {finding['count']}"
                if region_counts:
                    text += " (" + ", ".join(f"{r} {n}" for r, n in region_counts.items()) + ")"
                parts.append(text)
            elif region_counts:
                known = [f"{r} {n}" for r, n in region_counts.items() if n is not None]
                parts.append("count incomplete" + (" (" + ", ".join(known) + ")" if known else ""))
            if finding.get("regions") is not None:
                hits = [r for r, a in finding["regions"].items() if a == "A"]
                parts.append("location " + ", ".join(hits) if hits else "location not resolved")
            elif region_counts:
                hits = [r for r, n in region_counts.items() if n]
                parts.append("location " + ", ".join(hits) if hits else "location not resolved")
            present.append(" - " + "; ".join(parts))
        elif finding["presence"] == "B":
            absent.append(label)
        else:
            unclear.append(label)
    lines = [f"Image: {Path(result['image']).name}", "Findings present:"]
    lines += present or [" - none"]
    lines.append("Not seen: " + (", ".join(absent) or "none"))
    if unclear:
        lines.append("Not assessable (unparseable answer): " + ", ".join(unclear))
    lines.append("Experimental model output for dentist review; not a diagnosis.")
    return "\n".join(lines)
