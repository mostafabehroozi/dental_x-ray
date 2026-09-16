"""In-distribution DentVLM pipeline: presence, location, and multiplicity on panoramic radiographs.

Every text sent to the model is a question DentVLM (Meng et al., Nature
Communications 2026; arXiv 2509.23344) was trained and evaluated on:

* One yes/no question per panoramic task, worded as in the authors' released
  test set or as one of the nine templates of Supplementary Table 7, e.g.
  "Based on the imaging, determine whether the patient has {task}?".
* DentVLM answers "Yes"/"No" on line 1 and then writes a rationale that names
  the location with one of nine fixed descriptors ("the left posterior region
  of the upper dentition", ...). Location is read from that rationale exactly
  as the authors' scorer does; nothing about location is ever asked in words.
* Multiplicity is the number of distinct regions the model names (0-6). The
  model is never asked to count teeth: it only decides presence, and a finding
  is scored per image and per cell as present or absent.
* The region comparison asks every task once per dental-arch region, on the
  whole uncropped image, by naming the region inside the task's own question
  with the model's own words: "... has caries in the left posterior region of
  the lower dentition?". Every region is asked whatever the whole image
  answered (kept as a separate result), so a finding missed with the model's
  attention on the whole image can be recovered in a region. The image is
  never cropped: a cropped panoramic is outside the model's image
  distribution, while the six region descriptors are the exact strings it was
  trained to write in its rationales.

Nothing else (JSON contracts, <think> tags, invented region wording, paraphrase
retries, forced zeros) is used. Findings the model has no task for are not
asked by default and are reported as "not assessed".
"""
from __future__ import annotations

import base64
import hashlib
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

# Display names for the dentist report.
LABELS = {
    "dental_implant": "Dental implant",
    "prosthetic_restoration": "Dental crown or bridge",
    "dental_filling": "Dental filling",
    "endodontic_treatment": "Root canal treatment",
    "carious_lesion": "Dental caries",
    "periodontal_bone_loss": "Periodontal bone loss",
    "impacted_tooth": "Impacted tooth",
    "periapical_lesion": "Periapical lesion",
    "root_fragment": "Residual root",
    "furcation_lesion": "Furcation involvement",
    "apical_surgery": "Apical surgery",
    "root_resorption": "Root resorption",
    "orthodontic_device": "Orthodontic appliance",
    "surgical_device": "Surgical fixation plate or screws",
}

# DentVLM's panoramic tasks (Supplementary Tables S2-S3 and the image-task mapping of
# Figure 1a), in asking order. questions[0] is Table S7 template #2 or the verbatim
# wording of the authors' released test set; [1] and [2] are verbatim alternates used
# only by the phrasing-ensemble option. Edit wording here only.
TASKS = {
    "implant": {"name": "Implant", "questions": (
        "Based on the imaging, determine whether the patient has an implant?",
        "Based on the imaging, does the patient have any abnormalities with the implant?",
        "Please confirm whether the patient has an implant?")},
    "prosthetic_crown": {"name": "Prosthetic Crown", "questions": (
        "Based on the imaging analysis, does the patient have a prosthetic crown?",
        "Please confirm whether the patient has a prosthetic crown?",
        "Based on the imaging, determine whether the patient has a prosthetic crown?")},
    "prosthetic_bridge": {"name": "Prosthetic Bridge", "questions": (
        "Based on the imaging, determine whether the patient has a prosthetic bridge?",
        "Evaluate the images to confirm whether there is a prosthetic bridge disease?",
        "Please confirm whether the patient has a prosthetic bridge?")},
    "fillings": {"name": "Fillings", "questions": (
        "Based on the imaging analysis, does the patient have fillings?",
        "Evaluate the images to confirm if there is a filling disease?",
        "Based on the imaging, determine whether the patient has fillings?")},
    "root_canal_therapy": {"name": "Root Canal Therapy", "questions": (
        "Based on the imaging, determine whether the patient has root canal filling?",
        "Based on the imaging analysis, does the patient have a root canal filling?",
        "Please confirm whether the patient has root canal therapy?")},
    "caries": {"name": "Caries", "questions": (
        "Based on the imaging analysis, does the patient have caries?",
        "Examine the images to determine if there is the presence of caries.",
        "Based on the imaging, determine whether the patient has caries?")},
    "periodontal_disease": {"name": "Periodontal Disease", "questions": (
        "Based on the imaging, determine whether the patient has periodontal disease?",
        "Examine the images to determine if periodontal disease is present?",
        "Whether a patient has periodontal disease through imaging?")},
    "impacted_tooth": {"name": "Impacted Tooth", "questions": (
        "Based on the imaging, determine whether the patient has an impacted tooth?",
        "Please confirm whether the patient has an impacted tooth?",
        "Based on the imaging analysis, does the patient have an impacted tooth?")},
    "apical_periodontitis": {"name": "Apical Periodontitis", "questions": (
        "Based on the imaging, does the patient have apical periodontitis abnormalities?",
        "Is there apical periodontitis in the images?",
        "Based on the imaging, determine whether the patient has apical periodontitis?")},
    "residual_root": {"name": "Residual Root", "questions": (
        "Examine the imaging to determine if there is a disease related to residual roots?",
        "Does the patient have any oral diseases related to residual roots?",
        "Based on the imaging, determine whether the patient has residual roots?")},
    "residual_crown": {"name": "Residual Crown", "questions": (
        "Please confirm whether the patient has a residual crown?",
        "Is there any oral disease related to residual crowns identified in the images?",
        "Based on the imaging, determine whether the patient has a residual crown?")},
    "insufficient_eruption_space": {"name": "Insufficient Space for Primary Tooth Eruption", "questions": (
        "Based on the imaging, does the patient have insufficient space for the eruption of primary teeth?",
        "Does the patient have insufficient space for the eruption of primary teeth?",
        "Please confirm whether the patient has insufficient space for the eruption of primary teeth?")},
    "calculus": {"name": "Calculus", "questions": (
        "Evaluate the images to confirm if there is calculus disease?",
        "Evaluate the images to confirm if there is a calculus disease?",
        "Based on the imaging, determine whether the patient has calculus?")},
}
MAX_PHRASINGS = 3

# UMFIH class -> DentVLM task(s). A crown or a bridge both count as a prosthetic restoration.
CONDITION_TASKS = {
    "dental_implant": ("implant",),
    "prosthetic_restoration": ("prosthetic_crown", "prosthetic_bridge"),
    "dental_filling": ("fillings",),
    "endodontic_treatment": ("root_canal_therapy",),
    "carious_lesion": ("caries",),
    "periodontal_bone_loss": ("periodontal_disease",),
    "impacted_tooth": ("impacted_tooth",),
    "periapical_lesion": ("apical_periodontitis",),
    "root_fragment": ("residual_root",),
}
TRAINED = tuple(c for c in CONDITIONS if c in CONDITION_TASKS)

# UMFIH classes DentVLM has no task for. Asked only with Protocol.ask_untrained, through
# Table S7 template #2 with these labels. The paper's zero-shot accuracy on diseases it was
# not trained on is 52-64%, so this is off by default and scored as trained_task=False.
UNTRAINED_LABELS = {
    "furcation_lesion": "furcation involvement",
    "apical_surgery": "apical surgery",
    "root_resorption": "root resorption",
    "orthodontic_device": "an orthodontic appliance",
    "surgical_device": "surgical fixation plates or screws",
}
UNTRAINED_TEMPLATE = "Based on the imaging, determine whether the patient has {label}?"

# DentVLM panoramic tasks without a UMFIH class: asked for the dentist report, never scored.
EXTRA_TASKS = ("residual_crown", "insufficient_eruption_space", "calculus")

# The nine location descriptors DentVLM writes in its rationale (Supplementary Note S1), in
# the order of the authors' scorer, and the six dental-arch cells they map onto. "left" and
# "right" are DentVLM's own words: Table S6 defines its "left posterior region" as FDI
# quadrants 1 and 4, the patient's right, which is the left side of a panoramic as displayed.
# LEFT_IS_IMAGE_LEFT records that reading; the DENTEX side check in the notebook confirms it,
# and flipping it mirrors the cell windows and the FDI mapping together.
CELLS = ("upper-right", "upper-anterior", "upper-left", "lower-right", "lower-anterior", "lower-left")
DESCRIPTORS = {
    "the right posterior region of both the upper and lower dentition": ("upper-right", "lower-right"),
    "the anterior region of both the upper and lower dentition": ("upper-anterior", "lower-anterior"),
    "the left posterior region of both the upper and lower dentition": ("upper-left", "lower-left"),
    "the right posterior region of the upper dentition": ("upper-right",),
    "the anterior region of the upper dentition": ("upper-anterior",),
    "the left posterior region of the upper dentition": ("upper-left",),
    "the right posterior region of the lower dentition": ("lower-right",),
    "the anterior region of the lower dentition": ("lower-anterior",),
    "the left posterior region of the lower dentition": ("lower-left",),
}
LEFT_IS_IMAGE_LEFT = True

# Cell windows as normalized (left, top, right, bottom) image coordinates. The anterior
# window covers incisors and canines; windows overlap so a box on the canine line or the
# occlusal plane is whole in at least one cell. Location truth is scored against these same
# windows (dental_eval, 25%-area rule); DENTEX uses its FDI tooth numbers instead.
_X = {"left": (0.00, 0.45), "anterior": (0.35, 0.65), "right": (0.55, 1.00)}
_Y = {"upper": (0.00, 0.58), "lower": (0.42, 1.00)}
_FLIP = {"left": "right", "right": "left", "anterior": "anterior"}


def cell_windows(left_is_image_left: bool = LEFT_IS_IMAGE_LEFT) -> dict[str, tuple[float, float, float, float]]:
    windows = {}
    for cell in CELLS:
        row, col = cell.split("-")
        image_side = col if left_is_image_left else _FLIP[col]
        (left, right), (top, bottom) = _X[image_side], _Y[row]
        windows[cell] = (left, top, right, bottom)
    return windows


CELL_WINDOWS = cell_windows()

# Dental-arch units: FDI quadrant x {anterior, posterior}. This is how DentVLM's authors built
# their location labels (box -> nearest teeth -> tooth-region mapping; anterior = incisors and
# canine, positions 1-3) and the finest division the six cells are made of, so ground-truth
# boxes translated into units (location_adapter) map onto cells deterministically.
UNITS = ("Q1-posterior", "Q1-anterior", "Q2-anterior", "Q2-posterior",
         "Q3-posterior", "Q3-anterior", "Q4-anterior", "Q4-posterior")


def fdi_unit(quadrant: int, tooth: int) -> str:
    """Unit of an FDI tooth position; primary-dentition quadrants 5-8 fold onto 1-4."""
    quadrant = quadrant - 4 if quadrant > 4 else quadrant
    return f"Q{quadrant}-{'anterior' if tooth <= 3 else 'posterior'}"


def unit_cell(unit: str, left_is_image_left: bool = LEFT_IS_IMAGE_LEFT) -> str:
    """DentVLM's cell for a unit (Table S6): its 'left' is FDI quadrants 1/4, the patient's right."""
    if unit not in UNITS:
        raise ValueError(f"unknown unit {unit!r}")
    quadrant, zone = int(unit[1]), unit.split("-")[1]
    row = "upper" if quadrant in (1, 2) else "lower"
    if zone == "anterior":
        return f"{row}-anterior"
    patient_right = quadrant in (1, 4)
    col = ("left" if patient_right else "right") if left_is_image_left else ("right" if patient_right else "left")
    return f"{row}-{col}"


def units_to_cells(units, left_is_image_left: bool = LEFT_IS_IMAGE_LEFT) -> list[str]:
    cells = {unit_cell(u, left_is_image_left) for u in units}
    return [c for c in CELLS if c in cells]


LOCATION_LEVELS = ("rationale", "regions", "none")
REGION_VOTES = ("union", "majority")

# The descriptor of each single-region cell, inverted from DESCRIPTORS so the words asked in a
# region question and the words the scorer matches in a rationale can never drift apart. The three
# "both the upper and lower" descriptors name two cells at once and are never asked.
CELL_DESCRIPTORS = {cells[0]: text for text, cells in DESCRIPTORS.items() if len(cells) == 1}
assert set(CELL_DESCRIPTORS) == set(CELLS), "every cell needs one descriptor to be asked about"


# ----------------------------------------------------------------------------
# Questions
# ----------------------------------------------------------------------------
def questions_for(task: str) -> tuple[str, ...]:
    if task in TASKS:
        return TASKS[task]["questions"]
    if task in UNTRAINED_LABELS:
        return (UNTRAINED_TEMPLATE.format(label=UNTRAINED_LABELS[task]),)
    raise KeyError(task)


def task_name(task: str) -> str:
    return TASKS[task]["name"] if task in TASKS else LABELS[task]


def region_question(task: str, cell: str, phrasing: int = 0) -> str:
    """The task's own question restricted to one region, e.g. "Based on the imaging analysis, does
    the patient have caries in the left posterior region of the lower dentition?".

    The smallest change that adds a region to an in-distribution question: the verbatim sentence is
    kept, and the only words added are one of the nine descriptors DentVLM was trained to write as
    a location (Supplementary Note 1), so the region is asked in the model's own vocabulary and its
    own left/right convention. Nothing explains the region: an explanation is text the model never
    saw.
    """
    if cell not in CELL_DESCRIPTORS:
        raise ValueError(f"unknown cell {cell!r}; expected one of {CELLS}")
    stem = questions_for(task)[phrasing].strip().rstrip("?.").rstrip()
    return f"{stem} in {CELL_DESCRIPTORS[cell]}?"


def condition_tasks(condition: str, ask_untrained: bool = False) -> tuple[str, ...]:
    """Task keys that decide a condition; empty when the model is not asked about it."""
    if condition in CONDITION_TASKS:
        return CONDITION_TASKS[condition]
    if ask_untrained and condition in UNTRAINED_LABELS:
        return (condition,)
    return ()


# ----------------------------------------------------------------------------
# Answer extraction (the authors' scorer: line 1 decides, regions by exact match)
# ----------------------------------------------------------------------------
_YES = re.compile(r"\byes\b", re.I)
_NO = re.compile(r"\bno\b", re.I)


def first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def extract_answer(text: str) -> str | None:
    """'yes', 'no', or None (unparseable). Read from the first line; both words -> None."""
    line = first_line(text)
    yes, no = bool(_YES.search(line)), bool(_NO.search(line))
    if yes == no:
        return None
    return "yes" if yes else "no"


def extract_regions(text: str) -> list[str]:
    """Cells named anywhere in the reply through the nine descriptors (exact, case-insensitive)."""
    low = text.lower()
    found = set()
    for descriptor, cells in DESCRIPTORS.items():
        if descriptor in low:
            found.update(cells)
    return [c for c in CELLS if c in found]


# ----------------------------------------------------------------------------
# Images
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
        model: str = "dentvlm",
        max_tokens: int = 512,
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
        # Image before the question and no system message of our own: the chat template
        # injects Qwen's default "You are a helpful assistant.", which the authors use.
        request = {
            "model": self.model,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_data_uri(image)}},
                {"type": "text", "text": question},
            ]}],
            **llm_api.generation_fields(self.token_param, self.max_tokens, self.temperature),
        }
        if self.local:
            # Reuse the image KV prefix across the questions of one image; repetition penalty as
            # in the authors' inference script (1.05).
            request["extra_body"] = {"cache_prompt": self.cache_prompt, "repeat_penalty": 1.05, "seed": 0}
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
# Protocol, per-image analysis, dataset runs with resume
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Protocol:
    """Everything the wrapper may vary. Defaults are the paper's protocol."""

    phrasings: int = 1            # 1, or up to 3 verbatim wordings per task with a majority vote
    region_vote: str = "union"    # with phrasings > 1: "union" (matching voting) or "majority"
    location: str = "rationale"   # "rationale" (free) | "regions" (every region named in the question, every task) | "none"
    ask_untrained: bool = False   # ask the five UMFIH classes DentVLM was never trained on
    extra_tasks: bool = True      # ask residual crown, eruption space, calculus (reported, not scored)
    parse_retries: int = 0        # extra attempts per failed question; notebook defaults to 1

    def __post_init__(self) -> None:
        llm_api.validate_parse_retries(self.parse_retries)
        if not 1 <= self.phrasings <= MAX_PHRASINGS:
            raise ValueError(f"phrasings must be between 1 and {MAX_PHRASINGS}")
        if self.region_vote not in REGION_VOTES:
            raise ValueError(f"region_vote must be one of {REGION_VOTES}")
        if self.location not in LOCATION_LEVELS:
            if self.location == "crops":  # the cropping mode this replaced
                raise ValueError("location 'crops' is gone: 'regions' names the region inside the "
                                 "question and keeps the whole image, which the cropping never did")
            raise ValueError(f"location must be one of {LOCATION_LEVELS}")

    def tasks(self) -> tuple[str, ...]:
        """Task keys in asking order: condition tasks in ontology order, then the extras."""
        keys: list[str] = []
        for condition in CONDITIONS:
            keys.extend(condition_tasks(condition, self.ask_untrained))
        if self.extra_tasks:
            keys.extend(EXTRA_TASKS)
        return tuple(dict.fromkeys(keys))


def vote(answers: list[dict], region_vote: str) -> dict:
    """Presence by majority of the parsed answers; regions from the yes answers."""
    parsed = [a["answer"] for a in answers if a["answer"] is not None]
    yes, no = parsed.count("yes"), parsed.count("no")
    presence = "yes" if yes > no else "no" if no > yes else None
    if presence != "yes":
        return {"presence": presence, "regions": None}
    positive = [set(a["regions"]) for a in answers if a["answer"] == "yes"]
    if region_vote == "union":
        cells = set().union(*positive)
    else:
        needed = len(positive) // 2 + 1  # strict majority of the yes answers
        cells = {c for c in CELLS if sum(c in r for r in positive) >= needed}
    return {"presence": "yes", "regions": [c for c in CELLS if c in cells]}


def _record(calls: list, stage: str, task: str, cell: str | None, question: str, reply: dict) -> None:
    calls.append({"stage": stage, "task": task, "cell": cell, "question": question, **reply})


def _any_yes(answers) -> str | None:
    """'yes' when any answer is yes, 'no' when every answer is no, else None (unparseable)."""
    answers = list(answers)
    if "yes" in answers:
        return "yes"
    return "no" if all(a == "no" for a in answers) else None


def cell_answers(result: dict) -> dict[str, dict[str, str | None]]:
    """{task: {cell: yes/no/None}} from the saved region calls (location "regions"): each region's answer."""
    answers: dict[str, dict] = {}
    for call in result.get("calls") or []:
        if call.get("stage") == "region" and call.get("cell"):
            recovery = call.get("parse_recovery")
            answer = recovery["value"] if recovery is not None else extract_answer(call["text"])
            answers.setdefault(call["task"], {})[call["cell"]] = answer
    return answers


def _finding(condition: str, tasks: dict, protocol: Protocol) -> dict:
    keys = condition_tasks(condition, protocol.ask_untrained)
    if not keys or any(k not in tasks for k in keys):
        return {"asked": False, "tasks": [], "presence": None, "whole_image": None, "regions": None,
                "region_count": None}
    presence = _any_yes(tasks[k]["presence"] for k in keys)
    regions = None
    if presence == "yes" and protocol.location != "none":
        named = set()
        for k in keys:
            if tasks[k]["presence"] == "yes":
                named.update(tasks[k]["regions"] or [])
        regions = [c for c in CELLS if c in named]
    return {"asked": True, "tasks": list(keys), "presence": presence,
            "whole_image": _any_yes(tasks[k]["whole_image"] for k in keys), "regions": regions,
            "region_count": len(regions) if regions is not None else None}


def analyze_image(runner, image_path: str | Path, protocol: Protocol = Protocol()) -> dict:
    """One yes/no question per task on the whole image; regions from the rationale, or from the same
    question asked once per region with the region named in it (the whole-image answers are then kept
    under "whole_image"). The image sent is always the whole radiograph. Deterministic order."""
    path = Path(image_path)
    calls: list[dict] = []
    tasks: dict[str, dict] = {}
    aggregation_warnings: list[dict] = []

    def ask(stage, task, cell, image, question):
        hint = ("Start your reply with exactly Yes or No on the first line, choosing one. "
                "Then give your brief rationale and location as requested.")

        def parse(reply):
            # A cut-off rationale can contain incomplete locations even if line 1 is readable.
            value = None if reply.get("truncated") else extract_answer(reply["text"])
            return value, None if value is not None else "missing_or_ambiguous_decision"

        return llm_api.ask_parsed(
            runner, image, question, parse=parse, retries=protocol.parse_retries,
            context=f"image={path.name} | task={task} | stage={stage} | cell={cell or 'whole'}",
            record=lambda q, r: _record(calls, stage, task, cell, q, r),
            fallback=lambda _: question + "\n\n" + hint)

    for task in protocol.tasks():
        answers = []
        for question in questions_for(task)[:protocol.phrasings]:
            answer, reply = ask("presence", task, None, path, question)
            answers.append({"answer": answer, "regions": extract_regions(reply["text"]) if answer is not None else [],
                            "truncated": reply["truncated"]})
        tasks[task] = {"name": task_name(task), "answers": answers, **vote(answers, protocol.region_vote)}
        parsed_answers = [a["answer"] for a in answers if a["answer"] is not None]
        if parsed_answers and tasks[task]["presence"] is None:
            warning = {"kind": "phrasing_tie", "task": task, "answers": parsed_answers, "policy": "neutral"}
            aggregation_warnings.append(warning)
            llm_api.monitor("AGGREGATION WARNING", f"task={task}", reason="phrasing tie", policy="neutral")
        tasks[task]["whole_image"] = tasks[task]["presence"]

    if protocol.location == "regions":
        # The same whole image, one question per region: the task's own sentence with one of the model's
        # nine location descriptors inside it. Every region is asked every task, whatever the whole image
        # answered, so a task missed with the model's attention spread over the whole image can be
        # recovered in a region: it is present when any region says yes, absent when every region says no.
        # Task-major (each finding walked region by region); every call shares the same image prefix.
        cell_answers = {task: {} for task in tasks}
        for task in tasks:
            for cell in CELLS:
                answer, _ = ask("region", task, cell, path, region_question(task, cell))
                cell_answers[task][cell] = answer
        for task, answers in cell_answers.items():
            presence = tasks[task]["presence"] = _any_yes(answers.values())
            tasks[task]["regions"] = [c for c in CELLS if answers[c] == "yes"] if presence == "yes" else None
    elif protocol.location == "none":
        for task in tasks:
            tasks[task]["regions"] = None

    findings = {c: _finding(c, tasks, protocol) for c in CONDITIONS}
    for condition, finding in findings.items():
        decisions = [tasks[k]["presence"] for k in finding["tasks"] if tasks[k]["presence"] is not None]
        if "yes" in decisions and "no" in decisions:
            warning = {"kind": "task_conflict", "condition": condition, "answers": decisions,
                       "policy": "existing any-yes aggregation"}
            aggregation_warnings.append(warning)
            llm_api.monitor("AGGREGATION WARNING", f"condition={condition}", reason="task conflict",
                            policy="any-yes")

    return {
        "image": str(path.resolve()),
        "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "protocol": asdict(protocol),
        "location_level": protocol.location,
        "left_is_image_left": LEFT_IS_IMAGE_LEFT,
        "tasks": tasks,
        "findings": findings,
        "calls": calls,
        "call_count": len(calls),
        "inference_call_count": sum(not call.get("cache_hit", False) for call in calls),
        "cache_hit_count": sum(call.get("cache_hit", False) for call in calls),
        "parse_recovery": llm_api.parse_recovery_summary(calls),
        "aggregation_warnings": aggregation_warnings,
    }


def run_config(protocol: Protocol, runner_settings: dict, provenance: dict | None = None) -> dict:
    """Everything that defines a run; its hash guards resume. provenance = checkpoint/server facts."""
    config = {
        "protocol": asdict(protocol),
        "questions": {task: questions_for(task)[:protocol.phrasings] for task in protocol.tasks()},
        "descriptors": DESCRIPTORS, "cells": CELLS,
        # The region questions are model input; the cell windows never are (they are evaluation
        # geometry), so flipping LEFT_IS_IMAGE_LEFT does not invalidate a saved run.
        "region_questions": ({task: {cell: region_question(task, cell) for cell in CELLS}
                              for task in protocol.tasks()} if protocol.location == "regions" else None),
        "runner": runner_settings, "provenance": provenance or {},
        "parse_recovery_version": 1,
    }
    config["hash"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    return config


def result_line(result: dict) -> str:
    """Dense one-line state of one image: what was found, what stayed unresolved, what was never
    asked (a finding with no trained task), and any aggregation warning behind the answers."""
    findings = result["findings"]
    asked = [c for c in CONDITIONS if findings[c]["asked"]]
    present = [c for c in asked if findings[c]["presence"] == "yes"]
    unclear = [c for c in asked if findings[c]["presence"] is None]
    parts = [f"yes={len(present)} no={sum(findings[c]['presence'] == 'no' for c in asked)} "
             f"unclear={len(unclear)} not_asked={len(CONDITIONS) - len(asked)}"]
    if present:
        parts.append(", ".join(present[:3]) + (f", +{len(present) - 3}" if len(present) > 3 else ""))
    if unclear:
        parts.append("unresolved: " + ", ".join(unclear[:3]) + (f", +{len(unclear) - 3}" if len(unclear) > 3 else ""))
    warnings = result.get("aggregation_warnings") or ()
    if warnings:
        parts.append(f"aggregation warnings={len(warnings)}")
    return " | ".join(parts)


def run_dataset(runner, images: dict[str, str | Path], out_dir: str | Path, protocol: Protocol = Protocol(),
                resume: bool = True, provenance: dict | None = None,
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
    config = run_config(protocol, runner.settings(), provenance)
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
            result = analyze_image(runner, path, protocol=protocol)
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


# ----------------------------------------------------------------------------
# Dentist summary
# ----------------------------------------------------------------------------
def describe_cell(cell: str, left_is_image_left: bool = LEFT_IS_IMAGE_LEFT) -> str:
    """Patient-side wording for a cell, with the image side in parentheses."""
    row, col = cell.split("-")
    if col == "anterior":
        return f"{row} anterior"
    image_side = col if left_is_image_left else _FLIP[col]
    return f"patient's {row} {_FLIP[image_side]} posterior (image {image_side})"


def dentist_report(result: dict) -> str:
    """Deterministic plain-text summary of one image result for a dentist."""
    flag = result.get("left_is_image_left", LEFT_IS_IMAGE_LEFT)
    present, absent, unclear, not_assessed = [], [], [], []
    for condition in CONDITIONS:
        finding = result["findings"][condition]
        label = LABELS[condition]
        if not finding["asked"]:
            not_assessed.append(label)
        elif finding["presence"] == "yes":
            parts = [label]
            if finding["regions"]:
                parts.append(f"in {len(finding['regions'])} region(s): "
                             + ", ".join(describe_cell(c, flag) for c in finding["regions"]))
            elif finding["regions"] is not None:
                parts.append("region not stated")
            present.append(" - " + "; ".join(parts))
        elif finding["presence"] == "no":
            absent.append(label)
        else:
            unclear.append(label)
    extras_present = [t["name"] for k, t in result["tasks"].items() if k in EXTRA_TASKS and t["presence"] == "yes"]
    lines = [f"Image: {Path(result['image']).name}", "Findings present:"]
    lines += present or [" - none"]
    if extras_present:
        lines.append("Also present (no benchmark class): " + ", ".join(extras_present))
    lines.append("Not seen: " + (", ".join(absent) or "none"))
    if unclear:
        lines.append("Not assessable (unparseable answer): " + ", ".join(unclear))
    if not_assessed:
        lines.append("Not assessed by this model: " + ", ".join(not_assessed))
    lines.append("Experimental model output for dentist review; not a diagnosis.")
    return "\n".join(lines)
