"""Twelve source-backed PAN VQA tasks, native answers, and external parsing.

New inference uses a fixed protocol. Benchmark projection is evaluation-only;
historical voting helpers remain available for read-only artifact analysis.
Source-region names are not verified patient-side anatomical labels.
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

# Canonical clinical vocabulary. Benchmark class IDs live in benchmark_schema.py.
PROFILE = "pan_training_aligned_v1"
RESULT_SCHEMA = "dentvlm-pan/1"
PARSER_VERSION = 1
SYSTEM_MESSAGE = "You are a helpful assistant."
SOURCE_REVISION = "9463edb2af47f64510b0681efc20be6ecf870955"
SAMPLING = {"top_p": 0.001, "repeat_penalty": 1.05, "repeat_last_n": -1,
            "samplers": ["penalties", "temperature", "top_p"], "top_k": 0,
            "min_p": 0.0, "seed": 0}
# Location reference patterns only detect ambiguous associations; they never create diagnoses.
TASKS = {'impacted_tooth': {'name': 'Impacted Tooth',
                    'questions': ['Based on the imaging, determine whether the patient has an impacted '
                                  'tooth?'],
                    'question_provenance': {'evidence': 'author_released_training_example',
                                            'record_id': 'en_dis_panoramic_2666_Impacted Tooth',
                                            'revision': '9463edb2af47f64510b0681efc20be6ecf870955',
                                            'file': 'data/inst_data_2nd_train.json',
                                            'source_modality': 'PAN',
                                            'pan_support': 'arXiv:2509.23344 Figure 1a'},
                    'location_reference_pattern': '\\bimpacted (?:tooth|teeth)\\b'},
 'prosthetic_crown': {'name': 'Prosthetic Crown',
                      'questions': ['Based on the imaging analysis, does the patient have a prosthetic '
                                    'crown?'],
                      'question_provenance': {'evidence': 'author_released_training_example',
                                              'record_id': 'en_dis_panoramic_5156_Prosthetic Crown',
                                              'revision': '9463edb2af47f64510b0681efc20be6ecf870955',
                                              'file': 'data/inst_data_2nd_train.json',
                                              'source_modality': 'PAN',
                                              'pan_support': 'arXiv:2509.23344 Figure 1a'},
                      'location_reference_pattern': '(?<!residual )\\bcrowns?\\b'},
 'root_canal_therapy': {'name': 'Root Canal Therapy',
                        'questions': ['Based on the imaging, determine whether the patient has root '
                                      'canal filling?'],
                        'question_provenance': {'evidence': 'author_released_training_example',
                                                'record_id': 'en_dis_panoramic_2109_Root Canal Therapy',
                                                'revision': '9463edb2af47f64510b0681efc20be6ecf870955',
                                                'file': 'data/inst_data_2nd_train.json',
                                                'source_modality': 'PAN',
                                                'pan_support': 'arXiv:2509.23344 Figure 1a'},
                        'documented_alternative': {'question': 'Can root canal therapy be seen in this '
                                                               'dental X-ray?',
                                                   'evidence': 'paper_figure_1_example',
                                                   'used_by_baseline': False},
                        'location_reference_pattern': '\\broot canal (?:therapy|treatment|filling)\\b'},
 'fillings': {'name': 'Fillings',
              'questions': ['Based on the imaging analysis, does the patient have fillings?'],
              'question_provenance': {'evidence': 'author_released_training_example',
                                      'record_id': 'en_dis_panoramic_336_Fillings',
                                      'revision': '9463edb2af47f64510b0681efc20be6ecf870955',
                                      'file': 'data/inst_data_2nd_train.json',
                                      'source_modality': 'PAN',
                                      'pan_support': 'arXiv:2509.23344 Figure 1a'},
              'location_reference_pattern': '(?<!canal )\\bfillings?\\b'},
 'prosthetic_bridge': {'name': 'Prosthetic Bridge',
                       'questions': ['Based on the imaging, determine whether the patient has a '
                                     'prosthetic bridge?'],
                       'question_provenance': {'evidence': 'author_released_training_example',
                                               'record_id': 'en_dis_panoramic_2869_Prosthetic Bridge',
                                               'revision': '9463edb2af47f64510b0681efc20be6ecf870955',
                                               'file': 'data/inst_data_2nd_train.json',
                                               'source_modality': 'PAN',
                                               'pan_support': 'arXiv:2509.23344 Figure 1a'},
                       'location_reference_pattern': '\\bbridges?\\b'},
 'apical_periodontitis': {'name': 'Apical Periodontitis',
                          'questions': ['Based on the imaging, does the patient have apical '
                                        'periodontitis abnormalities?'],
                          'question_provenance': {'evidence': 'author_released_training_example',
                                                  'record_id': 'en_dis_panoramic_5065_Apical '
                                                               'Periodontitis',
                                                  'revision': '9463edb2af47f64510b0681efc20be6ecf870955',
                                                  'file': 'data/inst_data_2nd_train.json',
                                                  'source_modality': 'PAN',
                                                  'pan_support': 'arXiv:2509.23344 Figure 1a'},
                          'location_reference_pattern': '\\b(?:apical periodontitis|periapical '
                                                        'lesions?)\\b'},
 'residual_root': {'name': 'Residual Root',
                   'questions': ['Examine the imaging to determine if there is a disease related to '
                                 'residual roots?'],
                   'question_provenance': {'evidence': 'author_released_training_example',
                                           'record_id': 'en_dis_panoramic_1971_Residual Root',
                                           'revision': '9463edb2af47f64510b0681efc20be6ecf870955',
                                           'file': 'data/inst_data_2nd_train.json',
                                           'source_modality': 'PAN',
                                           'pan_support': 'arXiv:2509.23344 Figure 1a'},
                   'location_reference_pattern': '\\b(?:residual roots?|root fragments?)\\b'},
 'implant': {'name': 'Implant',
             'questions': ['Based on the imaging, determine whether the patient has an implant?'],
             'question_provenance': {'evidence': 'author_released_training_example',
                                     'record_id': 'en_dis_panoramic_392_Implant',
                                     'revision': '9463edb2af47f64510b0681efc20be6ecf870955',
                                     'file': 'data/inst_data_2nd_train.json',
                                     'source_modality': 'PAN',
                                     'pan_support': 'arXiv:2509.23344 Figure 1a'},
             'location_reference_pattern': '\\bimplants?\\b'},
 'residual_crown': {'name': 'Residual Crown',
                    'questions': ['Please confirm whether the patient has a residual crown?'],
                    'question_provenance': {'evidence': 'author_released_training_example',
                                            'record_id': 'en_dis_panoramic_1296_Residual Crown',
                                            'revision': '9463edb2af47f64510b0681efc20be6ecf870955',
                                            'file': 'data/inst_data_2nd_train.json',
                                            'source_modality': 'PAN',
                                            'pan_support': 'arXiv:2509.23344 Figure 1a'},
                    'location_reference_pattern': '\\bresidual crowns?\\b'},
 'insufficient_eruption_space': {'name': 'Insufficient Space for Primary Tooth Eruption',
                                 'questions': ['Based on the imaging, does the patient have insufficient '
                                               'space for the eruption of primary teeth?'],
                                 'question_provenance': {'evidence': 'author_released_training_example',
                                                         'record_id': 'en_dis_panoramic_2707_Insufficient '
                                                                      'Space for Primary Tooth Eruption',
                                                         'revision': '9463edb2af47f64510b0681efc20be6ecf870955',
                                                         'file': 'data/inst_data_2nd_train.json',
                                                         'source_modality': 'PAN',
                                                         'pan_support': 'arXiv:2509.23344 Figure 1a'},
                                 'location_reference_pattern': '\\binsufficient space\\b'},
 'caries': {'name': 'Caries',
            'questions': ['Examine the images to determine if there is the presence of caries.'],
            'question_provenance': {'evidence': 'author_released_training_example',
                                    'record_id': 'en_dis_panoramic_4414_Caries',
                                    'revision': '9463edb2af47f64510b0681efc20be6ecf870955',
                                    'file': 'data/inst_data_2nd_train.json',
                                    'source_modality': 'PAN',
                                    'pan_support': 'arXiv:2509.23344 Figure 1a'},
            'location_reference_pattern': '\\bcaries\\b'},
 'calculus': {'name': 'Calculus',
              'questions': ['Evaluate the images to confirm if there is calculus disease?'],
              'question_provenance': {'evidence': 'author_released_training_example',
                                      'record_id': 'en_dis_upper_1137_Calculus',
                                      'revision': '9463edb2af47f64510b0681efc20be6ecf870955',
                                      'file': 'data/inst_data_2nd_train.json',
                                      'source_modality': 'UPP',
                                      'pan_support': 'arXiv:2509.23344 Figure 1a'},
              'location_reference_pattern': '\\bcalculus\\b'}}
CONDITIONS = tuple(TASKS)
LABELS = {key: task["name"] for key, task in TASKS.items()}
TRAINED = CONDITIONS
MAX_PHRASINGS = 1
# Deprecated empty exports for readers of historical artifacts; never dispatch tasks.
EXTRA_TASKS = ()
UNTRAINED_LABELS = {}

# The nine location descriptors DentVLM writes in its rationale (Supplementary Note S1), in
# the order of the authors' scorer, and the six dental-arch cells they map onto. "left" and
# "right" are DentVLM's own words: Table S6 defines its "left posterior region" as FDI
# quadrants 1 and 4, the patient's right, which is the left side of a panoramic as displayed.
# LEFT_IS_IMAGE_LEFT is the Table S6 source-frame convention for reproducible scoring.
# Figure 1 conflicts with this convention: patient laterality is UNVERIFIED.
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
    return tuple(TASKS[task]["questions"])


def task_name(task: str) -> str:
    return TASKS[task]["name"] if task in TASKS else task.replace("_", " ")


def region_question(task: str, cell: str, phrasing: int = 0) -> str:
    raise ValueError("Regional questions are retired. Use the fixed PAN questions and rationale locations.")


def condition_tasks(condition: str, ask_untrained: bool = False) -> tuple[str, ...]:
    """Compatibility accessor for the evaluation mapping; never drives inference."""
    from benchmark_schema import CONDITION_TASKS
    return CONDITION_TASKS.get(condition, ())


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
    """Only an unambiguous leading Yes/No is a diagnosis; never search the rationale."""
    line = first_line(text).lstrip("* _")
    match = re.match(r"^(yes|no)\b", line, re.I)
    if match is None or (bool(_YES.search(line)) and bool(_NO.search(line))):
        return None
    answer = match.group(1).lower()
    opposite = "no" if answer == "yes" else "yes"
    rest = [v.strip() for v in text.splitlines() if v.strip()][1:]
    if any(re.match(rf"^{opposite}\b", v, re.I) and not (answer == "yes" and extract_regions(v)) for v in rest):
        return None
    return answer


def extract_regions(text: str) -> list[str]:
    """Cells named anywhere in the reply through the nine descriptors (exact, case-insensitive)."""
    low = text.lower()
    found = set()
    for descriptor, cells in DESCRIPTORS.items():
        if descriptor in low:
            found.update(cells)
    return [c for c in CELLS if c in found]


def location_evidence(text: str, task: str) -> list[dict]:
    matches = []
    for sentence in re.split(r"[.!?;\n]+", text):
        low = sentence.lower()
        negated = bool(re.search(r"\b(no|not|without|absent|absence|cannot|could|might|may)\b", low))
        other = any(key != task and re.search(spec["location_reference_pattern"], low)
                    for key, spec in TASKS.items())
        for descriptor, cells in DESCRIPTORS.items():
            for hit in re.finditer(re.escape(descriptor), sentence, re.I):
                matches.append({"text": hit.group(), "descriptor": descriptor, "regions": list(cells),
                                "context": sentence.strip(), "reportable": not negated and not other,
                                "reason": "negated_or_uncertain" if negated else "ambiguous_task" if other else None})
    return matches


def contradicts_positive(text: str, task: str) -> bool:
    """Abstain on explicit unlocalized denials of the very task answered Yes.

    Regional negations are handled separately; they do not undo image-level presence.
    This is a conservative textual guard, not a semantic diagnosis reader.
    """
    denial = r"\b(no evidence|no signs|not observed|not seen|not visible|not present|not detected|absent|does not have)\b"
    for sentence in re.split(r"[.!?;\n]+", text):
        if (re.search(TASKS[task]["location_reference_pattern"], sentence, re.I)
                and re.search(denial, sentence, re.I) and not extract_regions(sentence)):
            return True
    return False


def report_location(cell: str) -> str:
    """Patient laterality is deliberately unresolved until independently validated."""
    row, zone = cell.split("-", 1)
    return f"{row} anterior region" if zone == "anterior" else f"{row} posterior region (side unresolved)"


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
        temperature: float | None = 0.1,
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
        if local and (max_tokens != 512 or temperature != 0.1 or token_param != "max_tokens" or request_options):
            raise ValueError("Local PAN inference requires max_tokens=512, temperature=0.1 and no request overrides")
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
            "request_options": self.request_options, "sampling": SAMPLING if self.local else {},
            "runtime_provenance": getattr(self, "runtime_provenance", {}),
            "system_message": SYSTEM_MESSAGE, "api_call_retries": self.api_call_retries,
        }

    def ask(self, image: str | Path | bytes, question: str) -> dict:
        if self.local and question not in {q for t in TASKS.values() for q in t["questions"]}:
            raise ValueError("Only canonical PAN questions are allowed")
        # The authors' system message exactly once, then image before the canonical question.
        request = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_data_uri(image)}},
                {"type": "text", "text": question},
            ]}],
            **llm_api.generation_fields(self.token_param, self.max_tokens, self.temperature),
        }
        if self.local:
            # Reuse the image KV prefix across the questions of one image; repetition penalty as
            # in the authors' inference script (1.05).
            request["top_p"] = SAMPLING["top_p"]
            request["extra_body"] = {"cache_prompt": self.cache_prompt, **{k: v for k, v in SAMPLING.items() if k != "top_p"}}
        if self.local and self.request_options:
            raise ValueError("Request overrides are disabled for the fixed PAN runtime")
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
    """Fixed PAN protocol. Historical protocols are read as data, never instantiated."""
    profile: str = PROFILE
    location: str = "rationale"

    def __post_init__(self):
        if self.profile != PROFILE or self.location != "rationale":
            raise ValueError("New inference requires pan_training_aligned_v1 with rationale locations.")

    def tasks(self) -> tuple[str, ...]:
        return tuple(TASKS)

    @property
    def ask_untrained(self):
        return False


def vote(answers: list[dict], region_vote: str) -> dict:
    """Presence by majority of the parsed answers; regions from the yes answers.

    An answer whose location could not be read at all carries "regions_unresolved" (only an LLM
    parser can set it) and is left out of the region vote. When every answer that reported the
    finding is unresolved there is no location to report and regions stay None - unresolved, which
    is not the same claim as the empty set "the model named no region".
    """
    parsed = [a["answer"] for a in answers if a["answer"] is not None]
    yes, no = parsed.count("yes"), parsed.count("no")
    presence = "yes" if yes > no else "no" if no > yes else None
    if presence != "yes":
        return {"presence": presence, "regions": None}
    positive = [set(a["regions"]) for a in answers
                if a["answer"] == "yes" and not a.get("regions_unresolved")]
    if not positive:
        return {"presence": "yes", "regions": None}
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


def cell_answers(result: dict, parser=None) -> dict[str, dict[str, str | None]]:
    """{task: {cell: yes/no/None}} from the saved region calls (location "regions"): each region's answer.

    A call that carries its own parsed value is authoritative and is never read again: that value is
    the decision the run accepted, whatever reads the text today. Only a call saved without one is
    reconstructed from its text, and `parser` (a llm_parser.ParserService) may then be used for it -
    but only when it is configured exactly as the run was. A replay under a different parser
    configuration reads with code alone and says so, so an accepted result can never be quietly
    reinterpreted into a different answer.
    """
    service = parser
    if service is not None and service.policy.uses_llm():
        saved = result.get("parser_fingerprint")
        if saved != service.fingerprint():
            mon.monitor("PARSER REPLAY GUARD", f"image={result.get('image_id', '?')}",
                        saved=saved or "none", now=service.fingerprint(),
                        action="reconstructing saved answers with code only")
            service = None
    answers: dict[str, dict] = {}
    for call in result.get("calls") or []:
        if call.get("stage") == "region" and call.get("cell"):
            recovery = call.get("parse_recovery")
            if recovery is not None:
                answer = recovery["value"]
            elif service is None:
                answer = extract_answer(call["text"])
            else:
                answer = service.decision("saved_answer_reconstruction", call["text"],
                                          call.get("question", ""),
                                          truncated=bool(call.get("truncated")),
                                          context=f"image={result.get('image_id', '?')} | "
                                                  f"task={call.get('task')} | cell={call['cell']}").value
            answers.setdefault(call["task"], {})[call["cell"]] = answer
    return answers


COUNT_STATUSES = ("resolved", "partial", "unlocated", "unresolved", "no_location", "not_assessed")


def count_block(presence: str | None, regions: list[str] | None, unresolved=(), location: str = "rationale") -> dict:
    """The occupied-region count of one finding and the status that says what it rests on.

    The count is the size of the deduplicated region set, so several boxes, teeth or mentions in one
    region are one; it is never a tooth or lesion count. "region_count" is an integer only when the
    status is "resolved":

    * "resolved"    - an accepted No is 0 regions; a Yes with every region read is the number named.
    * "partial"     - region questions: some regions answered Yes and at least one stayed unresolved,
                      so `regions` is a confirmed lower bound and the unresolved ones the upper bound.
    * "unlocated"   - the finding was reported but no region was named (a rationale that says
                      nothing about where): the total is unavailable, not a reliable zero.
    * "unresolved"  - the decision, or the location of a reported finding, could not be read.
    * "no_location" - the protocol asked presence only, so there is no region evidence to count.
    """
    if location == "none":
        return {"region_count": None, "count_status": "no_location"}
    if presence is None or (presence == "yes" and regions is None):
        return {"region_count": None, "count_status": "unresolved"}
    if presence == "no":
        return {"region_count": 0, "count_status": "resolved"}
    if unresolved:
        return {"region_count": None, "count_status": "partial"}
    if not regions:
        return {"region_count": None, "count_status": "unlocated"}
    return {"region_count": len(regions), "count_status": "resolved"}


def finding_count(finding: dict, location: str, unresolved=()) -> dict:
    """The count block a saved finding carries, or the one its presence and regions imply.

    A result written before the block existed does not record which regions the region questions
    left unresolved; a caller that reconstructed them from the saved calls passes them as `unresolved`
    so a partial region set is not read as a total.
    """
    if "count_status" in finding:
        return {"region_count": finding["region_count"], "count_status": finding["count_status"]}
    if not finding.get("asked"):
        return {"region_count": None, "count_status": "not_assessed"}
    return count_block(finding["presence"], finding.get("regions"), finding.get("unresolved_regions") or unresolved,
                       location)


def _merge_regions(keys, tasks: dict, presence_field: str, regions_field: str) -> list[str] | None:
    """The deduplicated regions of a finding from the tasks that reported it under `presence_field`.

    A known region stays known even beside a task whose location is unresolved; only when no reporting
    task's location could be read at all do the regions stay None (unresolved, not the empty set).
    """
    named, unresolved, reported = set(), False, False
    for k in keys:
        if tasks[k][presence_field] == "yes":
            reported = True
            cells = tasks[k].get(regions_field)
            if cells is None:  # the task reported it but its location is unresolved
                unresolved = True
            else:
                named.update(cells)
    if not reported or (unresolved and not named):
        return None
    return [c for c in CELLS if c in named]


def _finding(condition: str, tasks: dict, protocol: Protocol, cells: dict | None = None) -> dict:
    """One benchmark finding from its tasks: presence, the deduplicated region set, and its count.

    `cells` ({task: {cell: yes/no/None}}, region questions only) lets the finding record the regions
    no task answered Yes for and at least one left unresolved, which makes its count partial rather
    than a total. The whole-image stage is kept next to the authoritative one (presence and regions
    from the rationales; in rationale mode both stages are the same answers).
    """
    keys = condition_tasks(condition, protocol.ask_untrained)
    if not keys or any(k not in tasks for k in keys):
        return {"asked": False, "tasks": [], "presence": None, "whole_image": None, "whole_image_regions": None,
                "regions": None, "unresolved_regions": [], "region_count": None, "count_status": "not_assessed"}
    presence = _any_yes(tasks[k]["presence"] for k in keys)
    regions = _merge_regions(keys, tasks, "presence", "regions") if protocol.location != "none" else None
    unresolved = []
    if cells is not None and presence == "yes":
        for cell in CELLS:
            votes = [cells[k].get(cell) for k in keys]
            if "yes" not in votes and None in votes:
                unresolved.append(cell)
    whole_image = _any_yes(tasks[k]["whole_image"] for k in keys)
    whole_regions = (_merge_regions(keys, tasks, "whole_image", "whole_image_regions")
                     if protocol.location == "regions" else regions)
    return {"asked": True, "tasks": list(keys), "presence": presence,
            "whole_image": whole_image, "whole_image_regions": whole_regions,
            "regions": regions, "unresolved_regions": unresolved,
            **count_block(presence, regions, unresolved, protocol.location)}


def analyze_image(runner, image_path: str | Path, protocol: Protocol = Protocol(), parser=None) -> dict:
    """Twelve independent canonical requests; failure of a task remains visible."""
    if not isinstance(protocol, Protocol):
        raise ValueError("New inference requires the fixed PAN Protocol")
    if parser is not None and parser.policy.uses_llm():
        raise ValueError("PAN inference uses deterministic parsing only")
    if isinstance(runner, VisionRunner) and not runner.local:
        raise ValueError("The PAN baseline requires the verified local DentVLM runtime")
    path = Path(image_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    calls, tasks, findings = [], {}, {}
    for task in protocol.tasks():
        question = questions_for(task)[0]
        error = None
        try:
            reply = runner.ask(path, question)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            reply = {"text": "", "truncated": False, "finish_reason": "error", "error": error,
                     "cache_hit": False}
        error = error or reply.get("error")
        reply["truncated"] = bool(reply.get("truncated") or reply.get("finish_reason") == "length")
        text = reply.get("text", "")
        answer = None if error or reply.get("truncated") else extract_answer(text)
        contradictory = answer == "yes" and contradicts_positive(text, task)
        if contradictory:
            answer = None
        reason = ("transport_error" if error else "truncated" if reply.get("truncated") else
                  "contradictory_response" if contradictory else
                  "missing_or_ambiguous_decision" if answer is None else None)
        evidence = location_evidence(text, task) if answer == "yes" else []
        regions = extract_regions(text) if answer == "yes" else None
        report_regions = [c for c in CELLS if any(c in m["regions"] and m["reportable"] for m in evidence)]
        location_status = ("unresolved" if answer is None else "not_applicable" if answer == "no" else
                           "located" if report_regions else "unresolved" if evidence else "not_stated")
        recovery = {"attempt": 1, "value": answer, "error": reason, "recovered": False, "status": "exhausted" if reason else "parsed"}
        _record(calls, "presence", task, None, question,
                {**reply, "parse_recovery": recovery, "question_provenance": TASKS[task]["question_provenance"]})
        block = {"name": task_name(task), "question": question, "raw_response": text,
                 "rationale": text, "finish_reason": reply.get("finish_reason"), "presence": answer, "regions": regions,
                 "whole_image": answer, "whole_image_regions": regions,
                 "answers": [{"answer": answer, "regions": regions or [], "truncated": bool(reply.get("truncated"))}],
                 "parse_status": "resolved" if answer else "unresolved", "parse_error": reason,
                 "error": error, "location_matches": evidence, "report_regions": report_regions,
                 "location_status": location_status, "patient_laterality": "unresolved",
                 "question_provenance": TASKS[task]["question_provenance"]}
        tasks[task] = block
        findings[task] = {**block, "asked": True, "tasks": [task], "unresolved_regions": [],
                          **count_block(answer, regions)}
    return {"schema": RESULT_SCHEMA, "profile": PROFILE, "image": str(path.resolve()),
            "image_sha256": digest, "protocol": asdict(protocol), "location_level": "rationale",
            "left_is_image_left": LEFT_IS_IMAGE_LEFT, "patient_laterality": "unresolved",
            "tasks": tasks, "findings": findings, "calls": calls, "call_count": len(calls),
            "inference_call_count": sum(not c.get("cache_hit", False) for c in calls),
            "cache_hit_count": sum(bool(c.get("cache_hit")) for c in calls),
            "parse_recovery": llm_api.parse_recovery_summary(calls), "aggregation_warnings": [],
            "parser_version": PARSER_VERSION, "runtime": runner.settings() if hasattr(runner, "settings") else {}}


def run_config(protocol: Protocol, runner_settings: dict, provenance: dict | None = None, parser=None) -> dict:
    if parser is not None and parser.policy.uses_llm():
        raise ValueError("PAN inference uses deterministic parsing only")
    config = {"schema": RESULT_SCHEMA, "protocol": asdict(protocol), "registry": TASKS,
              "questions": {task: questions_for(task) for task in protocol.tasks()},
              "descriptors": DESCRIPTORS, "cells": CELLS, "system_message": SYSTEM_MESSAGE,
              "parser_version": PARSER_VERSION, "runner": runner_settings, "provenance": provenance or {},
              "findings_version": 3, "patient_laterality": "unresolved"}
    config["hash"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    return config


def result_line(result: dict) -> str:
    values = list(result["findings"].values())
    return " | ".join([f"yes={sum(v.get('presence') == 'yes' for v in values)}",
                       f"no={sum(v.get('presence') == 'no' for v in values)}",
                       f"unresolved={sum(v.get('presence') is None for v in values)}"])


def run_dataset(runner, images: dict[str, str | Path], out_dir: str | Path, protocol: Protocol = Protocol(),
                resume: bool = True, provenance: dict | None = None,
                ledger: "mon.Ledger | None" = None, stop_after: int = 3, parser=None) -> Path:
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
    config = run_config(protocol, runner.settings(), provenance, parser)
    manifest_path = out / "manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("hash") != config["hash"]:
            raise ValueError(f"{out} holds a run with a different configuration; use a new directory.")
    else:
        if any(results_dir.glob("*.json")):
            raise ValueError("Legacy or incompatible results without a manifest; use a new directory")
        manifest_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    todo = [(image_id, Path(p)) for image_id, p in images.items()]
    failures = mon.Ledger(f"{out.parent.name}/{out.name}")
    progress = mon.Progress(len(todo), label=f"{out.parent.name}/{out.name}", unit="image")
    for image_id, path in todo:
        target = results_dir / f"{image_id}.json"
        if resume and target.is_file():
            saved = _load_result_file(target, image_id)
            if saved.get("schema") != RESULT_SCHEMA or saved.get("protocol") != asdict(protocol):
                raise ValueError("Legacy or incompatible result cannot resume a PAN run")
            if saved.get("image_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError(f"{image_id}: source image changed; use a new run directory")
            progress.skip(image_id)
            continue
        with mon.guard(f"{out.name}/{image_id}", failures) as step:
            result = analyze_image(runner, path, protocol=protocol, parser=parser)
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
        parsed = result.get("parser_usage") or {}
        progress.item(image_id, result_line(result), calls=result["inference_call_count"],
                      cache=result["cache_hit_count"] or None, retries=recovery["retry_calls"] or None,
                      unparsed=recovery["unresolved_checks"] or None,
                      parser_calls=sum(row.get("llm_calls", 0) for row in parsed.values()) or None,
                      parser_fallbacks=sum(row.get("fallbacks", 0) for row in parsed.values()) or None,
                      parser_unresolved=sum(row.get("unresolved", 0) for row in parsed.values()) or None)
    log = getattr(runner, "call_log", None)
    # The parser is a second model with its own bill; its line is kept next to the analyzer's,
    # never added into it, so "how many calls did the analyzer make" stays answerable.
    parser_log = getattr(getattr(parser, "model", None), "call_log", None)
    detail = log.line(counts=False) if isinstance(log, mon.CallLog) else ""
    if isinstance(parser_log, mon.CallLog) and parser_log.requests:
        detail = (detail + " | " if detail else "") + f"parser {parser_log.line()}"
    progress.done(detail=detail)
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
    from benchmark_schema import UMFIH_CLASSES
    expected = CONDITIONS if payload.get("schema") == RESULT_SCHEMA else UMFIH_CLASSES
    if payload.get("schema") not in (None, RESULT_SCHEMA):
        raise llm_api.artifact_error(path, "unsupported result schema")
    if not isinstance(findings, dict) or set(findings) != set(expected):
        raise llm_api.artifact_error(path, "incomplete findings schema")
    if not payload.get("schema"):
        payload["legacy_artifact"] = True
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
def describe_cell(cell: str, left_is_image_left: bool | None = None) -> str:
    """Unresolved laterality by default; explicit booleans retain legacy diagnostic formatting."""
    if left_is_image_left is None:
        return report_location(cell)
    row, col = cell.split("-")
    if col == "anterior":
        return f"{row} anterior"
    image_side = col if left_is_image_left else _FLIP[col]
    return f"patient's {row} {_FLIP[image_side]} posterior (image {image_side})"


def protocol_level(result: dict) -> str:
    """The location level a saved result was produced with."""
    return result.get("location_level", LOCATION_LEVELS[0])


def dentist_report(result: dict, counting: bool = False) -> str:
    from report_writer import structured_findings, render_facts
    return render_facts(structured_findings(result, counting=counting))
