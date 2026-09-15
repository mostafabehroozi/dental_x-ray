"""Translate ground-truth boxes into the quadrants DentalGPT localizes in, so location can be scored.

Ground truth is numeric (YOLO boxes); the pipeline localizes a finding as the set of
quadrant crops that answer True. Scoring location means deciding which quadrant windows
each true box occupies. Fixed image fractions are a crude way to do that: the midline and
the occlusal plane move with patient positioning and the shape of the arch. DentVLM's
authors, facing the same problem, built their location labels anatomically (box ->
nearest teeth -> tooth-region mapping). This module does the same with a model in the
loop:

* LLMAdapter (recommended): numbered boxes are drawn on the radiograph and a strong
  vision-language API model classifies each one into the eight units of the dental
  arch, FDI quadrant x {anterior, posterior}, plus the FDI tooth positions it covers.
  Units are the finest division the quadrant and arch windows are made of, so the
  mapping onto the pipeline's names is deterministic (dental_pipeline.unit_region) and
  the same output also serves a six-cell vocabulary. One call per image (chunked for
  crowded images); strict JSON back.
* FdmAdapter (experimental, off by default): DentalGPT itself. It was trained with
  reinforcement learning on multiple-choice questions (and its caption data includes
  annotated figures), so the task is split into two short multiple-choice questions
  per box, asked in the Figure 7 shape on a copy of the image with only that box drawn
  in red: which jaw, and which side of the image. Left and right are asked as image
  sides so the model never has to resolve the patient-side convention; the quadrant
  name follows in Python. An unparseable answer leaves the box to geometry.
* geometry: no model, the fixed windows (the previous behaviour).

adapt_dataset() runs an adapter once per dataset, one JSON per image with the drawn
images kept for audit, and resumes like the model runs. Every box records which
method placed it; dental_eval.apply_adapted attaches the result to the ground truth
and the evaluation summary reports the mix.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import time
from pathlib import Path

import dental_eval as ev
import dental_pipeline as dp
import llm_api
import run_monitor as mon

UNITS = dp.UNITS
QUADRANTS = tuple(dp.CROPS["quadrant"])
PALETTE = ("#ff3b30", "#34c759", "#00c7ff", "#ffcc00", "#ff2d95", "#ff9500", "#bf5af2", "#ffffff")


# ----------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------
def _font(size: int):
    from PIL import ImageFont

    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial Bold.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _text_bbox(draw, text: str, font) -> tuple[int, int, int, int]:
    try:
        return draw.textbbox((0, 0), text, font=font)
    except (AttributeError, ValueError):
        width, height = draw.textsize(text, font=font)
        return 0, 0, width, height


def _label(draw, text: str, x: int, y: int, fill: str, font) -> None:
    left, top, right, bottom = _text_bbox(draw, text, font)
    width, height = right - left, bottom - top
    draw.rectangle((x, y, x + width + 6, y + height + 4), fill=fill)
    draw.text((x + 3 - left, y + 2 - top), text, fill="black", font=font)


def draw_boxes(image_path: str | Path, boxes: list[dict], max_side: int = 2048, numbered: bool = True,
               corner_labels: bool = True, color: str | None = None) -> tuple[bytes, int, int, list[list[int]]]:
    """Draw the boxes on a copy of the radiograph (longest side <= max_side).

    Returns JPEG bytes, the drawn width and height, and the boxes in drawn-pixel coordinates
    [x1, y1, x2, y2]. Boxes are numbered 1..n in the given order; corner_labels burns the FDI
    quadrant names into the corners (Q1 top-left, Q2 top-right, Q3 bottom-right, Q4 bottom-left:
    the patient's right is on the viewer's left) so the reader cannot flip sides.
    """
    from PIL import Image, ImageDraw

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    scale = min(1.0, max_side / max(image.size))
    if scale < 1.0:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    width, height = image.size
    draw = ImageDraw.Draw(image)
    stroke = max(2, round(min(width, height) / 250))
    font = _font(max(14, round(min(width, height) / 32)))

    pixel_boxes = []
    for index, box in enumerate(boxes):
        x1 = max(0, round((box["xc"] - box["w"] / 2) * width))
        y1 = max(0, round((box["yc"] - box["h"] / 2) * height))
        x2 = min(width - 1, round((box["xc"] + box["w"] / 2) * width))
        y2 = min(height - 1, round((box["yc"] + box["h"] / 2) * height))
        colour = color or PALETTE[index % len(PALETTE)]
        draw.rectangle((x1, y1, x2, y2), outline=colour, width=stroke)
        if numbered:
            text = str(index + 1)
            _, top, _, bottom = _text_bbox(draw, text, font)
            text_height = bottom - top + 4
            y = y1 - text_height - stroke if y1 - text_height - stroke >= 0 else y1 + stroke
            _label(draw, text, min(x1, width - 3 * text_height), y, colour, font)
        pixel_boxes.append([x1, y1, x2, y2])

    if corner_labels:
        margin = stroke * 2
        _, top, right, bottom = _text_bbox(draw, "Q4", font)
        text_width, text_height = right + 6, bottom - top + 4
        for text, x, y in (("Q1", margin, margin), ("Q2", width - text_width - margin, margin),
                           ("Q4", margin, height - text_height - margin),
                           ("Q3", width - text_width - margin, height - text_height - margin)):
            _label(draw, text, x, y, "white", font)

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue(), width, height, pixel_boxes


# ----------------------------------------------------------------------------
# LLM adapter (OpenAI-compatible vision API: OpenAI, Gemini's compat endpoint, NIM, ...)
# ----------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert oral and maxillofacial radiologist. You convert bounding-box annotations on "
    "panoramic dental radiographs into anatomical dental-arch regions. You reason carefully about the "
    "anatomy that is actually visible and you answer with JSON only, no prose."
)

USER_PROMPT = """The image is a panoramic dental radiograph (orthopantomogram) in the standard display orientation: the patient's RIGHT side is on the LEFT side of the image and the patient's LEFT side is on the RIGHT side of the image; the maxilla (upper jaw) is at the top and the mandible (lower jaw) at the bottom. The FDI quadrant names are burned into the corners of the image: Q1 = upper right (top-left of the image), Q2 = upper left (top-right), Q3 = lower left (bottom-right), Q4 = lower right (bottom-left).

Numbered coloured boxes have been drawn on the image. Each box is a ground-truth annotation of one finding; its label is listed below. The drawn image is {width} x {height} pixels and the boxes are also given as pixel coordinates [x1, y1, x2, y2] (x from the left edge, y from the top edge):

{box_lines}

TASK: classify every box into the dental-arch UNIT(S) it occupies. A unit is an FDI quadrant plus a zone:
- "anterior" = the positions of the central incisor, lateral incisor and canine of that quadrant (FDI tooth numbers 1, 2, 3);
- "posterior" = the positions of the premolars and molars (FDI 4 to 8) and everything behind them (retromolar area, angle and ramus of the mandible, maxillary tuberosity).
The valid units are exactly: Q1-posterior, Q1-anterior, Q2-anterior, Q2-posterior, Q3-posterior, Q3-anterior, Q4-anterior, Q4-posterior.

Rules:
1. Decide from the anatomy that is visible: the midline between the central incisors, where the canines stand, the occlusal plane between the arches, and the jaws. Do not use fixed fractions of the image; the patient may be rotated, tilted or off-centre and teeth may be missing.
2. Edentulous spaces, implants, root fragments, bone, cysts and plates are classified by the tooth positions they occupy or replace.
3. List every unit that contains a substantial part of the box (about a quarter of its area or more). Most boxes get exactly one unit. A box that crosses the midline, the canine line or the occlusal plane gets two. A box that spans a whole arch or both arches lists every unit it covers.
4. "teeth": the FDI numbers of the tooth positions the box covers (for example [16, 17]); use an empty list if it covers none.
5. Never skip a box and never invent a box. If a box lies entirely outside the dental arches (for example in the condyle or the sinus) and cannot be placed, return an empty "units" list for it.

Answer with JSON only, exactly in this shape and nothing else:
{{"boxes": [{{"id": 1, "units": ["Q1-posterior"], "teeth": [16, 17]}}, {{"id": 2, "units": ["Q3-anterior", "Q4-anterior"], "teeth": [31, 41]}}]}}"""


def _extract_json(text: str) -> dict | None:
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


def parse_units(text: str, n_boxes: int) -> dict[int, dict]:
    """{box id: {"units": [valid units], "teeth": [ints]}} for the ids 1..n_boxes found in the reply."""
    payload = _extract_json(text)
    entries = payload.get("boxes") if payload else None
    parsed: dict[int, dict] = {}
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        try:
            box_id = int(entry.get("id"))
        except (TypeError, ValueError):
            continue
        if not 1 <= box_id <= n_boxes:
            continue
        units = entry.get("units") or []
        teeth = entry.get("teeth") or []
        parsed[box_id] = {
            "units": [u for u in UNITS if isinstance(units, list) and u in units],
            "teeth": [int(t) for t in teeth if isinstance(t, (int, float, str)) and str(t).strip().isdigit()],
        }
    return parsed


def parse_units_checked(text: str, n_boxes: int) -> tuple[dict[int, dict], str | None]:
    payload = _extract_json(text)
    parsed = parse_units(text, n_boxes)
    if payload is None:
        return parsed, "invalid_json"
    entries = payload.get("boxes")
    if not isinstance(entries, list):
        return parsed, "boxes_must_be_a_list"
    ids = [entry.get("id") for entry in entries if isinstance(entry, dict)]
    try:
        normalized_ids = [int(value) for value in ids]
    except (TypeError, ValueError):
        return parsed, "invalid_box_id"
    if len(normalized_ids) != len(set(normalized_ids)):
        return parsed, "duplicate_box_ids"
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("units"), list) or not isinstance(entry.get("teeth"), list):
            return parsed, "invalid_box_schema"
        if any(not isinstance(unit, str) or unit not in UNITS for unit in entry["units"]):
            return parsed, "unknown_unit"
        if any(not isinstance(tooth, (int, float, str)) or not str(tooth).strip().isdigit() for tooth in entry["teeth"]):
            return parsed, "invalid_tooth_number"
    missing = [str(i) for i in range(1, n_boxes + 1) if i not in parsed]
    extras = [str(i) for i in normalized_ids if not 1 <= i <= n_boxes]
    if missing or extras or len(entries) != n_boxes:
        return parsed, "box_id_mismatch"
    return parsed, None


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "-", text).strip("-").lower() or "model"


class LLMAdapter:
    """Numbered boxes on the image -> units per box, from an OpenAI-compatible vision API.

    from_api() builds one from an llm_api spec. token_param: "max_tokens" for most models,
    "max_completion_tokens" for OpenAI reasoning models (GPT-5 family), which also reject a
    temperature (leave it None). Other request fields (reasoning_effort, response_format, ...)
    go through request_options.
    """

    kind = "llm"

    def __init__(self, base_url: str | None, api_key: str, model: str, token_param: str = "max_tokens",
                 max_output_tokens: int = 4096, temperature: float | None = None, max_boxes_per_call: int = 12,
                 max_side: int = 2048, corner_labels: bool = True, timeout: float = 600.0,
                 request_options: dict | None = None, parse_retries: int = 1, api_call_retries: int = 2,
                 failure_policy: str = "geometry", call_log: str | None = None, client=None) -> None:
        if token_param not in llm_api.TOKEN_PARAMS:
            raise ValueError(f"token_param must be one of {llm_api.TOKEN_PARAMS}")
        llm_api.validate_parse_retries(parse_retries)
        llm_api.validate_api_retries(api_call_retries)
        if failure_policy not in ("geometry", "exclude", "error"):
            raise ValueError("failure_policy must be 'geometry', 'exclude', or 'error'")
        self.client = client if client is not None else llm_api.connect(base_url, api_key, timeout)
        self.base_url, self.model = base_url, model
        self.token_param, self.max_output_tokens, self.temperature = token_param, max_output_tokens, temperature
        self.max_boxes_per_call, self.max_side, self.corner_labels = max_boxes_per_call, max_side, corner_labels
        self.request_options = dict(request_options or {})
        self.parse_retries, self.api_call_retries, self.failure_policy = parse_retries, api_call_retries, failure_policy
        self.call_log = mon.CallLog("location", call_log)

    OPTIONS = ("token_param", "temperature", "max_output_tokens", "max_boxes_per_call", "max_side",
               "corner_labels", "request_options", "parse_retries", "api_call_retries", "failure_policy")

    @classmethod
    def from_api(cls, spec: dict, timeout: float = 600.0, client=None) -> "LLMAdapter":
        """Adapter for a hosted model. spec = {"provider", "model", ...} as documented in llm_api,
        plus any of the constructor options named in OPTIONS."""
        base_url, api_key = llm_api.resolve(spec)
        options = {k: spec[k] for k in cls.OPTIONS if k in spec}
        return cls(base_url, api_key, spec["model"], timeout=timeout, client=client, **options)

    @property
    def calls(self) -> int:
        return self.call_log.calls

    @property
    def name(self) -> str:
        return "llm-" + _slug(self.model)

    def settings(self) -> dict:
        return {"kind": self.kind, "model": self.model, "base_url": self.base_url, "token_param": self.token_param,
                "max_output_tokens": self.max_output_tokens, "temperature": self.temperature,
                "max_boxes_per_call": self.max_boxes_per_call, "max_side": self.max_side,
                "corner_labels": self.corner_labels, "request_options": self.request_options,
                "parse_retries": self.parse_retries, "api_call_retries": self.api_call_retries,
                "failure_policy": self.failure_policy,
                "system_prompt": SYSTEM_PROMPT, "user_prompt": USER_PROMPT}

    def _ask(self, jpeg: bytes, text: str) -> dict:
        request = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": dp.image_data_uri(jpeg, "image/jpeg")}},
                ]},
            ],
            **llm_api.generation_fields(self.token_param, self.max_output_tokens, self.temperature),
        }
        request.update(self.request_options)
        started = time.perf_counter()
        normalized = llm_api.call_with_retries(
            lambda: llm_api.chat_reply(self.client.chat.completions.create(**request)),
            self.api_call_retries, f"location model={self.model}")
        return self.call_log.live({**normalized, "latency_seconds": round(time.perf_counter() - started, 3)})

    def adapt(self, image_path: str | Path, boxes: list[dict], image_id: str | None = None,
              drawn_dir: str | Path | None = None) -> list[dict]:
        """One record per box: regions (quadrants) from the units, or regions=None when the reply lacked the box."""
        rows: list[dict] = [{"regions": None, "units": None, "teeth": [], "source": None, "raw": None,
                            "attempts": [], "fallback_reason": None} for _ in boxes]
        for start in range(0, len(boxes), self.max_boxes_per_call):
            chunk = boxes[start:start + self.max_boxes_per_call]
            jpeg, width, height, pixels = draw_boxes(image_path, chunk, self.max_side, True, self.corner_labels)
            if drawn_dir and image_id:
                part = f"_{start // self.max_boxes_per_call + 1}" if len(boxes) > self.max_boxes_per_call else ""
                Path(drawn_dir).mkdir(parents=True, exist_ok=True)
                Path(drawn_dir, f"{image_id}{part}.jpg").write_bytes(jpeg)
            lines = [f"{i + 1}. {dp.LABELS[box['condition']]} - {pixel}" for i, (box, pixel) in enumerate(zip(chunk, pixels))]
            prompt = USER_PROMPT.format(width=width, height=height, box_lines="\n".join(lines))
            parsed, reply, attempts = {}, {"text": ""}, []
            for attempt in range(self.parse_retries + 1):
                reply = self._ask(jpeg, prompt)
                parsed, error = parse_units_checked(reply["text"], len(chunk))
                attempts.append({**reply, "error": error})
                if not error:
                    if attempt:
                        llm_api.monitor("LOCATION PARSE RECOVERED", f"image={image_id or Path(image_path).name}",
                                        attempt=attempt + 1)
                    break
                llm_api.monitor("LOCATION PARSE WARNING", f"image={image_id or Path(image_path).name}",
                                attempt=f"{attempt + 1}/{self.parse_retries + 1}", reason=error)
                llm_api.failure_details("SYSTEM:\n" + SYSTEM_PROMPT + "\n\nUSER:\n" + prompt, reply["text"])
            if error and self.failure_policy == "error":
                raise ValueError(f"location reply remained unparseable for {image_id or image_path}: {error}")
            if error:
                llm_api.monitor("LOCATION FALLBACK", f"image={image_id or Path(image_path).name}",
                                policy=self.failure_policy, reason=error)
            for i in range(len(chunk)):
                entry = parsed.get(i + 1)
                row = rows[start + i]
                row["raw"], row["attempts"] = reply["text"], attempts
                if entry and entry["units"]:
                    row.update(regions=dp.units_to_regions(entry["units"]), units=entry["units"], teeth=entry["teeth"],
                               source="llm")
                elif entry:
                    row.update(units=[], teeth=entry["teeth"])  # answered but unplaceable -> geometry
                if entry is None:
                    row["fallback_reason"] = error or "missing_box"
                    if self.failure_policy == "exclude":
                        row.update(regions=[], source="excluded")
        return rows


# ----------------------------------------------------------------------------
# FDM adapter: DentalGPT itself, two short multiple-choice questions per marked box
# ----------------------------------------------------------------------------
# Figure 7 shape ("Kindly evaluate ... A. ... B. ..."), one fact per question. Sides are asked as
# image sides; UR/LR are the image-left windows (the patient's right).
JAW_QUESTION = ("Kindly evaluate in which jaw the region marked by the red box is located in this image.\n"
                "A. Upper jaw\nB. Lower jaw\nC. Both jaws")
SIDE_QUESTION = ("Kindly evaluate on which side of this image the region marked by the red box is located.\n"
                 "A. Left side of the image\nB. Right side of the image\nC. Both sides, crossing the midline")
JAW_OPTIONS = {"A": ("upper jaw", "upper", "maxilla"), "B": ("lower jaw", "lower", "mandible"), "C": ("both jaws", "both")}
SIDE_OPTIONS = {"A": ("left side", "left"), "B": ("right side", "right"), "C": ("both sides", "both", "midline")}
_JAWS = {"A": ("upper",), "B": ("lower",), "C": ("upper", "lower")}
_SIDES = {"A": ("left",), "B": ("right",), "C": ("left", "right")}
_QUADRANT = {("upper", "left"): "UR", ("upper", "right"): "UL", ("lower", "left"): "LR", ("lower", "right"): "LL"}


def extract_option(text: str, options: dict[str, tuple[str, ...]]) -> str | None:
    """Letter of the chosen option (keys of options) or None. Letters first, then a unique keyword."""
    body = dp.answer_body(text)
    restated = [rf"\b{letter}\s*[.)]\s*{re.escape(words[0])}\b" for letter, words in options.items()]
    if all(re.search(pattern, body, re.I) for pattern in restated):
        # The whole option list was repeated before answering; drop that first copy only.
        for pattern in restated:
            body = re.sub(pattern, " ", body, count=1, flags=re.I)
    letters = "".join(options)
    match = (re.search(rf"(?i:answer|option|choice)\s*(?:is|:)?\s*[\"'*(]*([{letters}])\b", body)
             or re.search(rf"(?:^|[\s(\[*\"'>])([{letters}])(?=[.),:\]*\"'\n]|\s+(?:is|because)\b|\s*$)", body))
    if match:
        return match.group(1)
    low = body.lower()
    hits = [letter for letter, words in options.items() if any(re.search(rf"\b{re.escape(w)}\b", low) for w in words)]
    return hits[0] if len(hits) == 1 else None


class FdmAdapter:
    """Draw one box in red, ask which jaw and which image side (Figure 7 multiple-choice shape).

    Two short calls per box. The quadrant set is the product of the two answers, so a box that
    crosses the midline or the occlusal plane can name two windows. An unparseable answer to
    either question leaves the box to geometry. Drawn boxes are outside the model's training
    images, so this is an experiment, not the default.
    """

    kind = "fdm"

    def __init__(self, runner, mode: str = "plain", parse_retries: int = 0,
                 failure_policy: str = "geometry") -> None:
        if mode not in dp.MODES:
            raise ValueError(f"mode must be one of {dp.MODES}")
        llm_api.validate_parse_retries(parse_retries)
        if failure_policy not in ("geometry", "exclude", "error"):
            raise ValueError("failure_policy must be 'geometry', 'exclude', or 'error'")
        self.runner, self.mode = runner, mode
        self.parse_retries, self.failure_policy = parse_retries, failure_policy
        self.calls = 0

    @property
    def name(self) -> str:
        return "fdm-mcq"

    def settings(self) -> dict:
        return {"kind": self.kind, "method": "mcq", "mode": self.mode,
                "parse_retries": self.parse_retries, "failure_policy": self.failure_policy,
                "runner": self.runner.settings(),
                "questions": [dp.with_mode(JAW_QUESTION, self.mode), dp.with_mode(SIDE_QUESTION, self.mode)]}

    def adapt(self, image_path: str | Path, boxes: list[dict], image_id: str | None = None,
              drawn_dir: str | Path | None = None) -> list[dict]:
        rows = []
        for index, box in enumerate(boxes):
            jpeg, _, _, _ = draw_boxes(image_path, [box], numbered=False, corner_labels=False, color="red")
            if drawn_dir and image_id:
                Path(drawn_dir).mkdir(parents=True, exist_ok=True)
                Path(drawn_dir, f"{image_id}_{index + 1}.jpg").write_bytes(jpeg)
            answers, replies, attempts = [], [], []
            for question, options in ((JAW_QUESTION, JAW_OPTIONS), (SIDE_QUESTION, SIDE_OPTIONS)):
                effective = dp.with_mode(question, self.mode)
                answer = None
                for attempt in range(self.parse_retries + 1):
                    reply = self.runner.ask(jpeg, effective)
                    self.calls += 1
                    answer = dp.graded(reply, lambda t, o=options: extract_option(t, o))
                    attempts.append({"question": effective, **reply, "error": None if answer else "missing_or_ambiguous_option"})
                    replies.append(reply["text"])
                    if answer:
                        if attempt:
                            llm_api.monitor("LOCATION PARSE RECOVERED", f"image={image_id} box={index + 1}", attempt=attempt + 1)
                        break
                    llm_api.monitor("LOCATION PARSE WARNING", f"image={image_id} box={index + 1}",
                                    attempt=f"{attempt + 1}/{self.parse_retries + 1}", reason="missing_or_ambiguous_option")
                    llm_api.failure_details(effective, reply["text"])
                    effective = dp.with_mode(question, self.mode) + "\n\nReturn exactly one option letter."
                answers.append(answer)
            row = {"regions": None, "units": None, "teeth": [], "source": None,
                   "raw": "\n---\n".join(replies), "attempts": attempts, "fallback_reason": None}
            if all(answers):
                names = {_QUADRANT[(jaw, side)] for jaw in _JAWS[answers[0]] for side in _SIDES[answers[1]]}
                row.update(regions=[q for q in QUADRANTS if q in names], source="fdm")
            else:
                row["fallback_reason"] = "parse_exhausted"
                if self.failure_policy == "error":
                    raise ValueError(f"location answer remained unparseable for {image_id} box {index + 1}")
                if self.failure_policy == "exclude":
                    row.update(regions=[], source="excluded")
                llm_api.monitor("LOCATION FALLBACK", f"image={image_id} box={index + 1}", policy=self.failure_policy)
            rows.append(row)
        return rows


# ----------------------------------------------------------------------------
# Dataset loop with resume, loading, summary
# ----------------------------------------------------------------------------
def adapt_dataset(adapter, gt: dict[str, dict], out_dir: str | Path, resume: bool = True,
                  ledger: "mon.Ledger | None" = None, stop_after: int = 3) -> dict[str, dict]:
    """Adapt every image's boxes, one JSON per image under out_dir/boxes, drawn images under out_dir/drawn.

    An image whose adaptation fails is recorded with its complete traceback and the
    loop continues (its boxes then fall back to the fixed windows at evaluation time,
    the same as an unparseable reply); `stop_after` consecutive failures stop the loop.
    """
    out = Path(out_dir)
    boxes_dir, drawn_dir = out / "boxes", out / "drawn"
    boxes_dir.mkdir(parents=True, exist_ok=True)
    config = {"adapter": adapter.settings(), "units": UNITS, "quadrants": QUADRANTS}
    config["hash"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    manifest_path = out / "manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("hash") != config["hash"]:
            raise ValueError(f"{out} holds adapted truth from a different adapter configuration; use a new directory.")
    else:
        manifest_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    todo = sorted(gt.items())
    failures = mon.Ledger(f"location {out.name}")
    progress = mon.Progress(len(todo), label=f"location truth {out.name}", unit="image")
    for image_id, entry in todo:
        target = boxes_dir / f"{image_id}.json"
        if resume and target.is_file():
            _load_adapted_file(target, image_id)  # a corrupt or foreign artifact stops the loop
            progress.skip(image_id)
            continue
        boxes = entry["boxes"]
        records = []
        with mon.guard(f"{out.name}/{image_id}", failures) as step:
            rows = adapter.adapt(entry["path"], boxes, image_id, drawn_dir) if boxes else []
            if len(rows) != len(boxes):
                raise ValueError(f"adapter returned {len(rows)} rows for {len(boxes)} boxes in {image_id}")
            for box, row in zip(boxes, rows):
                geometry = dp.quadrants_to_regions(ev.geometric_regions(box, "quadrant"))
                record = {"condition": box["condition"], "box": [box["xc"], box["yc"], box["w"], box["h"]],
                          "geometry": geometry, **row}
                if record["regions"] is None:
                    record["regions"], record["source"] = geometry, "geometry"
                records.append(record)
            payload = {"image_id": image_id, "image": entry["path"], "adapter": adapter.name, "boxes": records}
            tmp = target.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
            tmp.replace(target)
        if not step.ok:
            if progress.failure(image_id) >= stop_after:
                progress.stop(f"{stop_after} images in a row failed; fix the cause and rerun to resume")
                break
            continue
        sources: dict[str, int] = {}
        for record in records:
            sources[record["source"]] = sources.get(record["source"], 0) + 1
        detail = f"boxes={len(records)}" + (" | " + " ".join(f"{k}={v}" for k, v in sorted(sources.items()))
                                            if sources else "")
        progress.item(image_id, detail, fallbacks=sum(bool(r.get("fallback_reason")) for r in records) or None)
    log = getattr(adapter, "call_log", None)
    progress.done(detail=log.line(counts=False) if isinstance(log, mon.CallLog) else "")
    if failures:
        failures.report(path=out / "failures.json")
        if ledger is not None:
            ledger.entries.extend(failures.entries)
    return load_adapted(out)


def _load_adapted_file(path: Path, expected_id: str | None = None) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        llm_api.monitor("ARTIFACT ERROR", str(path), reason=str(exc))
        raise ValueError(f"invalid adapted-truth artifact {path}: {exc}") from exc
    image_id = payload.get("image_id") if isinstance(payload, dict) else None
    if not isinstance(image_id, str) or image_id != path.stem or (expected_id and image_id != expected_id):
        raise llm_api.artifact_error(path, "adapted image_id does not match filename/expected id")
    if not isinstance(payload.get("boxes"), list):
        raise llm_api.artifact_error(path, "adapted boxes must be a list")
    return payload


def load_adapted(out_dir: str | Path) -> dict[str, dict]:
    adapted = {}
    for path in sorted(Path(out_dir, "boxes").glob("*.json")):
        payload = _load_adapted_file(path)
        if payload["image_id"] in adapted:
            raise llm_api.artifact_error(path, f"duplicate adapted image_id {payload['image_id']!r}")
        adapted[payload["image_id"]] = payload
    return adapted


def summarize(adapted: dict[str, dict]) -> dict:
    """Boxes placed per source, how often the adapter agreed with the fixed windows, multi-quadrant boxes."""
    by_source: dict[str, int] = {}
    total = agree = multi = 0
    for payload in adapted.values():
        for record in payload["boxes"]:
            total += 1
            by_source[record["source"]] = by_source.get(record["source"], 0) + 1
            agree += set(record["regions"]) == set(record["geometry"])
            multi += len(record["regions"]) > 1
    return {"images": len(adapted), "boxes": total, "by_source": dict(sorted(by_source.items())),
            "agreement_with_geometry": round(agree / total, 4) if total else None, "multi_region_boxes": multi}
