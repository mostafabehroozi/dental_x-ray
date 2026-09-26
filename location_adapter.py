"""Translate ground-truth boxes into the regions DentVLM names, so location can be scored.

Ground truth is numeric (YOLO boxes); DentVLM reports six dental-arch cells. Scoring
location means deciding which cells each true box occupies. Fixed image windows
(dental_eval.geometric_regions) are a crude way to do that: the midline, the canine
line and the occlusal plane move with patient positioning and the shape of the arch.
DentVLM's authors built their own location labels the anatomical way, box -> nearest
teeth -> tooth-region mapping (Methods 4.2). This module does the same with a model
in the loop:

* LLMAdapter (recommended): numbered boxes are drawn on the radiograph and a strong
  vision-language API model classifies each one into the eight units of the dental
  arch, FDI quadrant x {anterior, posterior}, plus the FDI tooth positions it covers.
  Units are the finest division the six cells are made of (the two anterior units
  merge into one cell), so the mapping to DentVLM's vocabulary is deterministic
  (dental_pipeline.unit_cell) and the same adapter output also serves a quadrant
  vocabulary. One call per image (chunked for crowded images); strict JSON back.
* AreaAdapter: the same kind of model, asked once per image about the untouched radiograph -
  not about the findings. It returns where the six cells lie in THIS image as normalized areas,
  because that is what patient positioning, arch shape, centring and missing teeth move; Python
  then places every ground-truth box in the area covering most of it (or, when no area touches
  it, the nearest one). The model never sees a box, so it cannot classify a finding, and the
  placement is deterministic and auditable: the areas, the reply and a marked image are saved.
* FdmAdapter (experimental, off by default): DentVLM itself. It has no question about
  a marked region, so the task is split into one in-distribution question per box: a
  "spotlight" copy of the radiograph that shows only the box plus a margin, the
  finding's own Table S7 question, and the location descriptor DentVLM writes in its
  rationale. This puts the truth in the model's own convention, but the image is
  outside its training distribution; boxes it answers "No" to fall back to geometry.
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
from benchmark_schema import LABELS as BENCHMARK_LABELS
import llm_api
import run_monitor as mon

UNITS = dp.UNITS
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


def _canvas(image_path: str | Path, max_side: int):
    """The radiograph as an RGB canvas (longest side <= max_side) with its draw handle, stroke and font."""
    from PIL import Image, ImageDraw

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    scale = min(1.0, max_side / max(image.size))
    if scale < 1.0:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    short = min(image.size)
    return image, ImageDraw.Draw(image), max(2, round(short / 250)), _font(max(14, round(short / 32)))


def _jpeg(image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def draw_boxes(image_path: str | Path, boxes: list[dict], max_side: int = 2048, numbered: bool = True,
               corner_labels: bool = True, color: str | None = None) -> tuple[bytes, int, int, list[list[int]]]:
    """Draw the boxes on a copy of the radiograph (longest side <= max_side).

    Returns JPEG bytes, the drawn width and height, and the boxes in drawn-pixel coordinates
    [x1, y1, x2, y2]. Boxes are numbered 1..n in the given order; corner_labels burns the FDI
    quadrant names into the corners (Q1 top-left, Q2 top-right, Q3 bottom-right, Q4 bottom-left:
    the patient's right is on the viewer's left) so the reader cannot flip sides.
    """
    image, draw, stroke, font = _canvas(image_path, max_side)
    width, height = image.size

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

    return _jpeg(image), width, height, pixel_boxes


def spotlight(image_path: str | Path, box: dict, margin: float = 0.06) -> bytes:
    """Full-frame PNG showing only the box plus a margin (fraction of the image size); the rest is black.

    The frame geometry is kept so DentVLM's position-to-descriptor mapping still applies.
    """
    from PIL import Image

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    width, height = image.size
    x1 = max(0, round((box["xc"] - box["w"] / 2 - margin) * width))
    y1 = max(0, round((box["yc"] - box["h"] / 2 - margin) * height))
    x2 = min(width, round((box["xc"] + box["w"] / 2 + margin) * width))
    y2 = min(height, round((box["yc"] + box["h"] / 2 + margin) * height))
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    canvas.paste(image.crop((x1, y1, x2, y2)), (x1, y1))
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return buffer.getvalue()


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


def _reads_with_llm(parser) -> bool:
    """True when a parser service is present and at least one of its stages calls a model."""
    return parser is not None and parser.policy.uses_llm()


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
                 failure_policy: str = "geometry", call_log: str | None = None, client=None,
                 parser=None) -> None:
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
        # The reader for this adapter's own replies (llm_parser.ParserService); None reads with code.
        self.parser = parser
        self.call_log = mon.CallLog("location", call_log)

    OPTIONS = ("token_param", "temperature", "max_output_tokens", "max_boxes_per_call", "max_side",
               "corner_labels", "request_options", "parse_retries", "api_call_retries", "failure_policy")

    @classmethod
    def from_api(cls, spec: dict, timeout: float = 600.0, client=None, parser=None) -> "LLMAdapter":
        """Adapter for a hosted model. spec = {"provider", "model", ...} as documented in llm_api,
        plus any of the constructor options named in OPTIONS."""
        base_url, api_key = llm_api.resolve(spec)
        options = {k: spec[k] for k in cls.OPTIONS if k in spec}
        return cls(base_url, api_key, spec["model"], timeout=timeout, client=client, parser=parser, **options)

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
                "system_prompt": SYSTEM_PROMPT, "user_prompt": USER_PROMPT,
                **({"parser": self.parser.settings()} if _reads_with_llm(self.parser) else {})}

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

    def _read(self, reply: dict, n_boxes: int, image: str):
        """The adapter's reply as {box id: units/teeth}: the schema check, and the parser when allowed.

        The parser is given the reply, how many boxes were asked about and the valid unit names.
        It is never given the boxes' ground-truth conditions or coordinates, so it cannot infer the
        answer from the truth it is translating.
        """
        def code():
            return parse_units_checked(reply["text"], n_boxes)

        if self.parser is None:
            parsed, error = code()
            return parsed, error, None
        outcome = self.parser.location_json(reply["text"], n_boxes, code=code,
                                            truncated=bool(reply.get("truncated")),
                                            context=f"image={image}")
        return (outcome.value or {}), outcome.error, outcome.record

    def adapt(self, image_path: str | Path, boxes: list[dict], image_id: str | None = None,
              drawn_dir: str | Path | None = None) -> list[dict]:
        """One record per box: regions (cells) from the units, or regions=None when the reply lacked the box."""
        rows: list[dict] = [{"regions": None, "units": None, "teeth": [], "source": None, "raw": None,
                            "attempts": [], "fallback_reason": None} for _ in boxes]
        for start in range(0, len(boxes), self.max_boxes_per_call):
            chunk = boxes[start:start + self.max_boxes_per_call]
            jpeg, width, height, pixels = draw_boxes(image_path, chunk, self.max_side, True, self.corner_labels)
            if drawn_dir and image_id:
                part = f"_{start // self.max_boxes_per_call + 1}" if len(boxes) > self.max_boxes_per_call else ""
                Path(drawn_dir).mkdir(parents=True, exist_ok=True)
                Path(drawn_dir, f"{image_id}{part}.jpg").write_bytes(jpeg)
            lines = [f"{i + 1}. {BENCHMARK_LABELS[box['condition']]} - {pixel}" for i, (box, pixel) in enumerate(zip(chunk, pixels))]
            prompt = USER_PROMPT.format(width=width, height=height, box_lines="\n".join(lines))
            parsed, reply, attempts, parsing = {}, {"text": ""}, [], []
            for attempt in range(self.parse_retries + 1):
                reply = self._ask(jpeg, prompt)
                parsed, error, record = self._read(reply, len(chunk), image_id or Path(image_path).name)
                attempts.append({**reply, "error": error, **({"parsing": record} if record else {})})
                if record:
                    parsing.append(record)
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
                if parsing:
                    row["parsing"] = parsing
                if entry and entry["units"]:
                    row.update(regions=dp.units_to_cells(entry["units"]), units=entry["units"], teeth=entry["teeth"],
                               source="llm")
                elif entry:
                    row.update(units=[], teeth=entry["teeth"])  # answered but unplaceable -> geometry
                if entry is None:
                    row["fallback_reason"] = error or "missing_box"
                    if self.failure_policy == "exclude":
                        row.update(regions=[], source="excluded")
        return rows


# ----------------------------------------------------------------------------
# Area adapter: this image's own cell areas, then deterministic geometry per box
# ----------------------------------------------------------------------------
# The region schema: one anatomical definition per cell, in DentVLM's own words (CELL_DESCRIPTORS,
# the same phrases the scorer matches) plus the anatomy and the image side that name means. The side
# follows LEFT_IS_IMAGE_LEFT, so the convention cannot drift between the prompt and the windows, and
# the flag is part of the adapter settings: flipping it is a different adapter, not a silent re-reading
# of saved areas. Another named set of regions needs this builder and CELLS, not different code.
def region_definitions(left_is_image_left: bool = dp.LEFT_IS_IMAGE_LEFT) -> dict[str, str]:
    definitions = {}
    for cell in dp.CELLS:
        row, col = cell.split("-")
        arch = "maxillary (upper)" if row == "upper" else "mandibular (lower)"
        half = "TOP" if row == "upper" else "BOTTOM"
        behind = ("the maxillary tuberosity behind the last upper molar" if row == "upper"
                  else "the retromolar area, the angle and the ramus of the mandible")
        if col == "anterior":
            what = ("the central incisors, the lateral incisors and the canines on BOTH sides of the midline "
                    "(FDI positions 1, 2 and 3)")
            where = f"the MIDDLE of the {half} half of the image, and it crosses the midline"
        else:
            image_side = col if left_is_image_left else ("right" if col == "left" else "left")
            patient_side = "RIGHT" if image_side == "left" else "LEFT"
            what = (f"the premolars and molars on the patient's {patient_side} side, behind the canine "
                    f"(FDI positions 4 to 8), and everything behind them ({behind})")
            where = f"the {half}-{image_side.upper()} of the image"
        definitions[cell] = f"{dp.CELL_DESCRIPTORS[cell]}: in the {arch} arch, {what}. It lies in {where}."
    return definitions


AREA_SYSTEM_PROMPT = (
    "You are an expert oral and maxillofacial radiologist. You read a panoramic dental radiograph and report "
    "where its dental-arch regions lie in that particular image. You reason about the anatomy that is actually "
    "visible and you answer with JSON only, no prose."
)

AREA_PROMPT = """The image is a panoramic dental radiograph (orthopantomogram) in the standard display orientation: the patient's RIGHT side is on the LEFT side of the image and the patient's LEFT side is on the RIGHT side of the image; the maxilla (upper jaw) is at the top and the mandible (lower jaw) at the bottom. "Upper" means the maxillary arch and "lower" the mandibular arch; "anterior" means the incisors and canines around the midline and "posterior" the premolars, molars and everything behind them.

Nothing is drawn on this radiograph. TASK: report where each of these regions lies in THIS image:

{region_lines}

Read the boundaries from the anatomy that is visible here: the midline between the central incisors, where the canines stand on each side, the occlusal plane where the two arches meet, and how far each arch reaches to the side. Patient positioning, the shape of the arch, the centring of the image, missing teeth and a tilted occlusal plane move those boundaries from radiograph to radiograph, so measure them in this image instead of answering with fixed fractions of the frame.

Give each region as a normalized rectangle [x1, y1, x2, y2] that covers all of it: x runs from the left edge (0.0) to the right edge (1.0) and y from the top edge (0.0) to the bottom edge (1.0), with x1 < x2, y1 < y2 and every number between 0 and 1. The rectangle must contain the whole region including the teeth at its edges, and neighbouring regions may overlap where they meet (the anterior region overlaps the posterior ones around the canines). Report every region exactly once and no other name.

Answer with JSON only, exactly in this shape and nothing else, with your four measured numbers in place of x1, y1, x2, y2:
{{"regions": [{shape}]}}"""


def area_prompt() -> str:
    """The one prompt the adapter sends, built from the region schema."""
    lines = "\n".join(f'- "{name}" = {text}' for name, text in region_definitions().items())
    shape = ", ".join('{"region": "%s", "area": [x1, y1, x2, y2]}' % name for name in dp.CELLS)
    return AREA_PROMPT.format(region_lines=lines, shape=shape)


def parse_areas(text: str, regions: tuple[str, ...] = dp.CELLS) -> tuple[dict[str, list[float]], str | None]:
    """{cell: [x1, y1, x2, y2]} in cell order, or ({}, reason) when the set is not complete and valid.

    A reply is accepted only as a whole: every cell once, four numbers each, inside [0, 1] and with a
    positive width and height. A missing, duplicated, unknown, malformed or out-of-range area rejects the
    whole reply, because a partial set would place some boxes by model and the rest by something else
    without saying so.
    """
    payload = _extract_json(text)
    if payload is None:
        return {}, "invalid_json"
    entries = payload.get("regions")
    if not isinstance(entries, list):
        return {}, "regions_must_be_a_list"
    areas: dict[str, list[float]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            return {}, "invalid_region_entry"
        name, area = entry.get("region"), entry.get("area")
        if name not in regions:
            return {}, "unknown_region"
        if name in areas:
            return {}, "duplicate_region"
        if (not isinstance(area, list) or len(area) != 4
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in area)):
            return {}, "invalid_area"
        x1, y1, x2, y2 = (float(v) for v in area)
        if not all(0.0 <= v <= 1.0 for v in (x1, y1, x2, y2)):
            return {}, "area_out_of_range"
        if x2 <= x1 or y2 <= y1:
            return {}, "empty_area"
        areas[name] = [x1, y1, x2, y2]
    if any(name not in areas for name in regions):
        return {}, "missing_region"
    return {name: areas[name] for name in regions}, None


def _corners(box: dict) -> tuple[float, float, float, float]:
    return (box["xc"] - box["w"] / 2, box["yc"] - box["h"] / 2,
            box["xc"] + box["w"] / 2, box["yc"] + box["h"] / 2)


def covered_fraction(box: dict, area: list[float]) -> float:
    """How much of the box lies inside the area, as a fraction of the box (where the finding mostly is)."""
    x1, y1, x2, y2 = _corners(box)
    ax1, ay1, ax2, ay2 = area
    overlap = max(0.0, min(x2, ax2) - max(x1, ax1)) * max(0.0, min(y2, ay2) - max(y1, ay1))
    return overlap / max((x2 - x1) * (y2 - y1), 1e-9)


def gap_to(box: dict, area: list[float]) -> float:
    """Distance between the box and the area, 0 when they touch or overlap."""
    x1, y1, x2, y2 = _corners(box)
    ax1, ay1, ax2, ay2 = area
    dx, dy = max(ax1 - x2, x1 - ax2, 0.0), max(ay1 - y2, y1 - ay2, 0.0)
    return (dx * dx + dy * dy) ** 0.5


def place_box(box: dict, areas: dict[str, list[float]]) -> dict:
    """The one cell of a box: the area covering most of it, or, when none touches it, the nearest area.

    Equal scores fall to the first cell in CELLS order (max and min keep the first of equal values), so
    the same box always lands in the same cell. The scores behind the decision are kept.
    """
    coverage = {name: round(covered_fraction(box, area), 6) for name, area in areas.items()}
    best = max(areas, key=lambda name: coverage[name])
    if coverage[best] > 0:
        return {"region": best, "rule": "overlap", "coverage": coverage, "distance": None}
    distance = {name: round(gap_to(box, area), 6) for name, area in areas.items()}
    return {"region": min(areas, key=lambda name: distance[name]), "rule": "nearest",
            "coverage": coverage, "distance": distance}


def draw_areas(image_path: str | Path, areas: dict[str, list[float]], boxes: list[dict] = (),
               max_side: int = 2048) -> bytes:
    """Audit image: the proposed areas as labelled rectangles with the ground-truth boxes in white.

    It is never sent to the model - the adapter asks about the untouched radiograph - and exists so the
    proposed boundaries and the placements that follow from them can be inspected.
    """
    image, draw, stroke, font = _canvas(image_path, max_side)
    width, height = image.size
    for index, (name, (x1, y1, x2, y2)) in enumerate(areas.items()):
        colour = PALETTE[index % len(PALETTE)]
        left, top = round(x1 * width), round(y1 * height)
        draw.rectangle((left, top, round(x2 * width) - 1, round(y2 * height) - 1), outline=colour, width=stroke)
        _label(draw, name, left + stroke, top + stroke, colour, font)
    for box in boxes:
        x1, y1, x2, y2 = _corners(box)
        draw.rectangle((round(x1 * width), round(y1 * height), round(x2 * width), round(y2 * height)),
                       outline="white", width=max(1, stroke // 2))
    return _jpeg(image)


class AreaAdapter:
    """This image's cell areas from a vision API, then every box placed by geometry in Python.

    One call per image, on the untouched radiograph: the model never sees the ground-truth boxes, so it
    cannot classify a finding - it only says where this patient's six cells lie, which is the part that
    moves with positioning, arch shape, centring and missing teeth. Python then gives each box the area
    covering most of it (or the nearest area when none touches it), so every annotated occurrence is
    placed separately and identical boxes always land in the same cell. The reply is a short strict
    object of numbers and is read by code alone: there is nothing in it for the parser service to
    recover that a retry cannot. A reply that is not a complete valid set of areas is retried and then
    follows failure_policy; the areas, the raw reply and the marked image are saved for audit.
    from_api() and the request options are as in LLMAdapter.
    """

    kind = "areas"

    def __init__(self, base_url: str | None, api_key: str, model: str, token_param: str = "max_tokens",
                 max_output_tokens: int = 4096, temperature: float | None = None, max_side: int = 2048,
                 timeout: float = 600.0, request_options: dict | None = None, parse_retries: int = 1,
                 api_call_retries: int = 2, failure_policy: str = "geometry", call_log: str | None = None,
                 client=None) -> None:
        if token_param not in llm_api.TOKEN_PARAMS:
            raise ValueError(f"token_param must be one of {llm_api.TOKEN_PARAMS}")
        llm_api.validate_parse_retries(parse_retries)
        llm_api.validate_api_retries(api_call_retries)
        if failure_policy not in ("geometry", "exclude", "error"):
            raise ValueError("failure_policy must be 'geometry', 'exclude', or 'error'")
        self.client = client if client is not None else llm_api.connect(base_url, api_key, timeout)
        self.base_url, self.model = base_url, model
        self.token_param, self.max_output_tokens, self.temperature = token_param, max_output_tokens, temperature
        self.max_side = max_side
        self.request_options = dict(request_options or {})
        self.parse_retries, self.api_call_retries, self.failure_policy = parse_retries, api_call_retries, failure_policy
        self.prompt = area_prompt()
        self.call_log = mon.CallLog("location", call_log)

    OPTIONS = ("token_param", "temperature", "max_output_tokens", "max_side", "request_options",
               "parse_retries", "api_call_retries", "failure_policy")

    @classmethod
    def from_api(cls, spec: dict, timeout: float = 600.0, client=None) -> "AreaAdapter":
        base_url, api_key = llm_api.resolve(spec)
        options = {k: spec[k] for k in cls.OPTIONS if k in spec}
        return cls(base_url, api_key, spec["model"], timeout=timeout, client=client, **options)

    @property
    def calls(self) -> int:
        return self.call_log.calls

    @property
    def name(self) -> str:
        return "areas-" + _slug(self.model)

    def settings(self) -> dict:
        return {"kind": self.kind, "model": self.model, "base_url": self.base_url, "token_param": self.token_param,
                "max_output_tokens": self.max_output_tokens, "temperature": self.temperature,
                "max_side": self.max_side, "request_options": self.request_options,
                "parse_retries": self.parse_retries, "api_call_retries": self.api_call_retries,
                "failure_policy": self.failure_policy, "regions": list(dp.CELLS),
                "left_is_image_left": dp.LEFT_IS_IMAGE_LEFT,
                "system_prompt": AREA_SYSTEM_PROMPT, "user_prompt": self.prompt}

    def _ask(self, jpeg: bytes) -> dict:
        request = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": AREA_SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": self.prompt},
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

    def areas_for(self, image_path: str | Path, image_id: str | None = None) -> tuple[dict, str | None, dict, list]:
        """The image's areas, the error that remains, the last reply and every attempt."""
        jpeg, _, _, _ = draw_boxes(image_path, [], self.max_side, numbered=False, corner_labels=False)
        areas, error, reply, attempts = {}, None, {"text": ""}, []
        for attempt in range(self.parse_retries + 1):
            reply = self._ask(jpeg)
            areas, error = parse_areas(reply["text"])
            attempts.append({**reply, "error": error})
            if not error:
                if attempt:
                    llm_api.monitor("LOCATION PARSE RECOVERED", f"image={image_id or Path(image_path).name}",
                                    attempt=attempt + 1)
                break
            llm_api.monitor("LOCATION PARSE WARNING", f"image={image_id or Path(image_path).name}",
                            attempt=f"{attempt + 1}/{self.parse_retries + 1}", reason=error)
            llm_api.failure_details("SYSTEM:\n" + AREA_SYSTEM_PROMPT + "\n\nUSER:\n" + self.prompt, reply["text"])
        return areas, error, reply, attempts

    def adapt(self, image_path: str | Path, boxes: list[dict], image_id: str | None = None,
              drawn_dir: str | Path | None = None) -> list[dict]:
        """One record per box: its cell, the areas it was placed against and how it was placed."""
        areas, error, reply, attempts = self.areas_for(image_path, image_id)
        if error and self.failure_policy == "error":
            raise ValueError(f"region areas remained unparseable for {image_id or image_path}: {error}")
        if error:
            llm_api.monitor("LOCATION FALLBACK", f"image={image_id or Path(image_path).name}",
                            policy=self.failure_policy, reason=error)
        elif drawn_dir and image_id:
            Path(drawn_dir).mkdir(parents=True, exist_ok=True)
            Path(drawn_dir, f"{image_id}.jpg").write_bytes(draw_areas(image_path, areas, boxes, self.max_side))
        rows = []
        for box in boxes:
            row = {"regions": None, "units": None, "teeth": [], "source": None, "raw": reply["text"],
                   "attempts": attempts, "fallback_reason": error, "areas": areas or None, "assignment": None}
            if areas:
                assignment = place_box(box, areas)
                row.update(regions=[assignment["region"]], source="areas", assignment=assignment)
            elif self.failure_policy == "exclude":
                row.update(regions=[], source="excluded")
            rows.append(row)
        return rows


# ----------------------------------------------------------------------------
# FDM adapter: DentVLM itself, one in-distribution question per spotlighted box
# ----------------------------------------------------------------------------
class FdmAdapter:
    """Spotlight the box, ask the finding's own Table S7 question, read the descriptor from the rationale.

    Only findings with a DentVLM task can be placed (a crown-or-bridge box tries the crown question,
    then the bridge question). A "No" answer or a rationale without a descriptor leaves the box to
    geometry. Cropped or masked panoramics are outside the model's image distribution, so this is
    an experiment, not the default.
    """

    kind = "fdm"

    def __init__(self, runner, margin: float = 0.06, parse_retries: int = 0,
                 failure_policy: str = "geometry", parser=None) -> None:
        llm_api.validate_parse_retries(parse_retries)
        if failure_policy not in ("geometry", "exclude", "error"):
            raise ValueError("failure_policy must be 'geometry', 'exclude', or 'error'")
        self.runner, self.margin = runner, margin
        self.parse_retries, self.failure_policy = parse_retries, failure_policy
        self.parser = parser
        self.calls = 0

    def _read(self, reply: dict, question: str, context: str):
        """The decision and the location of one spotlight reply, each through its own parser stage.

        A location is only read when the reply reported the finding: a "No" places no box, so its
        words are never turned into a region. An unresolved location stays unresolved and the box
        falls back to the fixed windows, exactly as an unreadable descriptor always has.
        """
        if self.parser is None:
            return dp.extract_answer(reply["text"]), dp.extract_regions(reply["text"]), None
        records = []
        decision = self.parser.decision("spotlight_decision", reply["text"], question,
                                        truncated=bool(reply.get("truncated")), context=context)
        records.append(decision.record)
        if decision.value != "yes":
            return decision.value, [], records
        located = self.parser.location("spotlight_location", reply["text"], question=question,
                                       truncated=bool(reply.get("truncated")), context=context)
        records.append(located.record)
        return decision.value, list(located.value or []), records

    @property
    def name(self) -> str:
        return "fdm-spotlight"

    def settings(self) -> dict:
        return {"kind": self.kind, "method": "spotlight", "margin": self.margin,
                "parse_retries": self.parse_retries, "failure_policy": self.failure_policy,
                "runner": self.runner.settings(),
                "questions": {task: dp.questions_for(task)[0] for task in dp.TASKS},
                **({"parser": self.parser.settings()} if _reads_with_llm(self.parser) else {})}

    def adapt(self, image_path: str | Path, boxes: list[dict], image_id: str | None = None,
              drawn_dir: str | Path | None = None) -> list[dict]:
        rows = []
        for index, box in enumerate(boxes):
            row = {"regions": None, "units": None, "teeth": [], "source": None, "raw": None,
                   "attempts": [], "fallback_reason": None}
            tasks = dp.condition_tasks(box["condition"])
            if tasks:
                png = spotlight(image_path, box, self.margin)
                if drawn_dir and image_id:
                    Path(drawn_dir).mkdir(parents=True, exist_ok=True)
                    Path(drawn_dir, f"{image_id}_{index + 1}.png").write_bytes(png)
                replies = []
                for task in tasks:
                    question = dp.questions_for(task)[0]
                    answer, cells = None, []
                    for attempt in range(self.parse_retries + 1):
                        effective = question if attempt == 0 else question + "\n\nStart with exactly Yes or No, then state the location."
                        reply = self.runner.ask(png, effective)
                        self.calls += 1
                        answer, cells, records = self._read(reply, effective, f"{image_id} box={index + 1}")
                        error = ("missing_or_ambiguous_decision" if answer is None
                                 else "missing_location" if answer == "yes" and not cells else None)
                        row["attempts"].append({"task": task, "question": effective, **reply, "error": error,
                                                **({"parsing": records} if records else {})})
                        replies.append(f"[{task}] {reply['text']}")
                        if not error:
                            if attempt:
                                llm_api.monitor("LOCATION PARSE RECOVERED", f"image={image_id} box={index + 1}", attempt=attempt + 1)
                            break
                        llm_api.monitor("LOCATION PARSE WARNING", f"image={image_id} box={index + 1}",
                                        attempt=f"{attempt + 1}/{self.parse_retries + 1}", reason=error)
                        llm_api.failure_details(effective, reply["text"])
                    if answer == "yes" and cells:
                        row.update(regions=cells, source="fdm")
                        break
                row["raw"] = "\n".join(replies)
                if row["regions"] is None:
                    row["fallback_reason"] = "parse_exhausted_or_not_localized"
                    if self.failure_policy == "error":
                        raise ValueError(f"location could not be resolved for {image_id} box {index + 1}")
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
    config = {"adapter": adapter.settings(), "units": UNITS, "cells": dp.CELLS}
    config["hash"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    manifest_path = out / "manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("hash") != config["hash"]:
            raise ValueError(f"{out} holds adapted truth from a different adapter configuration; use a new directory.")
    else:
        manifest_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    todo = sorted(gt.items())
    parser = getattr(adapter, "parser", None)
    usage_at_start = parser.usage_snapshot() if parser is not None else None
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
                geometry = [c for c in dp.CELLS if c in ev.geometric_regions(box)]
                record = {"condition": box["condition"], "box": [box["xc"], box["yc"], box["w"], box["h"]],
                          "geometry": geometry, **row}
                if record["regions"] is None:
                    record["regions"], record["source"] = geometry, "geometry"
                records.append(record)
            payload = {"image_id": image_id, "image": entry["path"], "adapter": adapter.name, "boxes": records}
            parser = getattr(adapter, "parser", None)
            if parser is not None:
                payload["parser"] = parser.public()
                payload["parser_fingerprint"] = parser.fingerprint()
                payload["parser_usage"] = parser.usage_since(usage_at_start)
                usage_at_start = parser.usage_snapshot()
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
    detail = log.line(counts=False) if isinstance(log, mon.CallLog) else ""
    if parser is not None and parser.model is not None and parser.model.call_log.requests:
        detail = (detail + " | " if detail else "") + f"parser {parser.model.call_log.line()}"
    progress.done(detail=detail)
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
    """Boxes placed per source, how often the adapter agreed with the fixed windows, multi-cell boxes."""
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
