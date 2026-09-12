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
import llm_api

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


def _label(draw, text: str, x: int, y: int, fill: str, font) -> None:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
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
            _, top, _, bottom = draw.textbbox((0, 0), text, font=font)
            text_height = bottom - top + 4
            y = y1 - text_height - stroke if y1 - text_height - stroke >= 0 else y1 + stroke
            _label(draw, text, min(x1, width - 3 * text_height), y, colour, font)
        pixel_boxes.append([x1, y1, x2, y2])

    if corner_labels:
        margin = stroke * 2
        _, top, right, bottom = draw.textbbox((0, 0), "Q4", font=font)
        text_width, text_height = right + 6, bottom - top + 4
        for text, x, y in (("Q1", margin, margin), ("Q2", width - text_width - margin, margin),
                           ("Q4", margin, height - text_height - margin),
                           ("Q3", width - text_width - margin, height - text_height - margin)):
            _label(draw, text, x, y, "white", font)

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue(), width, height, pixel_boxes


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
                 request_options: dict | None = None, client=None) -> None:
        if token_param not in llm_api.TOKEN_PARAMS:
            raise ValueError(f"token_param must be one of {llm_api.TOKEN_PARAMS}")
        self.client = client if client is not None else llm_api.connect(base_url, api_key, timeout)
        self.base_url, self.model = base_url, model
        self.token_param, self.max_output_tokens, self.temperature = token_param, max_output_tokens, temperature
        self.max_boxes_per_call, self.max_side, self.corner_labels = max_boxes_per_call, max_side, corner_labels
        self.request_options = dict(request_options or {})
        self.calls = 0

    OPTIONS = ("token_param", "temperature", "max_output_tokens", "max_boxes_per_call", "max_side",
               "corner_labels", "request_options")

    @classmethod
    def from_api(cls, spec: dict, timeout: float = 600.0, client=None) -> "LLMAdapter":
        """Adapter for a hosted model. spec = {"provider", "model", ...} as documented in llm_api,
        plus any of the constructor options named in OPTIONS."""
        base_url, api_key = llm_api.resolve(spec)
        options = {k: spec[k] for k in cls.OPTIONS if k in spec}
        return cls(base_url, api_key, spec["model"], timeout=timeout, client=client, **options)

    @property
    def name(self) -> str:
        return "llm-" + _slug(self.model)

    def settings(self) -> dict:
        return {"kind": self.kind, "model": self.model, "base_url": self.base_url, "token_param": self.token_param,
                "max_output_tokens": self.max_output_tokens, "temperature": self.temperature,
                "max_boxes_per_call": self.max_boxes_per_call, "max_side": self.max_side,
                "corner_labels": self.corner_labels, "request_options": self.request_options,
                "system_prompt": SYSTEM_PROMPT, "user_prompt": USER_PROMPT}

    def _ask(self, jpeg: bytes, text: str) -> str:
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
        response = self.client.chat.completions.create(**request)
        self.calls += 1
        content = response.choices[0].message.content or ""
        print(f"adapter call {self.calls} | {time.perf_counter() - started:.1f}s | finish={response.choices[0].finish_reason}")
        return content if isinstance(content, str) else str(content)

    def adapt(self, image_path: str | Path, boxes: list[dict], image_id: str | None = None,
              drawn_dir: str | Path | None = None) -> list[dict]:
        """One record per box: regions (cells) from the units, or regions=None when the reply lacked the box."""
        rows: list[dict] = [{"regions": None, "units": None, "teeth": [], "source": None, "raw": None} for _ in boxes]
        for start in range(0, len(boxes), self.max_boxes_per_call):
            chunk = boxes[start:start + self.max_boxes_per_call]
            jpeg, width, height, pixels = draw_boxes(image_path, chunk, self.max_side, True, self.corner_labels)
            if drawn_dir and image_id:
                part = f"_{start // self.max_boxes_per_call + 1}" if len(boxes) > self.max_boxes_per_call else ""
                Path(drawn_dir).mkdir(parents=True, exist_ok=True)
                Path(drawn_dir, f"{image_id}{part}.jpg").write_bytes(jpeg)
            lines = [f"{i + 1}. {dp.LABELS[box['condition']]} - {pixel}" for i, (box, pixel) in enumerate(zip(chunk, pixels))]
            prompt = USER_PROMPT.format(width=width, height=height, box_lines="\n".join(lines))
            parsed, text = {}, ""
            for _attempt in range(2):  # a second try only when the reply is not complete JSON
                text = self._ask(jpeg, prompt)
                parsed = parse_units(text, len(chunk))
                if len(parsed) == len(chunk):
                    break
            for i in range(len(chunk)):
                entry = parsed.get(i + 1)
                row = rows[start + i]
                row["raw"] = text
                if entry and entry["units"]:
                    row.update(regions=dp.units_to_cells(entry["units"]), units=entry["units"], teeth=entry["teeth"],
                               source="llm")
                elif entry:
                    row.update(units=[], teeth=entry["teeth"])  # answered but unplaceable -> geometry
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

    def __init__(self, runner, margin: float = 0.06) -> None:
        self.runner, self.margin = runner, margin
        self.calls = 0

    @property
    def name(self) -> str:
        return "fdm-spotlight"

    def settings(self) -> dict:
        return {"kind": self.kind, "method": "spotlight", "margin": self.margin, "runner": self.runner.settings(),
                "questions": {task: dp.questions_for(task)[0] for task in dp.TASKS}}

    def adapt(self, image_path: str | Path, boxes: list[dict], image_id: str | None = None,
              drawn_dir: str | Path | None = None) -> list[dict]:
        rows = []
        for index, box in enumerate(boxes):
            row = {"regions": None, "units": None, "teeth": [], "source": None, "raw": None}
            tasks = dp.condition_tasks(box["condition"])
            if tasks:
                png = spotlight(image_path, box, self.margin)
                if drawn_dir and image_id:
                    Path(drawn_dir).mkdir(parents=True, exist_ok=True)
                    Path(drawn_dir, f"{image_id}_{index + 1}.png").write_bytes(png)
                replies = []
                for task in tasks:
                    reply = self.runner.ask(png, dp.questions_for(task)[0])
                    self.calls += 1
                    replies.append(f"[{task}] {reply['text']}")
                    cells = dp.extract_regions(reply["text"])
                    if dp.extract_answer(reply["text"]) == "yes" and cells:
                        row.update(regions=cells, source="fdm")
                        break
                row["raw"] = "\n".join(replies)
            rows.append(row)
        return rows


# ----------------------------------------------------------------------------
# Dataset loop with resume, loading, summary
# ----------------------------------------------------------------------------
def adapt_dataset(adapter, gt: dict[str, dict], out_dir: str | Path, resume: bool = True) -> dict[str, dict]:
    """Adapt every image's boxes, one JSON per image under out_dir/boxes, drawn images under out_dir/drawn."""
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
    for index, (image_id, entry) in enumerate(todo, start=1):
        target = boxes_dir / f"{image_id}.json"
        if resume and target.is_file():
            continue
        boxes = entry["boxes"]
        if boxes:
            print(f"[{index}/{len(todo)}] {image_id}: {len(boxes)} boxes")
        rows = adapter.adapt(entry["path"], boxes, image_id, drawn_dir) if boxes else []
        records = []
        for box, row in zip(boxes, rows):
            geometry = [c for c in dp.CELLS if c in ev.geometric_regions(box)]
            record = {"condition": box["condition"], "box": [box["xc"], box["yc"], box["w"], box["h"]],
                      "geometry": geometry, **row}
            if record["regions"] is None:
                record["regions"], record["source"] = geometry, "geometry"
            records.append(record)
        payload = {"image_id": image_id, "image": entry["path"], "adapter": adapter.name, "boxes": records}
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.replace(target)
    return load_adapted(out)


def load_adapted(out_dir: str | Path) -> dict[str, dict]:
    adapted = {}
    for path in sorted(Path(out_dir, "boxes").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        adapted[payload.get("image_id", path.stem)] = payload
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
