"""In-distribution DentalGPT pipeline: presence, tooth counts, and quadrant location.

Every text sent to the model is one of two shapes with evidence in the DentalGPT
paper (arXiv 2512.11558):

* Presence: the Panorama-Classification question of Figure 7, one condition per
  call, answered A (True) or B (False). This is the only panoramic skill the
  paper measures (84% accuracy).
* Count: the Figure 1/9 filling-count question, and the same tooth-anchored
  "How many ..." shape for the other countable findings. The model counts
  teeth, so findings whose boxes are regions or devices are presence-only.

Location is never asked in words. For each whole-image positive the same
presence question is sent to overlapping quadrant crops; the quadrant set is
whichever crops answer A. Nothing else (JSON contracts, region wording,
fallback paraphrases, forced zeros) is used.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import mimetypes
import re
import time
from pathlib import Path

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

# Tooth-anchored count questions. The filling question is Figure 1/9 verbatim.
COUNT_QUESTIONS = {
    "dental_implant": "How many dental implants are visualized in the panoramic radiograph?",
    "prosthetic_restoration": "How many teeth in the image have a dental crown or bridge?",
    "dental_filling": "How many visible teeth in the image appear to have dental fillings based on their radiopaque characteristics?",
    "endodontic_treatment": "How many teeth in the image have root canal treatment?",
    "carious_lesion": "How many teeth in the image are suspected to have caries?",
    "impacted_tooth": "How many impacted teeth are visualized in the panoramic radiograph?",
    "periapical_lesion": "How many teeth in the image show signs of a periapical lesion?",
    "root_fragment": "How many residual roots are visualized in the panoramic radiograph?",
    "root_resorption": "How many teeth in the image show root resorption?",
}
COUNTABLE = tuple(c for c in CONDITIONS if c in COUNT_QUESTIONS)  # the other five are presence-only

# The paper says a fixed sentence was appended during RL to request <think> and
# <answer> tags but does not publish it. This is the common VLM-R1 wording and is
# a reconstruction; probe() decides whether it is needed at all.
THINK_SUFFIX = "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
MODES = ("plain", "tagged")

# Crop windows as normalized (left, top, right, bottom). Quadrants use patient-side
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
LOCATION_LEVELS = ("none", "arch", "quadrant")


# ----------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------
def with_mode(question: str, mode: str) -> str:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    return question if mode == "plain" else f"{question}\n\n{THINK_SUFFIX}"


def presence_question(condition: str, mode: str = "plain") -> str:
    return with_mode(PRESENCE_QUESTION.format(label=LABELS[condition]), mode)


def count_question(condition: str, mode: str = "plain") -> str:
    return with_mode(COUNT_QUESTIONS[condition], mode)


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
    """Return {region: png_bytes} for the requested level, optionally cached on disk."""
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
    """One image + one question -> one answer. No paraphrase or sampling retries."""

    def __init__(
        self,
        base_url: str | None = "http://127.0.0.1:8080/v1",
        api_key: str = "local-llama-cpp",
        model: str = "dentalgpt",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        timeout: float = 600.0,
        local: bool = True,
        cache_prompt: bool = True,
        request_options: dict | None = None,
    ) -> None:
        from openai import OpenAI

        kwargs = {"api_key": api_key, "timeout": timeout, "max_retries": 0}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.local = local
        self.cache_prompt = cache_prompt
        self.request_options = dict(request_options or {})
        self.calls = 0

    def settings(self) -> dict:
        return {
            "model": self.model, "max_tokens": self.max_tokens, "temperature": self.temperature,
            "local": self.local, "cache_prompt": self.cache_prompt, "request_options": self.request_options,
        }

    def ask(self, image: str | Path | bytes, question: str) -> dict:
        request = {
            "model": self.model,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_data_uri(image)}},
                {"type": "text", "text": question},
            ]}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.local:
            # Keep <think>/<answer> text intact, reuse the image KV prefix, and use the same
            # repetition-penalty value as the backbone's generation_config (llama.cpp applies it
            # over the last 64 tokens rather than the whole sequence).
            request["extra_body"] = {"reasoning_format": "none", "cache_prompt": self.cache_prompt,
                                     "repeat_penalty": 1.05, "seed": 0}
        request.update(self.request_options)
        started = time.perf_counter()
        try:
            response = self.client.chat.completions.create(**request)
        except Exception as exc:  # transport or server error, never a format problem: retry once
            print(f"CALL ERROR ({type(exc).__name__}: {exc}); retrying once")
            time.sleep(2.0)
            response = self.client.chat.completions.create(**request)
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        self.calls += 1
        result = {
            "text": (choice.message.content or "").strip(),
            "finish_reason": choice.finish_reason,
            "truncated": choice.finish_reason == "length",
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "latency_seconds": round(time.perf_counter() - started, 3),
        }
        print(f"call {self.calls} | {result['latency_seconds']}s | prompt_tokens={result['prompt_tokens']} "
              f"| completion_tokens={result['completion_tokens']} | finish={result['finish_reason']}")
        return result


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
# Per-image analysis and dataset runs with resume
# ----------------------------------------------------------------------------
def _record(calls: list, stage: str, condition: str, region: str | None, question: str, reply: dict) -> None:
    calls.append({"stage": stage, "condition": condition, "region": region, "question": question, **reply})


def analyze_image(runner, image_path: str | Path, mode: str = "plain", location: str = "quadrant",
                  crop_dir: str | Path | None = None) -> dict:
    """Presence for all 14 findings, counts and crops for positives. Deterministic order."""
    if location not in LOCATION_LEVELS:
        raise ValueError(f"location must be one of {LOCATION_LEVELS}")
    path = Path(image_path)
    calls: list[dict] = []
    findings = {c: {"presence": None, "count": None, "regions": None} for c in CONDITIONS}

    for condition in CONDITIONS:
        question = presence_question(condition, mode)
        reply = runner.ask(path, question)
        _record(calls, "presence", condition, None, question, reply)
        findings[condition]["presence"] = graded(reply, extract_choice)

    positives = [c for c in CONDITIONS if findings[c]["presence"] == "A"]
    for condition in positives:
        if condition in COUNTABLE:
            question = count_question(condition, mode)
            reply = runner.ask(path, question)
            _record(calls, "count", condition, None, question, reply)
            findings[condition]["count"] = graded(reply, extract_count)

    if location != "none" and positives:
        crops = make_crops(path, location, crop_dir)
        for condition in positives:
            findings[condition]["regions"] = {}
        for region, png in crops.items():  # region-major order keeps the image prefix cached
            for condition in positives:
                question = presence_question(condition, mode)
                reply = runner.ask(png, question)
                _record(calls, "region", condition, region, question, reply)
                findings[condition]["regions"][region] = graded(reply, extract_choice)

    return {
        "image": str(path.resolve()),
        "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "mode": mode,
        "location_level": location,
        "findings": findings,
        "calls": calls,
        "call_count": len(calls),
    }


def run_config(mode: str, location: str, runner_settings: dict, provenance: dict | None = None) -> dict:
    """Everything that defines a run; its hash guards resume. provenance = checkpoint/server facts."""
    config = {
        "mode": mode, "location_level": location, "labels": LABELS, "presence_question": PRESENCE_QUESTION,
        "count_questions": COUNT_QUESTIONS, "think_suffix": THINK_SUFFIX if mode == "tagged" else None,
        "crops": CROPS.get(location), "runner": runner_settings, "provenance": provenance or {},
    }
    config["hash"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    return config


def run_dataset(runner, images: dict[str, str | Path], out_dir: str | Path, mode: str = "plain",
                location: str = "quadrant", resume: bool = True, provenance: dict | None = None) -> Path:
    """Analyze every image, one JSON per image, skipping finished ones on resume."""
    out = Path(out_dir)
    results_dir = out / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    config = run_config(mode, location, runner.settings(), provenance)
    manifest_path = out / "manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("hash") != config["hash"]:
            raise ValueError(f"{out} holds a run with a different configuration; use a new directory.")
    else:
        manifest_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    todo = [(image_id, Path(p)) for image_id, p in images.items()]
    for index, (image_id, path) in enumerate(todo, start=1):
        target = results_dir / f"{image_id}.json"
        if resume and target.is_file():
            continue
        print(f"[{index}/{len(todo)}] {image_id}")
        result = analyze_image(runner, path, mode=mode, location=location, crop_dir=out / "crops")
        result["image_id"] = image_id
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.replace(target)
    return out


def load_results(out_dir: str | Path) -> dict[str, dict]:
    results = {}
    for path in sorted(Path(out_dir, "results").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        results[payload.get("image_id", path.stem)] = payload
    return results


def dentist_report(result: dict) -> str:
    """Deterministic plain-text summary of one image result for a dentist."""
    present, absent, unclear = [], [], []
    for condition in CONDITIONS:
        finding = result["findings"][condition]
        label = LABELS[condition]
        if finding["presence"] == "A":
            parts = [label]
            if finding["count"] is not None:
                parts.append(f"count {finding['count']}")
            if finding["regions"] is not None:
                hits = [r for r, a in finding["regions"].items() if a == "A"]
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
