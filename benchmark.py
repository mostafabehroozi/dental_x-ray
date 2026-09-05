from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from openai_compat import create_openai_client
from prompts import CONDITIONS


YOLO_CLASS_TO_CONDITION = {index: condition for index, condition in enumerate(CONDITIONS)}
YOLO_CLASS_NAMES = {
    0: "Implant (IMP)",
    1: "Prosthetic restoration (PRR)",
    2: "Obturation/Filling (OBT)",
    3: "Endodontic treatment/Root canal treatment (END)",
    4: "Carious lesion/Caries (CAR)",
    5: "Bone resorption/Bone loss (BON)",
    6: "Impacted tooth (IMT)",
    7: "Apical periodontitis/Periapical lesion (API)",
    8: "Root fragment/Residual root (ROT)",
    9: "Furcation lesion (FUR)",
    10: "Apical surgery (APS)",
    11: "Root resorption (ROR)",
    12: "Orthodontic device (ORD)",
    13: "Surgical device (SRD)",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
LOCATION_VALUES = {
    "arch": {"upper", "lower", "unknown"},
    "side": {"left", "right", "center", "unknown"},
    "area": {"anterior", "posterior", "unknown"},
}

_YOLO_NAME_ALIASES = {
    "implant": "dental_implant",
    "dental_implant": "dental_implant",
    "prosthetic_restoration": "prosthetic_restoration",
    "prosthesis": "prosthetic_restoration",
    "dental_filling": "dental_filling",
    "filling": "dental_filling",
    "endodontic_treatment": "endodontic_treatment",
    "root_canal_treatment": "endodontic_treatment",
    "carious_lesion": "carious_lesion",
    "caries": "carious_lesion",
    "periodontal_bone_loss": "periodontal_bone_loss",
    "bone_loss": "periodontal_bone_loss",
    "impacted_tooth": "impacted_tooth",
    "periapical_lesion": "periapical_lesion",
    "root_fragment": "root_fragment",
    "furcation_lesion": "furcation_lesion",
    "furcation_involvement": "furcation_lesion",
    "apical_surgery": "apical_surgery",
    "root_resorption": "root_resorption",
    "orthodontic_device": "orthodontic_device",
    "surgical_device": "surgical_device",
}


@dataclass(frozen=True)
class GeometryThresholds:
    arch_split_y: float = 0.5
    center_x_min: float = 0.4
    center_x_max: float = 0.6
    anterior_x_min: float = 0.33
    anterior_x_max: float = 0.67
    image_left_is_patient_right: bool = True

    def __post_init__(self) -> None:
        values = (
            self.arch_split_y,
            self.center_x_min,
            self.center_x_max,
            self.anterior_x_min,
            self.anterior_x_max,
        )
        if any(value < 0 or value > 1 for value in values):
            raise ValueError("Geometry thresholds must be normalized values in [0, 1].")
        if self.center_x_min > self.center_x_max:
            raise ValueError("center_x_min must not exceed center_x_max.")
        if self.anterior_x_min > self.anterior_x_max:
            raise ValueError("anterior_x_min must not exceed anterior_x_max.")


@dataclass(frozen=True)
class YoloBox:
    box_id: str
    class_id: int
    condition: str
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass(frozen=True)
class BenchmarkImage:
    image_id: str
    image_path: Path
    label_path: Path
    boxes: tuple[YoloBox, ...]
    findings: dict[str, bool]
    source_fingerprint: str


@dataclass(frozen=True)
class YoloBenchmark:
    images: dict[str, BenchmarkImage]
    fingerprint: str
    images_dir: Path
    labels_dir: Path

    @property
    def image_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.images))


def _normalized_name(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.lower())).strip("_")


def validate_dataset_yaml(data_yaml: str | Path) -> None:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised only without dependencies
        raise RuntimeError("PyYAML is required when data_yaml is provided.") from exc

    path = Path(data_yaml)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "names" not in payload:
        raise ValueError(f"{path} does not contain a YOLO 'names' list or mapping.")

    names = payload["names"]
    if isinstance(names, list):
        indexed_names = {index: value for index, value in enumerate(names)}
    elif isinstance(names, dict):
        try:
            indexed_names = {int(index): value for index, value in names.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError("YOLO data.yaml class-name keys must be integer IDs.") from exc
    else:
        raise ValueError("YOLO data.yaml 'names' must be a list or mapping.")

    if set(indexed_names) != set(YOLO_CLASS_TO_CONDITION):
        raise ValueError("YOLO data.yaml must define exactly class IDs 0 through 13.")

    mismatches = []
    for class_id, expected in YOLO_CLASS_TO_CONDITION.items():
        normalized = _normalized_name(str(indexed_names[class_id]))
        actual = _YOLO_NAME_ALIASES.get(normalized, normalized)
        if actual != expected:
            mismatches.append(f"{class_id}: {indexed_names[class_id]!r} != {expected!r}")
    if mismatches:
        raise ValueError("YOLO class order does not match the project ontology: " + "; ".join(mismatches))


def count_yolo_class_instances(
    label_file_path: str | Path,
    class_names: Mapping[int, str] = YOLO_CLASS_NAMES,
) -> dict[str, int]:
    """Count repeated YOLO rows by class without interpreting box geometry."""
    label_path = Path(label_file_path)
    if not label_path.is_file():
        raise FileNotFoundError(label_path)

    counts = {name: 0 for name in class_names.values()}
    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(
                f"{label_path}:{line_number} must contain: "
                "class_id x_center y_center width height"
            )
        try:
            class_id = int(parts[0])
        except ValueError as exc:
            raise ValueError(
                f"Invalid class ID at {label_path}:{line_number}"
            ) from exc
        if class_id not in class_names:
            raise ValueError(
                f"Unknown class ID {class_id} at {label_path}:{line_number}. "
                f"Expected one of {sorted(class_names)}."
            )
        counts[class_names[class_id]] += 1
    return counts


def _parse_label_file(label_path: Path, image_id: str) -> tuple[YoloBox, ...]:
    if not label_path.is_file():
        return ()

    boxes: list[YoloBox] = []
    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{label_path}:{line_number} must contain 5 YOLO values.")
        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = (float(value) for value in parts[1:])
        except ValueError as exc:
            raise ValueError(f"{label_path}:{line_number} contains invalid numeric values.") from exc
        if class_id not in YOLO_CLASS_TO_CONDITION:
            raise ValueError(f"{label_path}:{line_number} has unknown class ID {class_id}.")
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
            raise ValueError(f"{label_path}:{line_number} has a box center outside [0, 1].")
        if not (0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"{label_path}:{line_number} has an invalid box width or height.")
        boxes.append(
            YoloBox(
                box_id=f"box_{len(boxes):04d}",
                class_id=class_id,
                condition=YOLO_CLASS_TO_CONDITION[class_id],
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
            )
        )
    return tuple(boxes)


def load_yolo_benchmark(
    images_dir: str | Path,
    labels_dir: str | Path,
    *,
    data_yaml: str | Path | None = None,
) -> YoloBenchmark:
    images_root = Path(images_dir).resolve()
    labels_root = Path(labels_dir).resolve()
    if not images_root.is_dir():
        raise NotADirectoryError(images_root)
    if not labels_root.is_dir():
        raise NotADirectoryError(labels_root)
    if data_yaml is not None:
        validate_dataset_yaml(data_yaml)

    image_paths = sorted(
        path for path in images_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise ValueError(f"No supported images were found under {images_root}.")

    images: dict[str, BenchmarkImage] = {}
    dataset_hash = hashlib.sha256()
    for image_path in image_paths:
        image_id = image_path.stem
        if image_id in images:
            raise ValueError(f"Duplicate benchmark image stem {image_id!r}; image IDs must be unique.")
        relative = image_path.relative_to(images_root)
        label_path = labels_root / relative.with_suffix(".txt")
        boxes = _parse_label_file(label_path, image_id)
        findings = {condition: False for condition in CONDITIONS}
        for box in boxes:
            findings[box.condition] = True

        source_hash = hashlib.sha256()
        source_hash.update(relative.as_posix().encode("utf-8"))
        source_hash.update(image_path.read_bytes())
        source_hash.update(label_path.read_bytes() if label_path.is_file() else b"<missing-label>")
        source_fingerprint = source_hash.hexdigest()
        dataset_hash.update(image_id.encode("utf-8"))
        dataset_hash.update(source_fingerprint.encode("ascii"))
        images[image_id] = BenchmarkImage(
            image_id=image_id,
            image_path=image_path,
            label_path=label_path,
            boxes=boxes,
            findings=findings,
            source_fingerprint=source_fingerprint,
        )

    return YoloBenchmark(
        images=images,
        fingerprint=dataset_hash.hexdigest(),
        images_dir=images_root,
        labels_dir=labels_root,
    )


def geometry_region(box: YoloBox, thresholds: GeometryThresholds, level: int) -> tuple[str, ...]:
    if level not in {1, 2}:
        raise ValueError("Geometry location level must be 1 or 2.")
    arch = "upper" if box.y_center < thresholds.arch_split_y else "lower"
    if thresholds.center_x_min <= box.x_center <= thresholds.center_x_max:
        side = "center"
    else:
        image_side = "left" if box.x_center < thresholds.center_x_min else "right"
        if thresholds.image_left_is_patient_right:
            side = "right" if image_side == "left" else "left"
        else:
            side = image_side
    if level == 1:
        return arch, side
    area = (
        "anterior"
        if thresholds.anterior_x_min <= box.x_center <= thresholds.anterior_x_max
        else "posterior"
    )
    return arch, side, area


def draw_annotated_boxes(image: BenchmarkImage, output_path: str | Path) -> Path:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:  # pragma: no cover - exercised only without dependencies
        raise RuntimeError("Pillow is required for vision-assisted location adaptation.") from exc

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image.image_path) as source:
        rendered = source.convert("RGB")
    draw = ImageDraw.Draw(rendered)
    font = ImageFont.load_default()
    line_width = max(2, round(min(rendered.size) / 300))
    for box in image.boxes:
        left = (box.x_center - box.width / 2) * rendered.width
        top = (box.y_center - box.height / 2) * rendered.height
        right = (box.x_center + box.width / 2) * rendered.width
        bottom = (box.y_center + box.height / 2) * rendered.height
        color = (255, 48 + (box.class_id * 37) % 180, 48 + (box.class_id * 71) % 180)
        draw.rectangle((left, top, right, bottom), outline=color, width=line_width)
        label = f"{box.box_id} {box.condition}"
        try:
            text_box = draw.textbbox((left, top), label, font=font)
        except (AttributeError, ValueError):  # Older Pillow bitmap-font compatibility
            text_box = (left, top, left + max(40, len(label) * 6), top + 12)
        draw.rectangle(text_box, fill=(0, 0, 0))
        draw.text((left, top), label, fill=color, font=font)
    rendered.save(destination, format="PNG")
    return destination.resolve()


def _image_data_uri(image_path: str | Path) -> str:
    path = Path(image_path)
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type if mime_type and mime_type.startswith("image/") else "image/png"
    return f"data:{mime_type};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


class VisionLocationResolver:
    """Small adapter for either an OpenAI-compatible vision LLM or expert-model runner."""

    def __init__(
        self,
        *,
        backend: Literal["llm", "expert_model"],
        model: str,
        client: Any | None = None,
        expert_model_runner: Any | None = None,
        max_tokens: int = 2048,
    ) -> None:
        if backend == "llm" and client is None:
            raise ValueError("client is required for the llm vision backend.")
        if backend == "expert_model" and expert_model_runner is None:
            raise ValueError("expert_model_runner is required for the expert_model backend.")
        self.backend = backend
        self.model = model
        self.client = client
        self.expert_model_runner = expert_model_runner
        self.max_tokens = max_tokens

    @classmethod
    def from_openai_compatible(
        cls,
        *,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 600,
        max_retries: int = 2,
        max_tokens: int = 2048,
    ) -> "VisionLocationResolver":
        return cls(
            backend="llm",
            model=model,
            client=create_openai_client(
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries,
            ),
            max_tokens=max_tokens,
        )

    @classmethod
    def from_expert_model(cls, runner: Any, *, max_tokens: int = 2048) -> "VisionLocationResolver":
        return cls(
            backend="expert_model",
            model=str(getattr(runner, "model_id", "unknown")),
            expert_model_runner=runner,
            max_tokens=max_tokens,
        )

    def resolve(self, annotated_image_path: str | Path, prompt: str) -> str:
        if self.backend == "expert_model":
            result = self.expert_model_runner.ask(
                str(annotated_image_path), prompt, max_tokens=self.max_tokens, temperature=0.0
            )
            return str(result["raw_answer"])

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": _image_data_uri(annotated_image_path)}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=self.max_tokens,
            temperature=0.0,
        )
        return str(completion.choices[0].message.content or "")


def _vision_prompt(image: BenchmarkImage) -> str:
    box_lines = "\n".join(f"- {box.box_id}: {box.condition}" for box in image.boxes)
    return f"""You are adapting known benchmark boxes into coarse dental locations.
The colored rectangles are authoritative finding sites. Do not decide whether a disease exists and do not add, remove, or merge boxes.
For every box ID below, classify only its location using patient anatomy:
- arch: upper, lower, or unknown
- side: left, right, center, or unknown
- area: anterior, posterior, or unknown

Boxes:
{box_lines}

Return exactly one JSON object inside <answer>...</answer> with this shape:
{{"locations":[{{"box_id":"box_0000","arch":"upper","side":"left","area":"posterior"}}]}}
Return every listed box_id exactly once and no other IDs."""


def _extract_json_object(raw_response: str) -> dict[str, Any]:
    answer_matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", raw_response, flags=re.I | re.S)
    candidate = answer_matches[-1] if answer_matches else raw_response
    candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate.strip(), flags=re.I | re.S)
    start, end = candidate.find("{"), candidate.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Vision location resolver did not return a JSON object.")
    try:
        parsed = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError("Vision location resolver returned invalid JSON.") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Vision location response must be a JSON object.")
    return parsed


def _validate_vision_locations(payload: dict[str, Any], image: BenchmarkImage) -> dict[str, dict[str, str]]:
    records = payload.get("locations")
    if not isinstance(records, list):
        raise ValueError("Vision location JSON must contain a 'locations' list.")
    expected = {box.box_id for box in image.boxes}
    normalized: dict[str, dict[str, str]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("box_id"), str):
            raise ValueError("Every vision location must contain a string box_id.")
        box_id = record["box_id"]
        if box_id in normalized:
            raise ValueError(f"Vision location response duplicated {box_id}.")
        values: dict[str, str] = {}
        for field, allowed in LOCATION_VALUES.items():
            value = _normalized_name(str(record.get(field, "unknown")))
            if value not in allowed:
                raise ValueError(f"Vision location {box_id} has invalid {field}={value!r}.")
            values[field] = value
        normalized[box_id] = values
    if set(normalized) != expected:
        missing = sorted(expected - set(normalized))
        extra = sorted(set(normalized) - expected)
        raise ValueError(f"Vision location box IDs do not match; missing={missing}, extra={extra}.")
    return normalized


def prepare_vision_location_cache(
    benchmark: YoloBenchmark,
    resolver: VisionLocationResolver,
    cache_path: str | Path,
    annotated_images_dir: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    cache_file = Path(cache_path)
    annotated_root = Path(annotated_images_dir)
    cached: dict[str, Any] = {}
    if cache_file.is_file() and not force:
        try:
            loaded = json.loads(cache_file.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and loaded.get("schema_version") == 1:
                cached = loaded.get("images", {})
        except (json.JSONDecodeError, OSError):
            cached = {}

    output_images: dict[str, Any] = {}
    for image_id in benchmark.image_ids:
        image = benchmark.images[image_id]
        expected_ids = {box.box_id for box in image.boxes}
        prior = cached.get(image_id) if isinstance(cached, dict) else None
        prior_locations = prior.get("locations", {}) if isinstance(prior, dict) else {}
        reusable = (
            not force
            and isinstance(prior, dict)
            and prior.get("source_fingerprint") == image.source_fingerprint
            and prior.get("backend") == resolver.backend
            and prior.get("model") == resolver.model
            and set(prior_locations) == expected_ids
        )
        if reusable:
            output_images[image_id] = prior
            continue

        annotated_path = annotated_root / f"{image_id}_boxes.png"
        if image.boxes:
            draw_annotated_boxes(image, annotated_path)
            raw_response = resolver.resolve(annotated_path, _vision_prompt(image))
            locations = _validate_vision_locations(_extract_json_object(raw_response), image)
        else:
            annotated_path = image.image_path
            raw_response = ""
            locations = {}
        output_images[image_id] = {
            "source_fingerprint": image.source_fingerprint,
            "backend": resolver.backend,
            "model": resolver.model,
            "annotated_image": str(Path(annotated_path).resolve()),
            "locations": locations,
            "raw_response": raw_response,
        }

    result = {
        "schema_version": 1,
        "dataset_fingerprint": benchmark.fingerprint,
        "resolver": {"backend": resolver.backend, "model": resolver.model},
        "images": output_images,
    }
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def geometry_thresholds_dict(thresholds: GeometryThresholds) -> dict[str, Any]:
    return asdict(thresholds)
