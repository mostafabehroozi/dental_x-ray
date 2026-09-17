"""Experiment table: one dictionary per configuration, so several settings run in one session.

CELL 3 of the notebook holds a list of experiment dictionaries. Each one names itself and lists only
the knobs it changes; everything else comes from DEFAULTS, then from the `shared` dictionary passed
to build(). Every experiment writes into <output_root>/<name>/, so two configurations never mix in
one run directory and the evaluation cell scores them side by side on the same images.

    EXPERIMENTS = xp.build([
        {"name": "base"},
        {"name": "separate-questions", "question_form": "separate"},
        {"name": "presence-only", "counting": False},
        {"name": "arch-regions", "region_scheme": "arch"},
        {"name": "gemini", "analyzer": {"provider": "gemini", "model": "gemini-3-pro"}},
        {"name": "dentalgpt-local", "backend": "local"},
    ], shared={"output_root": "/kaggle/working/dental_outputs"})

A dictionary knob (analyzer, adapter, reporter) merges key by key, so an experiment can change the
model and keep the provider's request options; every other knob is replaced. An unknown knob name is
rejected rather than ignored, because a silent typo would cost the whole sweep.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import re
from pathlib import Path

import dental_pipeline as dp
import llm_api
import location_adapter as la
import report_writer as rw
from response_cache import ResponseCache

BACKENDS = ("api", "local")
LOCATION_TRUTHS = ("llm", "areas", "fdm", "geometry")
QUESTION_FORMS = ("auto",) + dp.QUESTION_FORMS
MODES = ("auto",) + dp.MODES

DEFAULTS = {
    # Where runs are written. One directory per experiment: <output_root>/<name>/<dataset>/
    "output_root": "/kaggle/working/dental_outputs",

    # Analyzer: the model that answers the finding questions.
    "backend": "api",                  # "api": the analyzer spec below | "local": DentalGPT through llama.cpp
    "analyzer": {"provider": "openrouter", "model": "qwen/qwen3-vl-235b-a22b-thinking"},
    "max_tokens": 4096,
    "temperature": 0.0,                # None leaves the field out (reasoning models)
    "cache_prompt": True,              # local only: reuse the image KV prefix across one image's questions
    "reuse_local_responses": True,     # exact shared cache under output_root; safe across matching experiments
    "request_timeout_seconds": 600.0,
    "api_call_retries": 2,             # visible retries for transient API/transport failures

    # Protocol: the question shapes (README, "Two levels").
    "mode": "auto",                    # "auto" = probe decides (local); or force "plain" / "tagged"
    "presence_level": "region",        # "overall" | "region"
    "counting": True,                  # False: no count question at all, presence only (count_level and question_form idle)
    "count_level": "region",           # "overall" | "region"
    "region_scheme": "quadrant",       # "quadrant" | "arch"; the region is always named in the question
    "question_form": "auto",           # "auto" = "combined" on api, "separate" on local | "separate" | "combined"
    "parse_retries": 1,                # extra attempts per unparseable finding question
    "probe_images": 2,

    # Location truth: how ground-truth boxes reach the region windows.
    "evaluate_location": True,
    "location_truth": "llm",           # "llm" (units per box) | "areas" (this image's quadrant areas, then geometry)
                                       # | "fdm" (local DentalGPT) | "geometry" (fixed windows). "llm" and "areas"
                                       # both use the ADAPTER spec below.
    "adapter": {"provider": "nvidia", "model": "google/gemma-4-31b-it",
                "token_param": "max_completion_tokens", "temperature": None,
                "max_output_tokens": 8192, "max_boxes_per_call": 12},
    "location_parse_retries": 1,
    "location_failure_policy": "geometry",  # "geometry" | "exclude" | "error"

    # Dentist report: a text model turns one image's findings into a classified report.
    "reporter": {"provider": "nvidia", "model": "google/gemma-4-31b-it",
                 "token_param": "max_completion_tokens", "temperature": None, "max_output_tokens": 8192},
    "report_language": "English",
    "report_images": None,             # None = every image with a result; N = only the first N

    # Local DentalGPT files and llama.cpp runtime (backend="local"). Build and server location are
    # environment settings and stay in the notebook; these change what the model sees.
    "hf_repo_id": "mradermacher/DentalGPT-7B-1026-GGUF",
    "model_filename": "DentalGPT-7B-1026.Q6_K.gguf",
    "mmproj_filename": "DentalGPT-7B-1026.mmproj-f16.gguf",
    "n_gpu_layers": 999,
    "ctx_size": 16384,
    "image_max_tokens": 6144,          # a 2455x1383 panoramic needs ~4400 tokens; llama.cpp caps at 4096
    "image_min_tokens": None,
}

PROTOCOL_KEYS = ("presence_level", "count_level", "region_scheme", "question_form", "parse_retries",
                 "counting")
SERVER_KEYS = ("model_filename", "mmproj_filename", "n_gpu_layers", "ctx_size", "image_max_tokens", "image_min_tokens")
NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*$")


def _merge(base: dict, override: dict) -> dict:
    """Override base; a dictionary knob merges key by key so a role keeps its other options."""
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        merged[key] = {**current, **value} if isinstance(value, dict) and isinstance(current, dict) else value
    return merged


def _check_keys(config: dict, where: str) -> None:
    for key in config:
        if key not in DEFAULTS and key != "name":
            close = difflib.get_close_matches(key, list(DEFAULTS) + ["name"], n=1)
            hint = f"; did you mean {close[0]!r}?" if close else ""
            raise ValueError(f"{where}: unknown knob {key!r}{hint} (see experiments.DEFAULTS)")


def _check_spec(spec, role: str, name: str) -> None:
    if not isinstance(spec, dict) or not spec.get("model"):
        raise ValueError(f"{name}: {role} must be a dictionary with a 'model' (see llm_api)")
    if not spec.get("provider") and not spec.get("base_url"):
        raise ValueError(f"{name}: {role} needs a 'provider' from PROVIDERS, or its own 'base_url'")


def resolve(config: dict, shared: dict | None = None) -> dict:
    """One experiment dictionary merged onto DEFAULTS (and shared) and validated."""
    name = config.get("name")
    if not isinstance(name, str) or not NAME_PATTERN.match(name):
        raise ValueError(f"experiment name {name!r} must be a non-empty directory-safe string")
    _check_keys(config, f"experiment {name!r}")
    cfg = _merge(_merge(DEFAULTS, shared or {}), config)

    for value, allowed, knob in ((cfg["backend"], BACKENDS, "backend"), (cfg["mode"], MODES, "mode"),
                                 (cfg["question_form"], QUESTION_FORMS, "question_form"),
                                 (cfg["location_truth"], LOCATION_TRUTHS, "location_truth")):
        if value not in allowed:
            raise ValueError(f"{name}: {knob} must be one of {allowed}, got {value!r}")
    if cfg["question_form"] == "auto":
        # The combined form is for hosted models; DentalGPT gets its own separate question shapes.
        cfg["question_form"] = "combined" if cfg["backend"] == "api" else "separate"
    if cfg["location_truth"] == "fdm" and cfg["backend"] != "local":
        raise ValueError(f"{name}: location_truth 'fdm' needs backend 'local' (DentalGPT answers the questions)")
    for knob in ("max_tokens", "ctx_size", "probe_images"):
        if type(cfg[knob]) is not int or cfg[knob] <= 0:
            raise ValueError(f"{name}: {knob} must be a positive integer, got {cfg[knob]!r}")
    if cfg["report_images"] is not None and (type(cfg["report_images"]) is not int or cfg["report_images"] <= 0):
        raise ValueError(f"{name}: report_images must be None or a positive integer")
    if type(cfg["reuse_local_responses"]) is not bool:
        raise ValueError(f"{name}: reuse_local_responses must be True or False")
    if cfg["location_failure_policy"] not in ("geometry", "exclude", "error"):
        raise ValueError(f"{name}: location_failure_policy must be 'geometry', 'exclude' or 'error'")
    if cfg["backend"] == "api":
        _check_spec(cfg["analyzer"], "analyzer", name)
    if cfg["evaluate_location"] and cfg["location_truth"] in ("llm", "areas"):
        _check_spec(cfg["adapter"], "adapter", name)
    _check_spec(cfg["reporter"], "reporter", name)
    protocol(cfg)  # the Protocol validates its own knobs

    # Retry and failure settings reach the roles that need them; a spec may override any of them.
    cfg["analyzer"] = {"api_call_retries": cfg["api_call_retries"], **cfg["analyzer"]}
    cfg["adapter"] = {"api_call_retries": cfg["api_call_retries"], "parse_retries": cfg["location_parse_retries"],
                      "failure_policy": cfg["location_failure_policy"], **cfg["adapter"]}
    cfg["reporter"] = {"api_call_retries": cfg["api_call_retries"], **cfg["reporter"]}
    return cfg


def build(experiments: list[dict], shared: dict | None = None) -> list[dict]:
    """Resolve every experiment against DEFAULTS and `shared`, rejecting duplicates and unknown knobs."""
    if not experiments:
        raise ValueError("no experiments: CELL 3 needs at least one configuration dictionary")
    _check_keys(shared or {}, "shared")
    configs = [resolve(config, shared) for config in experiments]
    names = [c["name"] for c in configs]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"experiment names must be unique; repeated: {duplicates}")
    return configs


# ----------------------------------------------------------------------------
# Reading one resolved configuration
# ----------------------------------------------------------------------------
def is_local(cfg: dict) -> bool:
    return cfg["backend"] == "local"


def protocol(cfg: dict) -> dp.Protocol:
    """The Protocol of a resolved configuration (build() has already replaced question_form "auto")."""
    return dp.Protocol(**{k: cfg[k] for k in PROTOCOL_KEYS})


def run_dir(cfg: dict, dataset: str) -> Path:
    return Path(cfg["output_root"]) / cfg["name"] / dataset


def analyzer_name(cfg: dict) -> str:
    return cfg["model_filename"] if is_local(cfg) else cfg["analyzer"]["model"]


def provenance(cfg: dict, **extra) -> dict:
    """What the run manifest hashes besides the protocol: the checkpoint, or the hosted model."""
    if not is_local(cfg):
        return {**llm_api.public(cfg["analyzer"]), **extra}
    return {"model_file": cfg["model_filename"], "mmproj_file": cfg["mmproj_filename"],
            "ctx_size": cfg["ctx_size"], "image_max_tokens": cfg["image_max_tokens"],
            "image_min_tokens": cfg["image_min_tokens"], **extra}


def public(cfg: dict) -> dict:
    """The configuration without any API key, for printing and for experiment.json."""
    return {k: llm_api.public(v) if k in ("analyzer", "adapter", "reporter") else v for k, v in cfg.items()}


def record(cfg: dict) -> Path:
    """Save the resolved configuration next to the experiment's runs."""
    path = Path(cfg["output_root"], cfg["name"], "experiment.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(public(cfg), indent=1, default=str), encoding="utf-8")
    return path


def model_key(cfg: dict) -> tuple:
    """The GGUF files a local experiment needs; experiments sharing them download once."""
    return (cfg["hf_repo_id"], cfg["model_filename"], cfg["mmproj_filename"])


def server_key(cfg: dict) -> tuple:
    """The llama.cpp settings a local experiment needs; an unchanged key reuses the running server."""
    return tuple(cfg[k] for k in SERVER_KEYS)


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:8]


def truth_dir(cfg: dict, dataset: str, mode: str | None = None) -> Path:
    """Where this experiment's adapted location truth lives, under the shared output root.

    Keyed by the adapter configuration, not by the experiment, so every experiment using the same
    adapter reads the same translated boxes instead of paying for them again.
    """
    if cfg["location_truth"] in ("llm", "areas"):
        key = llm_api.public(cfg["adapter"])
        model = re.sub(r"[^a-z0-9]+", "-", str(cfg["adapter"]["model"]).lower()).strip("-")
        name = f"{cfg['location_truth']}-{model}"
    else:
        key = {"model_file": cfg["model_filename"], "mode": mode or cfg["mode"], "max_tokens": cfg["max_tokens"],
               "parse_retries": cfg["location_parse_retries"], "policy": cfg["location_failure_policy"]}
        name = "fdm-mcq"
    return Path(cfg["output_root"], "location_truth", dataset, f"{name}-{_digest(key)}")


# ----------------------------------------------------------------------------
# The three model roles of one experiment
# ----------------------------------------------------------------------------
def _file_identity(path: str | Path) -> dict:
    """Cheap restart-safe identity for a local runtime artifact."""
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def local_response_cache(cfg: dict, server) -> ResponseCache | None:
    """One shared exact-response cache for compatible local experiments."""
    if not cfg["reuse_local_responses"]:
        return None
    namespace = {
        "backend": "local",
        "hf_repo_id": cfg["hf_repo_id"],
        "model": _file_identity(server.model_path),
        "mmproj": _file_identity(server.mmproj_path),
        "llama_server": _file_identity(server.binary),
        "server": {key: cfg[key] for key in SERVER_KEYS},
    }
    return ResponseCache(Path(cfg["output_root"]) / "_response_cache", namespace)


def runner(cfg: dict, server=None) -> dp.VisionRunner:
    """The analyzer: this experiment's hosted model, or the running local llama.cpp server."""
    if not is_local(cfg):
        return dp.VisionRunner.from_api(cfg["analyzer"], max_tokens=cfg["max_tokens"],
                                        temperature=cfg["temperature"], timeout=cfg["request_timeout_seconds"])
    if server is None:
        raise ValueError(f"{cfg['name']}: backend 'local' needs a started llama.cpp server")
    return dp.VisionRunner(base_url=f"{server.base_url}/v1", model=server.alias, max_tokens=cfg["max_tokens"],
                           temperature=cfg["temperature"], timeout=cfg["request_timeout_seconds"], local=True,
                           cache_prompt=cfg["cache_prompt"], api_call_retries=cfg["api_call_retries"],
                           response_cache=local_response_cache(cfg, server))


def location_adapter(cfg: dict, runner=None, mode: str = "plain"):
    """The location-truth adapter, or None when the fixed windows are used (or location is not scored)."""
    if not cfg["evaluate_location"] or cfg["location_truth"] == "geometry":
        return None
    if cfg["location_truth"] == "llm":
        return la.LLMAdapter.from_api(cfg["adapter"], timeout=cfg["request_timeout_seconds"])
    if cfg["location_truth"] == "areas":
        return la.AreaAdapter.from_api(cfg["adapter"], timeout=cfg["request_timeout_seconds"])
    return la.FdmAdapter(runner, mode=mode, parse_retries=cfg["location_parse_retries"],
                         failure_policy=cfg["location_failure_policy"])


def report_writer(cfg: dict) -> rw.ReportWriter:
    return rw.ReportWriter.from_api(cfg["reporter"], language=cfg["report_language"],
                                    timeout=cfg["request_timeout_seconds"])


def resolve_mode(cfg: dict, runner=None, image_paths=(), datasets=()) -> tuple[str, dict | None]:
    """Whether this experiment sends the <think>/<answer> suffix, and the probe behind the answer.

    A mode already recorded in one of the experiment's run manifests, or in its saved probe, is
    reused, so restarting the notebook never flips the mode of a resumed run.
    """
    if cfg["mode"] != "auto":
        return cfg["mode"], None
    if not is_local(cfg):
        return "plain", None  # the suffix is a DentalGPT training artifact; a hosted model gets the bare question
    root = Path(cfg["output_root"], cfg["name"])
    for dataset in datasets:
        manifest = root / dataset / "manifest.json"
        if manifest.is_file():
            return json.loads(manifest.read_text(encoding="utf-8"))["mode"], None
    probe_path = root / "probe.json"
    if probe_path.is_file():
        probe = json.loads(probe_path.read_text(encoding="utf-8"))
        return probe["recommended_mode"], probe
    probe = dp.probe(runner, list(image_paths)[: cfg["probe_images"]], n=cfg["probe_images"])
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    probe_path.write_text(json.dumps(probe, indent=1), encoding="utf-8")
    return probe["recommended_mode"], probe


# ----------------------------------------------------------------------------
# Printing the table
# ----------------------------------------------------------------------------
def _cell(value) -> str:
    if isinstance(value, dict) and value.get("model"):
        return f"{value.get('provider', 'custom')}/{value['model']}"
    return "-" if value is None else str(value)


def table(configs: list[dict]) -> list[dict]:
    """One row per experiment holding only the knobs the experiments disagree on."""
    varying = [k for k in DEFAULTS if len({_digest(public(c)[k]) for c in configs}) > 1]
    if not varying:
        varying = ["backend", "analyzer", "presence_level", "counting", "count_level", "region_scheme", "question_form"]
    return [{"name": c["name"], **{k: _cell(public(c)[k]) for k in varying}} for c in configs]


def show(configs: list[dict]) -> None:
    rows = table(configs)
    columns = list(rows[0])
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in columns}
    print(f"{len(configs)} experiment(s); columns are the knobs they differ on, "
          f"output under {configs[0]['output_root']}")
    print("  ".join(c.ljust(widths[c]) for c in columns))
    for row in rows:
        print("  ".join(str(row[c]).ljust(widths[c]) for c in columns))
