"""Experiment table: one dictionary per configuration, so several settings run in one session.

CELL 3 of the notebook holds a list of experiment dictionaries. Each one names itself and lists only
the knobs it changes; everything else comes from DEFAULTS, then from the `shared` dictionary passed
to build(). Every experiment writes into <output_root>/<name>/, so two configurations never mix in
one run directory and the evaluation cell scores them side by side on the same images.

    EXPERIMENTS = xp.build([
        {"name": "base"},
        {"name": "three-phrasings", "phrasings": 3},
        {"name": "crops", "location": "crops"},
        {"name": "gemini", "analyzer": {"provider": "gemini", "model": "gemini-3-pro"}},
        {"name": "dentvlm-local", "backend": "local"},
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

BACKENDS = ("api", "local")
MODEL_SOURCES = ("convert", "local", "hf")
LOCATION_TRUTHS = ("llm", "fdm", "geometry")

DEFAULTS = {
    # Where runs are written. One directory per experiment: <output_root>/<name>/<dataset>/
    "output_root": "/kaggle/working/dental_outputs",

    # Analyzer: the model that answers the task questions.
    "backend": "api",                  # "api": the analyzer spec below | "local": DentVLM through llama.cpp
    "analyzer": {"provider": "openrouter", "model": "qwen/qwen3-vl-235b-a22b-thinking"},
    "max_tokens": 512,                 # the authors' output cap
    "temperature": 0.0,                # None leaves the field out (reasoning models)
    "cache_prompt": True,              # local only: reuse the image KV prefix across one image's questions
    "request_timeout_seconds": 600.0,
    "api_call_retries": 2,             # visible retries for transient API/transport failures
    "smoke_images": 2,                 # raw replies checked before the run; 0 skips the smoke test

    # Protocol: the paper's protocol by default (see dental_pipeline.Protocol).
    "phrasings": 1,                    # 3 = three verbatim wordings per task and a vote
    "region_vote": "union",            # with phrasings > 1: "union" | "majority"
    "location": "rationale",           # "rationale" | "crops" (every cell asked every task) | "none"
    "count_question": False,           # out-of-distribution tooth count for positive findings
    "ask_untrained": False,            # also ask the five UMFIH classes DentVLM has no task for
    "extra_tasks": True,               # residual crown, eruption space, calculus: reported, not scored
    "parse_retries": 1,                # extra attempts per unparseable question

    # Location truth: how ground-truth boxes reach the six cells.
    "evaluate_location": True,
    "location_truth": "llm",           # "llm" (adapter spec) | "fdm" (local DentVLM) | "geometry" (fixed windows)
    "adapter": {"provider": "openai", "model": "gpt-5",
                "token_param": "max_completion_tokens", "temperature": None,
                "max_output_tokens": 8192, "max_boxes_per_call": 12},
    "adapter_fdm_margin": 0.06,        # spotlight margin around the box, fraction of the image size
    "location_parse_retries": 1,
    "location_failure_policy": "geometry",  # "geometry" | "exclude" | "error"

    # Dentist report: a text model turns one image's answers into a classified report.
    "reporter": {"provider": "openai", "model": "gpt-5",
                 "token_param": "max_completion_tokens", "temperature": None, "max_output_tokens": 8192,
                 "include_rationale": False},
    "report_language": "English",
    "report_images": None,             # None = every image with a result; N = only the first N

    # Local DentVLM files and llama.cpp runtime (backend="local"). Where they are built, converted and
    # cached is a machine setting and stays in the notebook; these change what the model is and sees.
    "model_source": "convert",         # "convert" (from the gated checkpoint) | "local" | "hf"
    "gguf_repo_id": "REPLACE/DentVLM-GGUF",   # model_source "hf": your own GGUF repository
    "model_filename": "DentVLM-Q8_0.gguf",
    "mmproj_filename": "DentVLM-mmproj-f16.gguf",
    "n_gpu_layers": 999,
    "ctx_size": 16384,                 # the authors' max input
    "image_max_tokens": 8192,          # = the authors' max_pixels 8192x28x28; 1369 = their 1024x1024 ablation
    "image_min_tokens": None,
}

PROTOCOL_KEYS = ("phrasings", "region_vote", "location", "count_question", "ask_untrained", "extra_tasks",
                 "parse_retries")
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

    for value, allowed, knob in ((cfg["backend"], BACKENDS, "backend"),
                                 (cfg["model_source"], MODEL_SOURCES, "model_source"),
                                 (cfg["location_truth"], LOCATION_TRUTHS, "location_truth")):
        if value not in allowed:
            raise ValueError(f"{name}: {knob} must be one of {allowed}, got {value!r}")
    if cfg["location_truth"] == "fdm" and cfg["backend"] != "local":
        raise ValueError(f"{name}: location_truth 'fdm' needs backend 'local' (DentVLM answers the questions)")
    for knob in ("max_tokens", "ctx_size"):
        if type(cfg[knob]) is not int or cfg[knob] <= 0:
            raise ValueError(f"{name}: {knob} must be a positive integer, got {cfg[knob]!r}")
    if type(cfg["smoke_images"]) is not int or cfg["smoke_images"] < 0:
        raise ValueError(f"{name}: smoke_images must be a non-negative integer")
    if cfg["report_images"] is not None and (type(cfg["report_images"]) is not int or cfg["report_images"] <= 0):
        raise ValueError(f"{name}: report_images must be None or a positive integer")
    if cfg["location_failure_policy"] not in ("geometry", "exclude", "error"):
        raise ValueError(f"{name}: location_failure_policy must be 'geometry', 'exclude' or 'error'")
    if cfg["backend"] == "api":
        _check_spec(cfg["analyzer"], "analyzer", name)
    if cfg["evaluate_location"] and cfg["location_truth"] == "llm":
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
            "model_source": cfg["model_source"], "ctx_size": cfg["ctx_size"],
            "image_max_tokens": cfg["image_max_tokens"], "image_min_tokens": cfg["image_min_tokens"], **extra}


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
    """The GGUF files a local experiment needs; experiments sharing them convert or download once."""
    return (cfg["model_source"], cfg["gguf_repo_id"], cfg["model_filename"], cfg["mmproj_filename"])


def server_key(cfg: dict) -> tuple:
    """The llama.cpp settings a local experiment needs; an unchanged key reuses the running server."""
    return tuple(cfg[k] for k in SERVER_KEYS)


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:8]


def truth_dir(cfg: dict, dataset: str) -> Path:
    """Where this experiment's adapted location truth lives, under the shared output root.

    Keyed by the adapter configuration, not by the experiment, so every experiment using the same
    adapter reads the same translated boxes instead of paying for them again.
    """
    if cfg["location_truth"] == "llm":
        key = llm_api.public(cfg["adapter"])
        name = "llm-" + re.sub(r"[^a-z0-9]+", "-", str(cfg["adapter"]["model"]).lower()).strip("-")
    else:
        key = {"model_file": cfg["model_filename"], "margin": cfg["adapter_fdm_margin"],
               "max_tokens": cfg["max_tokens"], "parse_retries": cfg["location_parse_retries"],
               "policy": cfg["location_failure_policy"]}
        name = "fdm-spotlight"
    return Path(cfg["output_root"], "location_truth", dataset, f"{name}-{_digest(key)}")


# ----------------------------------------------------------------------------
# The three model roles of one experiment
# ----------------------------------------------------------------------------
def runner(cfg: dict, server=None) -> dp.VisionRunner:
    """The analyzer: this experiment's hosted model, or the running local llama.cpp server."""
    if not is_local(cfg):
        return dp.VisionRunner.from_api(cfg["analyzer"], max_tokens=cfg["max_tokens"],
                                        temperature=cfg["temperature"], timeout=cfg["request_timeout_seconds"])
    if server is None:
        raise ValueError(f"{cfg['name']}: backend 'local' needs a started llama.cpp server")
    return dp.VisionRunner(base_url=f"{server.base_url}/v1", model=server.alias, max_tokens=cfg["max_tokens"],
                           temperature=cfg["temperature"], timeout=cfg["request_timeout_seconds"], local=True,
                           cache_prompt=cfg["cache_prompt"], api_call_retries=cfg["api_call_retries"])


def location_adapter(cfg: dict, runner=None):
    """The location-truth adapter, or None when the fixed windows are used (or location is not scored)."""
    if not cfg["evaluate_location"] or cfg["location_truth"] == "geometry":
        return None
    if cfg["location_truth"] == "llm":
        return la.LLMAdapter.from_api(cfg["adapter"], timeout=cfg["request_timeout_seconds"])
    return la.FdmAdapter(runner, margin=cfg["adapter_fdm_margin"], parse_retries=cfg["location_parse_retries"],
                         failure_policy=cfg["location_failure_policy"])


def report_writer(cfg: dict) -> rw.ReportWriter:
    return rw.ReportWriter.from_api(cfg["reporter"], language=cfg["report_language"],
                                    timeout=cfg["request_timeout_seconds"])


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
        varying = ["backend", "analyzer", "phrasings", "location", "count_question", "ask_untrained"]
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
