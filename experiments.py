"""Experiment table: one dictionary per configuration, so several settings run in one session.

CELL 3 of the notebook holds a list of experiment dictionaries. Each one names itself and lists only
the knobs it changes; everything else comes from DEFAULTS, then from the `shared` dictionary passed
to build(). Every experiment writes into <output_root>/<name>/, so two configurations never mix in
one run directory and the evaluation cell scores them side by side on the same images.

    EXPERIMENTS = xp.build([
        {"name": "base"},
        {"name": "three-phrasings", "phrasings": 3},
        {"name": "regions", "location": "regions"},
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
import llm_parser as lp
import location_adapter as la
import report_writer as rw
from response_cache import ResponseCache

BACKENDS = ("api", "local")
MODEL_SOURCES = ("convert", "local", "hf")
LOCATION_TRUTHS = ("llm", "areas", "fdm", "geometry")

DEFAULTS = {
    # Where runs are written. One directory per experiment: <output_root>/<name>/<dataset>/
    "output_root": "/kaggle/working/dental_outputs",

    # Analyzer: the model that answers the task questions.
    "backend": "api",                  # "api": the analyzer spec below | "local": DentVLM through llama.cpp
    "analyzer": {"provider": "openrouter", "model": "qwen/qwen3-vl-235b-a22b-thinking"},
    "max_tokens": 512,                 # the authors' output cap
    "temperature": 0.0,                # None leaves the field out (reasoning models)
    "cache_prompt": True,              # local only: reuse the image KV prefix across one image's questions
    "reuse_local_responses": True,     # exact shared cache under output_root for matching DentVLM requests
    "request_timeout_seconds": 600.0,
    "api_call_retries": 2,             # visible retries for transient API/transport failures
    "smoke_images": 2,                 # raw replies checked before the run; 0 skips the smoke test

    # Protocol: the paper's protocol by default (see dental_pipeline.Protocol).
    "phrasings": 1,                    # 3 = three verbatim wordings per task and a vote
    "region_vote": "union",            # with phrasings > 1: "union" | "majority"
    "location": "rationale",           # "rationale" | "regions" (every region named in the question) | "none"
    "ask_untrained": False,            # also ask the five UMFIH classes DentVLM has no task for
    "extra_tasks": True,               # residual crown, eruption space, calculus: reported, not scored
    "parse_retries": 1,                # extra attempts per unparseable question

    # Location truth: how ground-truth boxes reach the six cells.
    "evaluate_location": True,
    "location_truth": "llm",           # "llm" (units per box) | "areas" (this image's cell areas, then geometry)
                                       # | "fdm" (local DentVLM) | "geometry" (fixed windows). "llm" and "areas"
                                       # both use the adapter spec below.
    "adapter": {"provider": "openai", "model": "gpt-5",
                "token_param": "max_completion_tokens", "temperature": None,
                "max_output_tokens": 8192, "max_boxes_per_call": 12},
    "adapter_fdm_margin": 0.06,        # spotlight margin around the box, fraction of the image size
    "location_parse_retries": 1,
    "location_failure_policy": "geometry",  # "geometry" | "exclude" | "error"

    # Dentist report: a text model turns one image's answers into a classified report.
    "reporter": {"provider": "openai", "model": "gpt-5",
                 "token_param": "max_completion_tokens", "temperature": None, "max_output_tokens": 8192,
                 "include_rationale": False,
                 # ON: the report is also given the vote behind each answer (how many of the question's
                 # wordings reported the finding, how many named each region) and describes that
                 # agreement in words. Needs phrasings > 1 to measure anything. It changes the report
                 # only: the predictions, the union/majority vote and the evaluation are untouched.
                 "vote_agreement": False},
    "report_language": "English",
    "report_images": None,             # None = every image with a result; N = only the first N

    # Parser: the model that reads what the other models wrote, when the strict readers cannot.
    # "parser_mode" is the one switch over every stage: "code" reads with code only (this default
    # is exactly the behaviour of every run written before the parser existed), "llm" reads every
    # stage with the parser model, "code_then_llm" tries the strict reader first everywhere, and
    # None hands each stage back to its own setting in "parser_modes" below.
    "parser_mode": "code",
    # Manual mode ("parser_mode": None): one mode per stage. The defaults are
    # llm_parser.DEFAULT_MODES - "code_then_llm" where the strict reader is right whenever it
    # succeeds, "llm" where it can succeed while losing the meaning (see llm_parser.STAGES for the
    # reason behind each one). A dictionary knob merges key by key, so naming one stage here keeps
    # the rest at their defaults.
    "parser_modes": dict(lp.DEFAULT_MODES),
    "parser": {"provider": "openai", "model": "gpt-5",
               "token_param": "max_completion_tokens", "temperature": None,
               "max_output_tokens": 2048},
    "parser_parse_retries": 1,         # extra attempts when the parser's own reply is malformed
    "reuse_parser_responses": True,    # exact shared cache under output_root for parser requests

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

PROTOCOL_KEYS = ("phrasings", "region_vote", "location", "ask_untrained", "extra_tasks", "parse_retries")
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
    for knob in ("reuse_local_responses", "reuse_parser_responses"):
        if type(cfg[knob]) is not bool:
            raise ValueError(f"{name}: {knob} must be True or False")
    llm_api.validate_parse_retries(cfg["parser_parse_retries"])
    lp.validate_mode(cfg["parser_mode"], allow_none=True, where=f"{name}: parser_mode")
    if not isinstance(cfg["parser_modes"], dict):
        raise ValueError(f"{name}: parser_modes must be a dictionary of stage -> mode")
    policy = lp.ParserPolicy(cfg["parser_mode"], cfg["parser_modes"])  # rejects unknown stages and modes
    if policy.uses_llm():
        _check_spec(cfg["parser"], "parser", name)
    if cfg["location_failure_policy"] not in ("geometry", "exclude", "error"):
        raise ValueError(f"{name}: location_failure_policy must be 'geometry', 'exclude' or 'error'")
    if cfg["backend"] == "api":
        _check_spec(cfg["analyzer"], "analyzer", name)
    if cfg["evaluate_location"] and cfg["location_truth"] in ("llm", "areas"):
        _check_spec(cfg["adapter"], "adapter", name)
    _check_spec(cfg["reporter"], "reporter", name)
    if type(cfg["reporter"].get("vote_agreement", False)) is not bool:
        raise ValueError(f"{name}: reporter vote_agreement must be True or False")
    protocol(cfg)  # the Protocol validates its own knobs

    # Retry and failure settings reach the roles that need them; a spec may override any of them.
    cfg["analyzer"] = {"api_call_retries": cfg["api_call_retries"], **cfg["analyzer"]}
    cfg["adapter"] = {"api_call_retries": cfg["api_call_retries"], "parse_retries": cfg["location_parse_retries"],
                      "failure_policy": cfg["location_failure_policy"], **cfg["adapter"]}
    cfg["reporter"] = {"api_call_retries": cfg["api_call_retries"], **cfg["reporter"]}
    cfg["parser"] = {"api_call_retries": cfg["api_call_retries"],
                     "parse_retries": cfg["parser_parse_retries"], **cfg["parser"]}
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


ROLE_KEYS = ("analyzer", "adapter", "reporter", "parser")


def public(cfg: dict) -> dict:
    """The configuration without any API key, for printing and for experiment.json."""
    return {k: llm_api.public(v) if k in ROLE_KEYS else v for k, v in cfg.items()}


def record(cfg: dict) -> Path:
    """Save the resolved configuration next to the experiment's runs.

    The parser modes are saved twice on purpose: as the knobs that were set, and as the modes those
    knobs resolve to after the global override, because the second is what the run actually did.
    """
    path = Path(cfg["output_root"], cfg["name"], "experiment.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = {**public(cfg),
             "parser_resolved_modes": lp.ParserPolicy(cfg["parser_mode"], cfg["parser_modes"]).resolved()}
    path.write_text(json.dumps(saved, indent=1, default=str), encoding="utf-8")
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
    if cfg["location_truth"] in ("llm", "areas"):
        key = llm_api.public(cfg["adapter"])
        model = re.sub(r"[^a-z0-9]+", "-", str(cfg["adapter"]["model"]).lower()).strip("-")
        name = f"{cfg['location_truth']}-{model}"
    else:
        key = {"model_file": cfg["model_filename"], "margin": cfg["adapter_fdm_margin"],
               "max_tokens": cfg["max_tokens"], "parse_retries": cfg["location_parse_retries"],
               "policy": cfg["location_failure_policy"]}
        name = "fdm-spotlight"
    # How the adapter's replies are read is part of what the adapted truth is, so two experiments
    # reading them differently get two directories instead of one they would refuse to share. The
    # key is read from the configuration, never from a built service: naming a directory must not
    # need an API key. The area adapter reads its own reply by code, so no parser belongs in its key.
    policy = lp.ParserPolicy(cfg["parser_mode"], cfg["parser_modes"])
    reader = ({"policy": policy.settings(), "model": llm_api.public(cfg["parser"]),
               "prompts": lp.PROMPT_VERSION} if policy.uses_llm() and cfg["location_truth"] != "areas" else None)
    key = {"adapter": key, **({"parser": reader} if reader else {})}
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
    """One shared exact-response cache for compatible local DentVLM experiments."""
    if not cfg["reuse_local_responses"]:
        return None
    namespace = {
        "backend": "local",
        "model_source": cfg["model_source"],
        "gguf_repo_id": cfg["gguf_repo_id"],
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


def parser(cfg: dict) -> lp.ParserService:
    """The one reader of this experiment, shared by the analyzer run, the adapter and the reporter.

    One service per experiment, so every parser call is counted once, its records sit next to the
    text they read, and one fingerprint describes how the whole experiment read its replies.
    """
    root = Path(cfg["output_root"]) if cfg["reuse_parser_responses"] else None
    return lp.build(cfg["parser"], cfg["parser_mode"], cfg["parser_modes"],
                    timeout=cfg["request_timeout_seconds"], cache_root=root)


def location_adapter(cfg: dict, runner=None, parser=None):
    """The location-truth adapter, or None when the fixed windows are used (or location is not scored)."""
    if not cfg["evaluate_location"] or cfg["location_truth"] == "geometry":
        return None
    if cfg["location_truth"] == "llm":
        return la.LLMAdapter.from_api(cfg["adapter"], timeout=cfg["request_timeout_seconds"], parser=parser)
    if cfg["location_truth"] == "areas":
        # No parser: the areas are a short strict object of numbers, read by code.
        return la.AreaAdapter.from_api(cfg["adapter"], timeout=cfg["request_timeout_seconds"])
    return la.FdmAdapter(runner, margin=cfg["adapter_fdm_margin"], parse_retries=cfg["location_parse_retries"],
                         failure_policy=cfg["location_failure_policy"], parser=parser)


def report_writer(cfg: dict, parser=None) -> rw.ReportWriter:
    return rw.ReportWriter.from_api(cfg["reporter"], language=cfg["report_language"],
                                    timeout=cfg["request_timeout_seconds"], parser=parser)


# ----------------------------------------------------------------------------
# Printing the table
# ----------------------------------------------------------------------------
def _cell(value, knob: str | None = None) -> str:
    if isinstance(value, dict) and value.get("model"):
        return f"{value.get('provider', 'custom')}/{value['model']}"
    if knob == "parser_mode" and value is None:
        return "manual"  # None is manual mode, not "unset"; the stage table below spells it out
    if isinstance(value, dict) and knob == "parser_modes":
        return ", ".join(f"{k}={v}" for k, v in sorted(value.items()) if v != lp.DEFAULT_MODES.get(k)) or "defaults"
    return "-" if value is None else str(value)


def table(configs: list[dict]) -> list[dict]:
    """One row per experiment holding only the knobs the experiments disagree on."""
    varying = [k for k in DEFAULTS if len({_digest(public(c)[k]) for c in configs}) > 1]
    if not varying:
        varying = ["backend", "analyzer", "phrasings", "location", "ask_untrained"]
    return [{"name": c["name"], **{k: _cell(public(c)[k], k) for k in varying}} for c in configs]


def parser_summary(cfg: dict) -> list[str]:
    """How this experiment reads model text: the parser model, and the mode of every stage.

    Printed with the table and saved in experiment.json, because a mode is as much a part of what a
    run measured as the protocol is: the same replies read two ways are two different results.
    """
    policy = lp.ParserPolicy(cfg["parser_mode"], cfg["parser_modes"])
    spec = llm_api.public(cfg["parser"])
    head = (f"parser model: {spec.get('provider', 'custom')}/{spec.get('model')} "
            f"(max_output_tokens={spec.get('max_output_tokens')}, temperature={spec.get('temperature')}, "
            f"parse_retries={spec.get('parse_retries')}, api_call_retries={spec.get('api_call_retries')}, "
            f"cache={cfg['reuse_parser_responses']})"
            if policy.uses_llm() else "parser model: none (every stage reads with code)")
    return [head] + policy.summary_lines()


def show(configs: list[dict], parsers: bool = True) -> None:
    rows = table(configs)
    columns = list(rows[0])
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in columns}
    print(f"{len(configs)} experiment(s); columns are the knobs they differ on, "
          f"output under {configs[0]['output_root']}")
    print("  ".join(c.ljust(widths[c]) for c in columns))
    for row in rows:
        print("  ".join(str(row[c]).ljust(widths[c]) for c in columns))
    if not parsers:
        return
    shown = set()
    for cfg in configs:
        lines = parser_summary(cfg)
        key = "\n".join(lines)
        if key in shown:  # identical for every experiment: print it once
            continue
        shown.add(key)
        label = cfg["name"] if len(configs) > 1 else ""
        print(f"\nhow model text is read{' (' + label + ' and every experiment like it)' if label else ''}:")
        print("\n".join("  " + line for line in lines))
