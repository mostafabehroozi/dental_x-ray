"""Sequential, resumable research suites. No notebook globals or inference at import."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import time
import uuid
from pathlib import Path

from benchmark import count_yolo_class_instances, YOLO_CLASS_TO_CONDITION
from dentalgpt import DentalExpertModelRunner, LLMVisionAnalysisRunner
from prompts import CONDITION_LABELS
from research_prompts import PROMPT_TEMPLATES, COUNT_SUBJECTS, FINDING_GROUPS, LOCATION_MODES, ATOMIC_FINDING_LABELS
from model_routing import credentials

DEFAULT_SETTINGS = {
    "max_tokens": 1024, "temperature": 0.0, "top_p": 1.0, "timeout": 600,
    "atomic_protocol": "combined", "count_only_if_present": True,
    "recovery": {"retries_per_template": 2, "temperatures": [0.0, 0.2, 0.4],
                 "fallback_templates": ["atomic_2", "atomic_3"],
                 "broad_fallback_templates": ["broad_2", "broad_3"]},
}


def merge(*items):
    result = {}
    for item in items:
        for key, value in item.items():
            result[key] = merge(result.get(key, {}), value) if isinstance(value, dict) else copy.deepcopy(value)
    return result


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf8")
    os.replace(temporary, path)


def local_preset(models, experiments):
    presets = {models[e["model"]].get("preset", "QUALITY") for e in experiments
               if models[e["model"]]["backend"] == "local"}
    if presets - {"QUALITY", "FAST"} or len(presets) > 1:
        raise ValueError("Select at most one local DentalGPT preset: QUALITY or FAST")
    return next(iter(presets), None)


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError(f"Use letters, digits, underscores or hyphens for IDs: {value!r}")


def template_order(job, stage):
    primary = job["strategy"]["template_id"]
    recovery = job["settings"]["recovery"]
    alternatives = recovery["broad_fallback_templates" if stage == "broad" else "fallback_templates"]
    return list(dict.fromkeys([primary, *alternatives]))


def question(job, stage, template_id, condition=None, region=None):
    text = job["templates"][stage][template_id]
    if stage == "broad":
        definitions = "\n".join(f"{c}: {COUNT_SUBJECTS[c]} ({CONDITION_LABELS[c]})" for c in job["conditions"])
        example = json.dumps(dict.fromkeys(job["conditions"], 0))
        return (text + "\nCounting categories:\n" + definitions +
                "\nCount each spatially distinct instance once. Include every key exactly once; "
                "use non-negative integer counts and zero for absent findings. "
                "Return only <answer>" + example + "</answer>, replacing the example zeros with your counts.")
    whole = job["strategy"]["location_mode"] == "whole"
    image_scope = "this image" if whole else f"{region} of this image"
    count_scope = "the image" if whole else f"{region} of the image"
    count_question = (
        f"How many visible teeth in {count_scope} appear to have dental fillings based on their radiopaque characteristics?"
        if condition == "dental_filling" else
        f"How many {COUNT_SUBJECTS[condition]} are visible in {count_scope}?"
    )
    rendered = text.format(finding=ATOMIC_FINDING_LABELS[condition], count_subject=COUNT_SUBJECTS[condition],
                           region=region, image_scope=image_scope, count_scope=count_scope,
                           count_question=count_question)
    # Presence is a pure A/B question; do not append counting/partition instructions.
    if stage in {"count", "combined"} and not whole:
        rendered += ("\nCount only the specified region. For an instance crossing a regional boundary, "
                     "assign it once to the region containing its center.")
    return rendered


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def parse_answer(raw, stage, conditions, positive=False):
    bodies = re.findall(r"<answer>\s*(.*?)\s*</answer>", raw, re.I | re.S)
    if len(bodies) != 1:
        raise ValueError("expected_one_answer_tag")
    body = bodies[0].strip()
    if stage == "presence":
        if body.upper() not in {"A", "B"}:
            raise ValueError("expected_A_or_B")
        return body.upper()
    if stage == "count":
        if not re.fullmatch(r"\d+", body):
            raise ValueError("expected_nonnegative_integer")
        value = int(body)
        if positive and value == 0:
            raise ValueError("positive_presence_zero_count")
        return value
    value = json.loads(body, object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise ValueError("expected_object")
    if stage == "combined":
        if set(value) != {"choice", "count"} or value["choice"] not in {"A", "B"}:
            raise ValueError("invalid_choice_count_schema")
        count = value["count"]
        if type(count) is not int or count < 0 or (value["choice"] == "A") != (count > 0):
            raise ValueError("inconsistent_choice_count")
    elif set(value) != set(conditions) or any(type(v) is not int or v < 0 for v in value.values()):
        raise ValueError("invalid_broad_counts")
    return value


def prepare_suite(providers, models, strategies, experiments, images, defaults=None, prompt_overrides=None,
                  local_runtime=None):
    """Validate and freeze an execution plan, without network/model calls."""
    if not experiments or not images:
        raise ValueError("Select at least one experiment and image")
    if len({e["id"] for e in experiments}) != len(experiments):
        raise ValueError("Duplicate experiment ID")
    if len({i["id"] for i in images}) != len(images):
        raise ValueError("Duplicate image ID")
    local_preset(models, experiments)
    def check_secrets(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key.lower() in {"api_key", "authorization", "password", "extra_headers", "default_headers"}:
                    raise ValueError("Keep credentials in PROVIDERS, not model/settings snapshots")
                check_secrets(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                check_secrets(item)
    check_secrets([models, defaults or {}, experiments, local_runtime or {}])
    templates = merge(PROMPT_TEMPLATES, prompt_overrides or {})
    prepared_images = []
    for image in images:
        _identifier(image["id"])
        image = dict(image)
        for key in ("image_path", "label_path"):
            path = Path(image[key]).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            image[key] = str(path)
            image[key + "_hash"] = hashlib.sha256(path.read_bytes()).hexdigest()
        count_yolo_class_instances(image["label_path"], YOLO_CLASS_TO_CONDITION)
        prepared_images.append(image)
    jobs = []
    for experiment in experiments:
        _identifier(experiment["id"])
        model = copy.deepcopy(models[experiment["model"]])
        if model.get("backend") not in {"api", "local"}:
            raise ValueError("Model backend must be api or local")
        if not model.get("model") or (model["backend"] == "local" and model["model"] != "DentalGPT"):
            raise ValueError("Missing model ID or unsupported local model")
        provider = providers[model["provider"]] if model["backend"] == "api" else {}
        if model["backend"] == "api" and not any(provider.get(k) for k in ("api_key", "api_key_env", "api_key_secret")):
            raise ValueError(f"No credential reference for {model['provider']}")
        if not experiment["strategies"] or len(set(experiment["strategies"])) != len(experiment["strategies"]):
            raise ValueError("Select unique strategies")
        for strategy_id in experiment["strategies"]:
            _identifier(strategy_id)
            strategy = copy.deepcopy(strategies[strategy_id])
            settings = merge(DEFAULT_SETTINGS, defaults or {}, model.get("settings", {}), experiment.get("settings", {}))
            settings["atomic_protocol"] = experiment.get("atomic_protocol", strategy.get("atomic_protocol", settings["atomic_protocol"]))
            if settings["atomic_protocol"] not in {"combined", "presence_then_count"}:
                raise ValueError("Unknown atomic protocol")
            if strategy["mode"] not in {"broad", "atomic"}:
                raise ValueError("Unknown strategy mode")
            regions = LOCATION_MODES[strategy["location_mode"]]
            if strategy["mode"] == "broad" and strategy["location_mode"] != "whole":
                raise ValueError("Structured broad currently uses whole image")
            conditions = FINDING_GROUPS[strategy["finding_group"]]
            recovery = settings["recovery"]
            if type(recovery["retries_per_template"]) is not int or recovery["retries_per_template"] < 0:
                raise ValueError("retries_per_template must be a non-negative integer")
            if not recovery["temperatures"] or any(not isinstance(t, (int, float)) or not 0 <= t <= 2 for t in recovery["temperatures"]):
                raise ValueError("Temperatures must be a non-empty sequence between 0 and 2")
            if list(recovery["temperatures"]) != sorted(recovery["temperatures"]):
                raise ValueError("Retry temperatures must be non-decreasing")
            if not 0 <= settings["temperature"] <= 2:
                raise ValueError("temperature must be between 0 and 2")
            if type(settings["max_tokens"]) is not int or settings["max_tokens"] <= 0 or not math.isfinite(settings["timeout"]) or settings["timeout"] <= 0:
                raise ValueError("Positive max_tokens and timeout required")
            if not math.isfinite(settings["top_p"]) or not 0 < settings["top_p"] <= 1:
                raise ValueError("top_p must be in (0, 1]")
            if type(settings["count_only_if_present"]) is not bool:
                raise ValueError("count_only_if_present must be boolean")
            prices = model.get("prices_per_million", {})
            if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in prices.values()):
                raise ValueError("Token prices must be finite non-negative numbers")
            if settings.get("token_limit_parameter", "max_tokens") not in {"max_tokens", "max_completion_tokens"}:
                raise ValueError("Unsupported token limit parameter")
            if set(settings.get("omit_parameters", [])) - {"temperature", "top_p", "seed"}:
                raise ValueError("Only sampling parameters can be omitted")
            if set(settings.get("request_options", {})) & {"model", "messages", "max_tokens", "max_completion_tokens", "temperature", "top_p", "stream"}:
                raise ValueError("request_options cannot override routing, generation settings or streaming")
            if model["backend"] == "local" and (settings.get("request_options") or settings.get("token_limit_parameter", "max_tokens") != "max_tokens"):
                raise ValueError("Local DentalGPT uses its existing llama.cpp request contract")
            for image in prepared_images:
                job = {"id": f"{experiment['id']}__{strategy_id}__{image['id']}",
                       "experiment_id": experiment["id"], "strategy_id": strategy_id,
                       "model_key": experiment["model"], "model": model, "settings": settings,
                       "strategy": strategy, "conditions": conditions, "regions": regions,
                       "image": image, "templates": templates,
                       "provider": {k: v for k, v in provider.items() if k in {"base_url", "api_key_env", "api_key_secret"}},
                       "local_runtime": local_runtime if model["backend"] == "local" else None}
                stages = ["broad"] if strategy["mode"] == "broad" else (
                    ["combined"] if settings["atomic_protocol"] == "combined" else ["presence", "count"])
                # Hash actual rendered inputs as well as template text: renderer/label
                # changes must invalidate resume even when template IDs stay the same.
                job["prompt_fingerprint"] = digest([
                    question(job, stage, template_id, condition, region)
                    for stage in stages for template_id in template_order(job, stage)
                    for condition, region in ([(None, None)] if stage == "broad" else
                        [(condition, region) for condition in conditions for _, region in regions])
                ])
                checks = 1 if strategy["mode"] == "broad" else len(conditions) * len(regions)
                job["planned_checks"] = checks
                job["min_calls"] = checks * (2 if len(stages) == 2 and not settings["count_only_if_present"] else 1)
                job["max_calls"] = checks * sum(len(template_order(job, stage)) for stage in stages) * (recovery["retries_per_template"] + 1)
                jobs.append(job)
    if len({job["id"] for job in jobs}) != len(jobs):
        raise ValueError("Expanded job IDs collide; choose distinct experiment/strategy/image IDs")
    return {"version": 1, "jobs": jobs, "fingerprint": digest(jobs)}


def make_runner(job, providers):
    settings, model = job["settings"], job["model"]
    common = {k: settings[k] for k in ("max_tokens", "temperature", "top_p", "timeout")}
    if model["backend"] == "local":
        runtime = job.get("local_runtime")
        if not runtime:
            raise ValueError("Local runtime is not initialized; rerun setup cells")
        return DentalExpertModelRunner(**runtime, **common, seed=settings.get("seed", 0), max_retries=0,
                                      omit_parameters=settings.get("omit_parameters", ()))
    provider = providers[model["provider"]]
    return LLMVisionAnalysisRunner(model=model["model"], base_url=provider.get("base_url"),
        api_key=credentials(provider), **common, max_retries=0,
        omit_parameters=settings.get("omit_parameters", ()),
        token_limit_parameter=settings.get("token_limit_parameter", "max_tokens"),
        request_options=settings.get("request_options"))


class PermanentFailure(Exception):
    pass


def _permanent(exc):
    return getattr(exc, "status_code", None) in {400, 401, 403, 404, 405, 422} or isinstance(exc, (TypeError, FileNotFoundError))


def _stage(job, runner, state, path, stage, condition=None, region_id="whole", region=None, positive=False):
    key = f"{region_id}/{condition or 'all'}/{stage}"
    if key in state["stages"]:
        return state["stages"][key]
    recovery = job["settings"]["recovery"]
    for template_index, template_id in enumerate(template_order(job, stage)):
        for attempt_index in range(recovery["retries_per_template"] + 1):
            attempt_id = f"{key}/{template_id}/{attempt_index}"
            previous = next((a for a in state["attempts"] if a["id"] == attempt_id), None)
            if previous:
                if previous.get("permanent"):
                    raise PermanentFailure(previous["error"])
                if previous.get("success"):
                    result = {"success": True, "parsed": previous["parsed"], "template_id": template_id}
                    state["stages"][key] = result
                    save_json(path, state)
                    return result
                continue
            # Schedule gives increments relative to its first entry, starting from model temperature.
            temperature = min(2.0, job["settings"]["temperature"] +
                recovery["temperatures"][min(attempt_index, len(recovery["temperatures"]) - 1)] - recovery["temperatures"][0])
            prompt = question(job, stage, template_id, condition, region)
            attempt = {"id": attempt_id, "stage": stage, "template_id": template_id,
                       "fallback_template": template_index > 0, "retry": attempt_index > 0,
                       "question": prompt, "requested_temperature": temperature,
                       "temperature_omitted": "temperature" in job["settings"].get("omit_parameters", []),
                       "success": False, "error": "interrupted_before_response", "status": "started"}
            state["attempts"].append(attempt)
            save_json(path, state)  # A crash consumes this attempt; never silently repeats it.
            started = time.perf_counter()
            try:
                response = runner.ask(job["image"]["image_path"], prompt,
                                      max_tokens=job["settings"]["max_tokens"], temperature=temperature)
                attempt["response"] = response
            except Exception as exc:
                attempt.update(error=f"{type(exc).__name__}: request failed", failure_type="operational", permanent=_permanent(exc))
            else:
                try:
                    if response.get("truncated"):
                        raise ValueError("truncated_output")
                    attempt["parsed"] = parse_answer(response.get("raw_answer", ""), stage, job["conditions"], positive)
                    attempt.update(success=True, error=None)
                except (ValueError, TypeError) as exc:
                    attempt.update(error=str(exc), failure_type="format")
            finally:
                attempt["latency_seconds"] = time.perf_counter() - started
                attempt["status"] = "interrupted" if attempt["error"] == "interrupted_before_response" else "finished"
                save_json(path, state)
            if attempt.get("permanent"):
                raise PermanentFailure(attempt["error"])
            if attempt["success"]:
                result = {"success": True, "parsed": attempt["parsed"], "template_id": template_id}
                state["stages"][key] = result
                save_json(path, state)
                return result
    result = {"success": False, "parsed": None}
    state["stages"][key] = result
    save_json(path, state)
    return result


def _execute(job, runner, state, path):
    counts = dict.fromkeys(job["conditions"], 0)
    checks, region_counts = [], {}
    if job["strategy"]["mode"] == "broad":
        result = _stage(job, runner, state, path, "broad")
        if result["success"]:
            counts = result["parsed"]
        checks.append({"forced_zero": not result["success"]})
    else:
        for region_id, region in job["regions"]:
            region_counts[region_id] = {}
            for condition in job["conditions"]:
                args = (job, runner, state, path)
                kwargs = {"condition": condition, "region_id": region_id, "region": region}
                if job["settings"]["atomic_protocol"] == "combined":
                    result = _stage(*args, "combined", **kwargs)
                    count = result["parsed"]["count"] if result["success"] else 0
                else:
                    result = _stage(*args, "presence", **kwargs)
                    count = 0
                    if result["success"] and (result["parsed"] == "A" or not job["settings"]["count_only_if_present"]):
                        result = _stage(*args, "count", positive=result["parsed"] == "A", **kwargs)
                        count = result["parsed"] if result["success"] else 0
                counts[condition] += count
                region_counts[region_id][condition] = count
                checks.append({"region_id": region_id, "condition": condition, "forced_zero": not result["success"]})
    state.update(status="completed", prediction_counts_by_condition=counts,
                 region_counts_by_condition=region_counts, checks=checks)
    save_json(path, state)


def run_suite(plan, providers, output_dir, resume_dir=None, runner_factory=None):
    """Persist every attempt. Resuming terminal results never calls a model again."""
    if digest(plan["jobs"]) != plan["fingerprint"]:
        raise ValueError("Prepared plan changed; rerun prepare_suite")
    for job in plan["jobs"]:
        for key in ("image_path", "label_path"):
            if hashlib.sha256(Path(job["image"][key]).read_bytes()).hexdigest() != job["image"][key + "_hash"]:
                raise ValueError("Input contents changed; rerun prepare_suite")
    run_dir = Path(resume_dir) if resume_dir else Path(output_dir) / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
    if resume_dir:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf8"))
        if manifest["fingerprint"] != plan["fingerprint"]:
            raise ValueError("Resume mismatch: configs, prompts, image or label contents changed")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        save_json(run_dir / "manifest.json", plan)
    print(f"Research checkpoints: {run_dir}")
    runners, failed_experiments = {}, {}
    for job in plan["jobs"]:
        path = run_dir / (job["id"] + ".json")
        state = json.loads(path.read_text(encoding="utf8")) if path.exists() else {
            "job_id": job["id"], "status": "running", "attempts": [], "stages": {}}
        if state["status"] in {"completed", "failed"}:
            if state["status"] == "failed":
                failed_experiments[job["experiment_id"]] = state["error"]
            continue
        try:
            if job["experiment_id"] in failed_experiments:
                raise PermanentFailure(failed_experiments[job["experiment_id"]])
            cache_key = digest({"model": job["model"], "settings": job["settings"], "runtime": job["local_runtime"]})
            if cache_key not in runners:
                try:
                    runners[cache_key] = (runner_factory or (lambda j: make_runner(j, providers)))(job)
                except Exception as exc:
                    raise PermanentFailure(f"{type(exc).__name__}: runner initialization failed") from exc
            _execute(job, runners[cache_key], state, path)
        except PermanentFailure as exc:
            failed_experiments[job["experiment_id"]] = str(exc)
            state.update(status="failed", error=str(exc))
            save_json(path, state)
        print(f"{job['id']}: {state['status']} ({len(state['attempts'])} calls)")
    return str(run_dir)


def load_suite(run_dir):
    run_dir = Path(run_dir)
    plan = json.loads((run_dir / "manifest.json").read_text(encoding="utf8"))
    if digest(plan["jobs"]) != plan["fingerprint"]:
        raise ValueError("Saved manifest contents do not match their fingerprint")
    results = []
    for job in plan["jobs"]:
        path = run_dir / (job["id"] + ".json")
        state = json.loads(path.read_text(encoding="utf8")) if path.exists() else {"status": "pending", "attempts": []}
        results.append({**job, **state, "output_id": job["id"], "selected_conditions": job["conditions"]})
    return results
