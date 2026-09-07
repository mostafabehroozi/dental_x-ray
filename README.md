# Dental expert-model reporting pipeline

The pipeline treats the image-capable dental foundational model as a replaceable **dental expert model**. It can use DentalGPT through a local llama.cpp multimodal server or an OpenAI-compatible multimodal LLM API. Reporting and evaluation code do not depend on the selected backend.

## Runtime model pair

- `DentalGPT-7B-1026.Q6_K.gguf`
- `DentalGPT-7B-1026.mmproj-f16.gguf`

The model and projector are downloaded from `mradermacher/DentalGPT-7B-1026-GGUF`.

## Architecture

`panoramic image -> local DentalGPT or multimodal LLM API -> multi-level text observations -> dentist-report synthesis -> evaluation adaptation`

The expert-model layer answers independent broad, family, atomic, and location questions. Python preserves those source observations and normalizes atomic `PRESENT/ABSENT/UNCERTAIN` answers. When enabled, text-only models run two separate phases. The adapter can reuse the orchestrator or use another configured API model:

1. **Dentist report:** comprehensively combines all useful supported content, removes redundancy, and preserves locations, uncertainty, conflicts, limitations, and relevant negative findings.
2. **Evaluation adaptation report:** maps the completed dentist report into the closed evaluation ontology without changing deterministic atomic statuses.

Neither orchestration phase receives the radiograph.

## Modes

- `BASIC`: four broad independent passes.
- `DISEASE_HIERARCHY`: broad + family + one atomic question per condition.
- `DISEASE_AND_LOCATION`: hierarchy plus three coarse-to-fine location questions for each PRESENT/UNCERTAIN candidate.

## Kaggle

Run `main_notebook.ipynb` and edit Cell 3 only. Each API provider keeps its `base_url` and `api_key` together in `PROVIDERS`. Each model role has one configuration: use `{"backend": "local", "model": "DentalGPT", "preset": "QUALITY"}` for DentalGPT, or `{"backend": "api", "provider": "provider-name", "model": "model-name"}` for an API model. Set an optional role to `None` to disable it. An API analyzer sends the image and the same analysis questions directly to the configured multimodal model and skips all llama.cpp/GGUF setup cells.

Cell 3 also centralizes API reliability with `API_CALL_DELAY_SECONDS = 1.0`,
`API_CALL_MAX_RETRIES = 10`, and `LOG_API_CALLS = True`. The delay is the
minimum gap between request starts. Every request error is printed and retried
up to the configured count; after the first attempt plus all retries fail, the
provider, model, and final error are printed and an exception stops execution.
The OpenAI SDK's separate hidden retry loop is disabled to prevent double retries.

The optional external orchestrator is disabled by default so the expert-model stage can be tested fully locally first.

For a lower-memory/faster development run, switch to `DentalGPT-7B-1026.Q4_K_M.gguf` + `DentalGPT-7B-1026.mmproj-Q8_0.gguf`.

The notebook pins llama.cpp to `b10516` rather than building an arbitrary future `master`, because multimodal APIs are evolving quickly.

## Offline benchmark evaluation

`benchmark.py` loads a YOLO image/label split and converts its boxes into the stable
14-condition ontology. `evaluation.py` then scores a directory containing one saved
pipeline JSON per image. Metric calculation is deterministic; the existing
`evaluation_adaptation_report` is used only as normalized prediction input.

```python
from benchmark import load_yolo_benchmark
from evaluation import EvaluationConfig, evaluate_experiment

benchmark = load_yolo_benchmark(
    "/kaggle/input/dataset/images/test",
    "/kaggle/input/dataset/labels/test",
    data_yaml="/kaggle/input/dataset/data.yaml",
)
result = evaluate_experiment(
    benchmark,
    "/kaggle/working/experiment_broad_v1",
    config=EvaluationConfig(),
    experiment_metadata={"name": "broad_v1"},
    output_path="/kaggle/working/experiment_broad_v1/evaluation_results.json",
)
```

Finding evaluation is enabled by default and location evaluation is disabled. Set
`evaluate_location=True`, choose level 1 or 2, and select `location_adapters` as
`("geometry",)`, `("vision",)`, or both. Vision-derived benchmark locations must be
prepared with `prepare_vision_location_cache`; the notebook's optional Cell 16 shows
both the external multimodal-LLM and dental expert-model resolver paths. Keep each
prompt/system variant in a separate prediction directory, then use
`compare_experiments()` on their saved evaluation result files.


## Multi-model research suite (cells 15.x)

The research suite runs an explicit set of model/strategy configurations on the
same labeled images, without an orchestrator or adapter. Standard templates and
region definitions live in `research_prompts.py`; execution, parsing, recovery,
and checkpointing live in `research_experiments.py`.

1. In Cell 3, leave `RESEARCH_SUITE_MODE=True` and ordinary smoke/pipeline/benchmark
   switches off. Configure `PROVIDERS`, `MODELS`, and `EXPERIMENTS` before running
   setup. Use `api_key_env` for an environment variable or `api_key_secret` for a
   Kaggle Secret. Direct `api_key` values remain supported but are not saved.
2. Run Cells 1-10. Local setup is required only if a selected research model is
   local. A mixed suite starts one DentalGPT runtime; API-only suites skip it.
   When model selection or local preset changes, rerun Cells 3-9.
3. In Cell 15.01, edit `STRATEGIES`, `SUITE_DEFAULTS`, `PROMPT_OVERRIDES`, and
   `SUITE_IMAGES`. Each image needs a unique simple ID and an explicit label path.
   The current image is the default. Preview checks and call bounds before execution.
4. Run 15.02 to execute, then 15.2 and 15.3 to load and evaluate. Cell 15.1 is an
   independent optional text synthesis experiment and can be skipped.

Example Cell 3 model selection (replace model IDs with those your endpoints accept):

```python
MODELS = {
    "api_a": {"backend": "api", "provider": "openai", "model": "YOUR_MODEL_ID",
              "settings": {"max_tokens": 2048, "temperature": 0.0}},
    "api_b": {"backend": "api", "provider": "nvidia", "model": "YOUR_MODEL_ID"},
    "dentalgpt": {"backend": "local", "model": "DentalGPT", "preset": "QUALITY",
                  "settings": {"atomic_protocol": "presence_then_count"}},
}
EXPERIMENTS = [
    {"id": name + "_baseline", "model": name,
     "strategies": ["broad_whole", "atomic_whole", "atomic_arch"]}
    for name in MODELS
]
```

Model generation settings are merged over `SUITE_DEFAULTS`; an experiment's
`settings` dictionary overrides both. The optional `atomic_protocol` field on a
strategy overrides the model default, and the experiment's `atomic_protocol`
field overrides that. Use `combined` or `presence_then_count` on either backend.
Thus DentalGPT versus API comparisons can use identical protocols or deliberately
compare different ones; the report records the protocol.

API model `settings` can include `omit_parameters=["temperature", "top_p"]`,
`token_limit_parameter="max_completion_tokens"`, and `request_options` for
supported endpoint options (for example reasoning controls). Omission never
silently changes model/provider. Check endpoint compatibility on one image first.
No native Anthropic/Gemini API adapter is included; use an image-capable
OpenAI-compatible endpoint. Only one local QUALITY/FAST preset can be used per suite.

The default three combined strategies take 43 calls per image before retries:
one structured broad response, 14 whole-image checks, and 28 arch checks.
Two-step atomic protocols add count calls after positive presence answers.
Arch checks form one report by summing counts for each finding across both jaws.
Finer region levels remain available as `atomic_quadrant` and `atomic_six_zone`.
They are independent strategies, never added to the whole/arch reports.

Prompt experiments use overrides such as:

```python
PROMPT_OVERRIDES = {
    "combined": {"atomic_1": "Your prompt using {finding}, {region}, and {count_subject} ..."}
}
```

Keep the relevant `<answer>` contract when changing wording. Broad overrides are
introductory instructions; the fixed category definitions and count JSON contract
are appended automatically. All default wording remains in the Python module.

Recovery retries the primary wording twice, then tries each alternative wording
with the same sampling schedule, then records a forced zero. The default schedule
is 0.0, 0.2, 0.4; increments are relative to the model's initial temperature and
capped at 2. Models that omit temperature repeat without that parameter, and this
is recorded. Set fallback lists to `[]` for fixed-wording comparisons. SDK retries
are disabled for research runners, so attempts are counted explicitly. Truncated
or malformed responses use the same recovery. Failed broad responses become a
flagged all-zero report. Authentication/configuration errors fail that experiment
and allow others to continue. Terminal failed experiments are not retried on
resume; correct the configuration and start a new run.

Each run prints its directory before inference. It contains a sanitized manifest
and one result/checkpoint file per experiment/strategy/image, including every
attempt and its prompt, response, and metadata. To resume, set `RESUME_RUN_DIR` in
15.01 to that directory and rerun 15.01/15.02. Changed configuration, prompt, image,
or label contents reject resume. An interrupted in-flight request consumes its
attempt slot because delivery/billing cannot be known; it is never silently repeated.
Do not run two processes against the same resume directory.

To evaluate saved results without inference, set `RESEARCH_RUN_DIR` and execute
15.2/15.3. `evaluation/` contains summary, per-image, and per-finding CSV files,
`evaluation.json`, and `full_results.json`. Ground-truth label files must remain
available and unchanged. `SHOW_FINDING_DETAILS` controls the notebook detail view;
exports always contain all rows. No image is sent to a model during evaluation.

Presence metrics use one binary decision per image/finding. Count metrics report
matched, excess, and missed counts, MAE, and exact-count accuracy; these are count
agreement, not spatial instance matching. Regions guide prompts but are not scored.
Forced-zero predictions are included in metrics with their rate shown. Missing or
failed images remain incomplete, and coverage differences are flagged. Undefined
metrics are null. Token totals/cost are unavailable when any attempt lacks usage;
optional model `prices_per_million={"input": ..., "output": ...}` enables a simple
estimate, not an invoice (no cached-token or provider billing adjustments).

Offline validation: `python -m unittest -q test_research_suite`. These tests use
mocked responses and synthetic labels; they do not validate real provider quality,
model compatibility, DentalGPT loading, or GPU performance.


### Atomic prompts aligned with the DentalGPT paper

The two-step templates now follow the examples in the supplied DentalGPT paper
(arXiv:2512.11558v1): Section 3.2, page 5, describes reasoning in `<think>` and a
final answer in `<answer>`; Figure 7, page 10, shows the short condition question
with `A. True / B. False`; Figure 9, page 13, shows a direct filling-count question.
Figure 7 is an evaluation example, not evidence of the exact training question text.
The complete training templates and exact appended format instruction are not
published in this PDF, so these prompts are close adaptations, not an exact
reproduction of the training distribution.

`atomic_1` uses the Figure 7 panoramic stem. `atomic_2` adapts its intraoral stem,
and `atomic_3` is a local fallback paraphrase. Whole-image questions say simply
"this image"; regional questions substitute the selected region. Presence-only
questions no longer receive counting instructions. The filling-count question
uses Figure 9's wording; count questions for other findings are analogous extensions.
The 14 benchmark labels and their counting units remain unchanged; the paper's
panoramic classification benchmark itself contains six categories.

Choose `atomic_protocol="presence_then_count"` for the closest supported workflow
(already the notebook's DentalGPT default). Both local DentalGPT stages request the tagged reasoning
format without imposing "brief" reasoning. The count's final answer is still
restricted to one integer for deterministic evaluation; Figure 9 instead shows a
prose final answer. Local combined JSON remains an explicitly experimental alternative,
not the paper's multiple-choice output format. Its question stem follows the
same short style. API models use the separate profile described below.
Broad JSON is also an experimental evaluation contract.

Parsing evaluates only the final `<answer>` payload; it does not grade or require
reasoning text. Recovery still retries sampling, then wording, then records B/0.
The training response cap reported on page 6 is 8192 tokens; this is not a proven
inference optimum. Notebook budgets are unchanged. Increase the local model's
`settings.max_tokens` if truncation is frequent, within the server context limit.

Start a new run after these prompt changes. Reload the Python modules (or restart
the kernel and rerun setup) before 15.01. Resume fingerprints now also cover the
rendered prompts, so old prompt runs cannot be silently mixed with new ones.

### API analyzer comparison with a separate adapter (parallel Kaggle notebook)

Use the updated `dentalgpt.py`, `research_prompts.py`, `research_experiments.py`,
and `evaluation.py` in the Kaggle project folder, then restart the kernel. The
research adapter is selected by `EXPERIMENTS[*]["adapter"]`, independently of the
ordinary pipeline's `ADAPTER` and `ORCHESTRATOR` globals. Those old roles should
be disabled for this experiment. The existing FDM configuration needs no changes.

In the copied notebook, add this block at the end of Cell 3's model/experiment
definitions, **before** `RESEARCH_LOCAL_PRESET = ...`. Replace model IDs and set
the two Kaggle Secrets. Provider entries can use different compatible endpoints.

```python
OUTPUT_DIR = "/kaggle/working/dental_llm_comparison"
RESEARCH_SUITE_MODE = True
RUN_SMOKE_TEST = RUN_PIPELINE = RUN_EVALUATION = False
EVALUATE_LOCATION = False
ORCHESTRATOR = None
ADAPTER = None  # The research adapter is selected below, not through this old role.

PROVIDERS = {
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key_secret": "NVIDIA_API_KEY",
    },
    "openai": {
        "base_url": None,
        "api_key_secret": "OPENAI_API_KEY",
    },
}
MODELS = {
    "analyzer_a": {
        "backend": "api", "provider": "nvidia", "model": "YOUR_VISION_MODEL_ID",
        "settings": {"max_tokens": 2048, "temperature": 0.0, "atomic_protocol": "combined"},
    },
    "analyzer_b": {
        "backend": "api", "provider": "openai", "model": "YOUR_OTHER_VISION_MODEL_ID",
        "settings": {"max_tokens": 2048, "temperature": 0.0, "atomic_protocol": "combined"},
    },
    "count_adapter": {
        "backend": "api", "provider": "openai", "model": "YOUR_TEXT_MODEL_ID",
        "settings": {"max_tokens": 2048, "temperature": 0.0},
    },
}
SELECTED_ANALYZERS = ["analyzer_a"]  # Add "analyzer_b" to compare both in one run.
EXPERIMENTS = [
    {"id": name, "model": name, "adapter": "count_adapter", "adapter_scope": "broad",
     "strategies": ["broad_whole", "atomic_whole", "atomic_arch"]}
    for name in SELECTED_ANALYZERS
]
ANALYZER = MODELS[SELECTED_ANALYZERS[0]]
SMOKE_ANALYZER = LOCATION_ANALYZER = ANALYZER
ANALYZER_BACKEND = "api"
```

Keep the existing `RESEARCH_LOCAL_PRESET` / `NEEDS_LOCAL_RUNTIME` calculations
after this block. Run setup through Cell 10; the local model build/download/start
steps will skip automatically. Each selected analyzer must accept image input
through the configured Chat Completions endpoint. For models that require it,
add `omit_parameters=["temperature", "top_p"]` and/or
`token_limit_parameter="max_completion_tokens"` inside that model's `settings`.
These settings are also supported for the adapter. Select exact model IDs supported
by your provider; this configuration does not claim every current model is compatible.

In Cell 15.01 replace the `STRATEGIES` assignment with:

```python
STRATEGIES = {
    "broad_whole": {
        "mode": "broad", "location_mode": "whole", "finding_group": "all_14",
        "template_id": "broad_1", "output_format": "narrative",
    },
    "atomic_whole": {
        "mode": "atomic", "location_mode": "whole", "finding_group": "all_14",
        "template_id": "atomic_1", "atomic_protocol": "combined",
    },
    "atomic_arch": {
        "mode": "atomic", "location_mode": "arch", "finding_group": "all_14",
        "template_id": "atomic_1", "atomic_protocol": "combined",
    },
}
```

Keep the remaining Cell 15.01 code, including your image/label list and recovery
settings. Set `RESUME_RUN_DIR=None` for a new run and `RUN_RESEARCH_SUITE=True`.
Run 15.01 → 15.02 → 15.2 → 15.3, skipping the optional orchestrator Cell 15.1.
The three summary rows per analyzer represent:

| Strategy | Analyzer image calls | Adapter text calls | Evaluation |
| --- | ---: | ---: | --- |
| broad_whole | 1 | 1 | Counts extracted from the broad narrative |
| atomic_whole | 14 | 0 | Whole-image atomic counts |
| atomic_arch | 28 | 0 | Atomic counts summed across both arches |

Thus the default is 44 calls per image/analyzer before retries. The adapter is a
text-only count normalizer; it receives only the analyzer report, never the image,
file paths, ground-truth labels, or previous experiments. Keep the same adapter
across analyzers for a controlled comparison. Broad accuracy measures the combined
analyzer-plus-adapter path, so it can include extraction errors.

To adapt all three grouped reports instead, use `adapter_scope="all"`: one
adapter call per group, not per atomic check (46 total calls before retries).
For atomic groups the adapter receives deterministic regional counts and their
whole-image sum, with instructions not to add the two representations. Original
counts remain in the saved result alongside adapted counts.

Adapter generation and recovery are independent of analyzer settings:
`ADAPTER_DEFAULTS` → adapter model `settings` → experiment `adapter_settings`.
Default recovery uses two resampling retries followed by two alternative adapter
wordings. Customize their IDs through `adapter_template_id` and
`adapter_settings.recovery.fallback_templates`; override template text in
`PROMPT_OVERRIDES["adapter"]`. No SDK retries are hidden underneath this policy.
An exhausted adapter emits a flagged all-zero result. Ambiguous positive counts
also use zero but are explicitly listed in `adapter_unresolved_conditions`; no
arbitrary count is inferred from "multiple" or "several". Unmentioned findings
are treated as absent. Missing/refused/incorrect analyzer reports remain a source
of benchmark error; the adapter is not another image reader.

Cell 15.3's exported tables and `evaluation_summary_table_15` now include adapter
model/provider, separate analyzer/adapter call counts, unresolved finding counts,
and adapter fallback counts. Total token usage includes both roles, and estimated
cost uses each role's own optional `prices_per_million`. To display these new fields
in an existing copied notebook, append to Cell 15.3:

```python
display(evaluation_summary_table_15[[
    "experiment_id", "strategy_id", "model", "provider",
    "adapter_model", "adapter_provider", "analyzer_calls", "adapter_calls",
    "adapter_unresolved_findings", "adapter_fallback_images", "estimated_cost",
]])
```

If using separate Kaggle sessions, assign each session its own output directory
and preserve its run folder. A single session with multiple selected analyzers
automatically produces the combined report. Adapter credentials are excluded from
saved manifests, and changing the adapter/configuration invalidates resume.

### Provider-neutral API analyzer prompts

API analyzers now automatically use a separate `api` prompt profile. Local
DentalGPT defaults to `dentalgpt`, preserving its existing paper-aligned prompts.
The API profile provides three wording variants for broad, combined atomic, and
optional two-step presence/count questions. It does not request `<think>` text or
a reasoning transcript. The existing `<answer>` parser contract remains intact.
Reasoning-model API settings remain model-specific; prompt text is not an API
switch for enabling or disabling native reasoning.

All API variants specify visible evidence, independent finding decisions, the
existing counting unit, and counting each supported instance once. Whole-image
presence means anywhere in the image, not a generalized condition. Atomic queries
restrict counting to the requested region; regional boundary instructions continue
to avoid counting a spanning instance twice. There are no image-specific answers,
ground-truth examples, or new disease categories in these templates.

Broad narrative output now asks for a concise limitations summary and one table
row per category: category key, PRESENT/ABSENT/UNCERTAIN, integer count or UNKNOWN,
and brief visible evidence. This gives the adapter explicit source data, including
unresolvable counts. A broad report is still one image-model inference for all
categories. Structured broad output continues to use its count-only schema.

Combined atomic output retains exactly `choice` and `count`. It counts supported
instances and uses B/0 when none are supported; this binary contract cannot
separately express unassessable images or uncertain presence. It should not be
interpreted as proof of clinical absence. Broad narrative uncertainty remains
available to the adapter and is flagged under the existing evaluation policy.
Neither the ontology, counting units, retry policy, nor inference call counts changed.

No notebook configuration change is needed for API analyzers. Copy the updated
`research_prompts.py`, `research_experiments.py`, and `evaluation.py` to Kaggle,
restart the kernel, and start a new API experiment run with `RESUME_RUN_DIR=None`.
Keep `PROMPT_OVERRIDES={}` to use the defaults. Existing overrides take precedence.
For a controlled prompt-profile experiment, set `settings.prompt_profile` to
`"api"` or `"dentalgpt"` on any model or experiment, independently of its backend.
The evaluation exports include `prompt_profile`; older saved runs are labeled
`legacy/dentalgpt` rather than retrospectively labeled as the new API prompts.

To inspect the exact generated prompt after Cell 15.01:

```python
from research_experiments import question
job = next(j for j in SUITE_PLAN["jobs"] if j["strategy_id"] == "atomic_arch")
print(question(job, "combined", "atomic_1", job["conditions"][0], job["regions"][0][1]))
```

Design rationale: explicit scope, output format, and constraints follow the general
guidance in [Google's prompt design strategies](https://ai.google.dev/gemini-api/docs/prompting-strategies)
and [Anthropic's prompt engineering overview](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview).
These are testable starting prompts, not evidence that any model is clinically
accurate or that the new prompts outperform the previous ones. Compare on the same
images with a fixed adapter; inspect finding metrics, count error, unresolved
findings, truncations, forced-zero fallbacks, and cost before selecting a winner.
