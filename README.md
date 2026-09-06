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
(already the notebook's DentalGPT default). Both stages request the tagged reasoning
format without imposing "brief" reasoning. The count's final answer is still
restricted to one integer for deterministic evaluation; Figure 9 instead shows a
prose final answer. Combined JSON remains an explicitly experimental alternative,
not the paper's multiple-choice output format. Its question stem now follows the
same short style. Broad JSON is also an experimental evaluation contract.

Parsing evaluates only the final `<answer>` payload; it does not grade or require
reasoning text. Recovery still retries sampling, then wording, then records B/0.
The training response cap reported on page 6 is 8192 tokens; this is not a proven
inference optimum. Notebook budgets are unchanged. Increase the local model's
`settings.max_tokens` if truncation is frequent, within the server context limit.

Start a new run after these prompt changes. Reload the Python modules (or restart
the kernel and rerun setup) before 15.01. Resume fingerprints now also cover the
rendered prompts, so old prompt runs cannot be silently mixed with new ones.
