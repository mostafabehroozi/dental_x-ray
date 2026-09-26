# DentVLM PAN training-aligned inference

`pan_training_aligned_v1` sends the complete original panoramic image with **12 independent canonical questions**, sequentially, through a single-slot local DentVLM server. Python parses native answers and available rationale locations. A report organizer receives only normalized facts, and code renders the clinical statements.

Matching published task names and released training questions does not guarantee zero false positives, the full training distribution, or equivalence to the authors' native runtime.

## Supported findings and question evidence

The authoritative registry is `dental_pipeline.TASKS`. It drives inference and clinical reports independently of benchmark annotations. All 12 tasks are mandatory.

| Finding | Task ID | Released example |
|---|---|---|
| Impacted Tooth | `impacted_tooth` | `panoramic_2666` |
| Prosthetic Crown | `prosthetic_crown` | `panoramic_5156` |
| Root Canal Therapy | `root_canal_therapy` | `panoramic_2109` |
| Fillings | `fillings` | `panoramic_336` |
| Prosthetic Bridge | `prosthetic_bridge` | `panoramic_2869` |
| Apical Periodontitis | `apical_periodontitis` | `panoramic_5065` |
| Residual Root | `residual_root` | `panoramic_1971` |
| Implant | `implant` | `panoramic_392` |
| Residual Crown | `residual_crown` | `panoramic_1296` |
| Insufficient Space for Primary Tooth Eruption | `insufficient_eruption_space` | `panoramic_2707` |
| Caries | `caries` | `panoramic_4414` |
| Calculus | `calculus` | `upper_1137` |

Questions are copied exactly from English records in the [authors' released training examples](https://github.com/ZJUI-AI4H/DentVLM/blob/9463edb2af47f64510b0681efc20be6ecf870955/data/inst_data_2nd_train.json), with `<image>` handled by the multimodal interface. Each registry entry records its revision, record ID, source modality and evidence type. The independent fixture `tests_fixtures/pan_source_questions.json` verifies these strings offline.

Calculus has PAN task support in Figure 1 of the [paper](https://arxiv.org/abs/2509.23344); its selected exact sentence comes from an **upper-arch** training example. The registry does not describe that sentence as PAN-example evidence. The paper's root-canal example is retained as a documented, unused alternative. Periodontal Disease and unsupported dataset classes are excluded from clinical inference and reports. Crowns and bridges stay separate clinically.

## Notebook workflow

Keep the existing numbered notebook sequence. Cell 3 resolves one baseline:

```python
EXPERIMENTS = xp.build([{"name": dp.PROFILE}], shared={
    "output_root": "/kaggle/working/dentvlm_pan_training_aligned_v1",
    "location_truth": "geometry",
    "evaluate_location": True,
    "counting": False,
    "parser_mode": "code",
})
```

1. Enable a suitable GPU, provide gated checkpoint access through `HF_TOKEN`, and set dataset paths. Secrets are read through the existing environment/Kaggle integration.
2. Build pinned llama.cpp `b10516`. Convert the pinned checkpoint in the **new** model directory, keeping both GGUF files and `dentvlm_provenance.json` together.
3. Leave `LIMIT = 5` initially. Cell 8 runs all 12 questions on up to five images **per dataset**, saving full artifacts. Cell 9 resumes those artifacts without rerunning inference.
4. Inspect runtime preflight, exact questions, raw replies, finish reasons, unresolved outputs and location evidence. Then use `LIMIT = None` and rerun the run/evaluation cells for available splits.
5. Optional `HISTORICAL_RUNS` in the evaluation cell reads old saved runs and compares only identical image hashes. Historical artifacts remain historical.
6. Cell 14 optionally uses the reporter API. The deterministic clinical renderer currently supports English. `dp.dentist_report(result)` requires no reporter API.

The budget is derived from the registry: **12 logical questions per image**, excluding identical transport retries. The optional custom-image cell also runs this fixed contract. Arbitrary prompts, unsupported tasks, regional questions, wording votes and changed-prompt parsing retries are retired. Old configuration keys fail with migration errors.

## Runtime and resumability

The [authors' inference script](https://github.com/ZJUI-AI4H/DentVLM/blob/9463edb2af47f64510b0681efc20be6ecf870955/inference.py) specifies temperature `0.1`, top-p `0.001`, repetition penalty `1.05`, output limit `512`, context `16384` and image bounds equivalent to `4?8192` visual tokens. The llama.cpp deployment explicitly selects penalties ? temperature ? top-p, disables top-k/min-p, uses the full-context repetition window and records seed `0` as an application choice.

Checkpoint: `ZJU-AI4H/DentVLM` at `2ad8e71ea6708eee92723e7eca6e30e6dac48d85`. Conversion records the checkpoint/template, converter revision and content hashes of the Q8 model and matching f16 projector. HF GGUF downloads require a pinned repository commit and the conversion sidecar. Existing files without provenance are rejected; use a fresh conversion directory rather than attaching invented provenance to old files.

Before inference the server verifies effective generation settings through `/props`, process image-bound flags, model identity and all canonical multimodal chat wrappers through `/apply-template`. Its internal media placeholder is normalized for comparison with the authors' image marker; actual image embedding/preprocessing equivalence remains unverified. An unidentified or mismatched running server cannot be reused. See the [pinned runtime API](https://github.com/ggml-org/llama.cpp/blob/b10516/tools/server/README.md).

Q8 GGUF remains a quantized approximation to the authors' native runtime. Passing software tests does not validate real template rendering or model behavior: the authorized notebook must pass its live preflight.

Artifacts use `dentvlm-pan/1`, reports `dentvlm-findings/2`, and evaluation `dentvlm-evaluation/2`. Manifests hash the full registry, exact questions, parser version, system message, verified template, runtime/model identity and generation settings. Raw response caches require matching image bytes, request and runtime identities. Resume also checks image hashes. New runs cannot resume old schemas or incompatible manifests. Use the new output root; historical results are never migrated in place.

## Parsing, location and reports

A clear leading Yes/No on the first nonempty line determines diagnosis. Missing, ambiguous, explicitly contradictory, truncated and failed answers remain unresolved; they are never converted to No. Transport retries repeat the identical request. Every saved call retains the exact historical question, full response, finish reason, error/parse outcome and question provenance. Runtime provenance is saved with results and manifests. Incidental diagnoses in another task's rationale never create findings.

The nine descriptors in [Supplementary Note S1/Table S6](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-026-75718-x/MediaObjects/41467_2026_75718_MOESM1_ESM.pdf) retain their six-cell source-frame mapping. Exact matches are preserved for scorer reproduction, including both-arch descriptors. Reporting separately filters negated, uncertain and ambiguously associated mentions. A positive diagnosis without a usable location stays positive. Unmentioned regions mean **not stated**, not negative.

Figure 1 and Table S6 disagree about laterality. Reports therefore use upper/lower and anterior/posterior, with posterior side unresolved. Raw source descriptors stay in evidence. No benchmark-driven side selection is offered. Optional occupied-source-region diagnostics are not tooth or lesion counts; counting is off by default.

Reports include only the 12 canonical findings, grouped into pathology and existing treatments. The reporter sees normalized IDs, statuses and permitted locations, without raw rationale text. Validation rejects invented IDs, altered statuses/regions, extra clinical prose and impressions referencing nonpositive findings. Rendering uses validated facts only; a failed report falls back to the deterministic summary. Historical Q/A evidence comes from saved calls, never reconstructed current prompts.

## Evaluation

`benchmark_schema.py` owns the original 14 UMFIH YOLO class IDs. Unsupported labels remain loadable for exclusion/accounting but never dispatch inference. Only the annotated supported intersection is scored:

- Primary: implants, combined crown-or-bridge, fillings, root canal treatment, caries, impacted teeth and residual roots.
- Proxy, separate from primary summaries: periapical-lesion truth versus Apical Periodontitis prediction.
- No current benchmark truth: Residual Crown, Insufficient Space for Primary Tooth Eruption and Calculus. Their predictions remain reportable and appear in `task_coverage.csv`; no accuracy is invented.

Crown/bridge aggregation exists only at this boundary: any Yes ? Yes; both No ? No; otherwise unresolved. Missing label files and unannotated classes are never negative truth.

Tables lead with FP, false-positive rate, precision and denominators, then TP/sensitivity. Unknown outputs are excluded from binary confusion counts, with positive/negative truth counts, resolved coverage and all-eligible accuracy that gives abstentions no success credit. Exact and proxy rows are explicitly labeled. Caries and Calculus have separate task rows.

Localization is secondary: conditional region IoU on correctly detected positives, unlocated cases and overall localization coverage. Unstated cells have no negative status in new results. DENTEX FDI annotations take precedence; six-cell left/right scores still assume the supplement's unvalidated source-frame translation. UMFIH geometry or independent LLM/area adapters are approximate secondary truth. DentVLM-generated location truth is rejected, and partial/excluded location truth is not scored as a complete set.

## Validation

Run `python -m pytest -q` for offline acceptance. Tests cover source questions, independent request shape, malformed/truncated/failed responses, all nine descriptors, conservative location reporting, fixed dataset IDs, missing truth, proxies, report validation, cache/resume boundaries and mocked runtime preflight.

Real five-image smoke checks, full-split false positives, uncertainty and comparison to compatible historical predictions must be measured in the notebook environment. No clinical accuracy or native-runtime equivalence is asserted by offline acceptance.
