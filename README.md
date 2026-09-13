# DentVLM panoramic findings pipeline

A small wrapper that gets findings, their regions, and a multiplicity signal out
of DentVLM (a 7B dental vision-language model) on panoramic radiographs, while
sending it only questions it was trained and evaluated on. Everything else
(task decomposition, aggregation, scoring) happens in Python.

## Experiments

The notebook runs a list of configurations, not one. Cell 3 holds one dictionary
per experiment: a name, plus the knobs that experiment changes. Everything it
does not mention comes from the `SHARED` dictionary above it, and then from
`experiments.DEFAULTS`:

```python
EXPERIMENTS = xp.build([
    {"name": "base"},
    {"name": "three-phrasings", "phrasings": 3},
    {"name": "with-counts", "count_question": True},
    {"name": "gemini", "analyzer": {"provider": "gemini", "model": "gemini-3-pro"}},
    {"name": "dentvlm-local", "backend": "local"},
], shared=SHARED)
```

Anything in `experiments.DEFAULTS` may vary per experiment: the backend (local
DentVLM through llama.cpp or a hosted model), the analyzer, adapter and reporter
models, the seven protocol knobs, the retry budgets, the location truth, the
report language, and the local checkpoint source, context size and image-token
cap. A dictionary knob (`analyzer`, `adapter`, `reporter`) merges key by key, so
changing the model keeps the provider and its request options; every other knob
is replaced. An unknown knob name is an error rather than a silent default, and
so are a duplicate name, an invalid value, and `location_truth="fdm"` without the
local backend.

Every experiment writes into `<output_root>/<name>/<dataset>/`, with its resolved
configuration saved as `<output_root>/<name>/experiment.json`, so two
configurations never share a run directory and each of them resumes on its own.
Cell 8 shows raw replies for each experiment before the run (`smoke_images`, 0
skips it), Cell 9 runs them one at a time (an experiment that fails is reported
and the sweep continues; local experiments restart the llama.cpp server only when
their server settings differ), and Cell 10 translates the ground-truth boxes once
per adapter instead of once per experiment. Cell 11 then scores every experiment
and ranks them:

* `<output_root>/leaderboard.csv`: one row per experiment and dataset - F1,
  sensitivity, specificity, PPV, macro F1, false alarms per image, unparseable
  rate, coverage, not-assessed checks, count MAE, exact region-set rate,
  presence-per-cell F1, calls per image. Each row is scored against that
  experiment's own location truth.
* `<output_root>/comparison/<dataset>/`: the same experiments compared **paired**
  on the same images against the first one - `paired_f1_delta`, checks corrected
  and worsened, newly unresolved, recorded calls and tokens.

Cells 12 to 14 look at one experiment at a time: `INSPECT_EXPERIMENT` selects it
for the per-finding tables and saved diagnostics, for one image's raw answers,
and for the dentist report (one report call per image, so it defaults to the
inspected experiment).

Running several experiments multiplies model calls. `"limit"` in `DATASETS` keeps
a first sweep cheap, and every cell resumes, so a sweep can be extended, or an
experiment added, without recomputing what is already saved.

## Why it looks like this

DentVLM (Meng et al., Nature Communications 2026; arXiv 2509.23344) is
Qwen2-VL-7B fine-tuned in two stages on 2.46 million bilingual dental VQA pairs:
stage 1 answers only, stage 2 answer plus rationale plus location. For panoramic
X-rays it was trained on 13 yes/no tasks: caries, periodontal disease, impacted
tooth, apical periodontitis, residual root, residual crown, insufficient space
for primary tooth eruption, calculus, prosthetic crown, root canal therapy,
fillings, prosthetic bridge, implant. Each task is asked with one of nine fixed
question templates (Supplementary Table 7). The model answers `Yes` or `No` on
line 1 and then writes a rationale that names the location with one of nine
fixed descriptors ("the left posterior region of the upper dentition", ...).
It has no count task, no "list all findings" task for oral diseases, no
cropped-panoramic training, and no JSON or tag format. So:

* **Presence** uses one Table S7 question per task, worded as in the authors'
  released test set, on the whole image. Line 1 decides, as in the authors'
  scorer; both words or neither is unparseable.
* **Location** is never asked in words. The nine descriptors are read from the
  rationale by exact match, exactly as the authors compute their IoU, and mapped
  onto six dental-arch cells (upper/lower x left/anterior/right).
* **Multiplicity** is the number of cells named (0 to 6), reported as
  "in N region(s)". The tooth-count question from the DentalGPT branch is kept
  behind `count_question` as an explicitly out-of-distribution
  experiment. That knob is the counting switch, and it is off by default: the
  model is asked presence only, and the evaluation scores each finding as
  present or absent per image and per cell (`region_presence.csv`), so a class
  that occurs several times in an image or a cell is scored once. Only
  `count_question=True` adds a count and the count tables.
* **Findings without a DentVLM task** (furcation involvement, apical surgery,
  root resorption, orthodontic appliances, surgical plates) are not asked and
  are reported as "not assessed by this model". `ask_untrained`
  asks them anyway and scores them under `trained_task=False`; the paper's
  zero-shot accuracy on untrained diseases is 52-64%.
* **Prosthetic restoration** (crowns or bridges in the benchmark) is the OR of
  the prosthetic crown and prosthetic bridge tasks, regions merged.
* **Bounded parse recovery.** The notebook sets `parse_retries = 1`: one extra
  attempt per unparseable answer, on the same image/model with a reminder to put
  Yes/No on line 1 and retain the rationale/location. Optional counts use an
  integer-only reminder. Every failed attempt prints the full prompt and response;
  all attempts and the recovery summary are saved. Truncated replies are unresolved.
  Retries repair individual phrasings, not votes: valid but conflicting phrasings
  still follow the existing vote rule. Exhausted results remain `None`, never
  forced negatives. TP/FP/TN/FN and per-image recall exclude unresolved findings;
  complete-case rate excludes images with unresolved findings. Unasked classes
  remain `not_assessed` and are outside the expected checks.
  `expected_finding_checks = scored_finding_checks + excluded_unparseable_checks`.
  This means confusion-table totals can still differ when final coverage differs.
  The recovery policy is hashed into the manifest; give the changed setting a new
  experiment name (its own directory) rather than reusing one. Direct Python `Protocol()` keeps retries off unless specified.
* **Visible failure control.** `api_call_retries` retries transient API failures without hidden SDK
  retries. `location_parse_retries` controls location-format retries and
  `location_failure_policy` selects `geometry`, `exclude`, or `error`. Failure-only console blocks
  print the full prompt and response; saved JSON keeps every attempt. Valid vote ties and task
  conflicts are saved as aggregation warnings. Invalid resumed artifacts stop with `ARTIFACT ERROR`.
  A hashed control that changed under an existing experiment name stops the run instead
  of mixing two configurations.

Left and right follow the model's own convention (Supplementary Table S6): its
"left posterior region" is FDI quadrants 1 and 4, the patient's right, which is
the left side of a panoramic as displayed. `dental_pipeline.LEFT_IS_IMAGE_LEFT`
records that reading, the dentist summary translates cells to the patient's
side, and the notebook checks the convention against DENTEX boxes.

## Files

| File | Role |
| --- | --- |
| `dental_pipeline.py` | task table and verbatim questions, answer and region extraction, protocol knobs, model runner, resumable run loop, dentist summary |
| `dental_eval.py` | ground-truth loaders (UMFIH YOLO, DENTEX with FDI tooth numbers), location truth (adapted, FDI, or fixed windows), metrics incl. presence per cell, side-convention check, CSV/JSON export |
| `dental_analysis.py` | offline phrasing/vote and crop comparisons, recovery, case breakdowns, paired saved-run comparisons |
| `location_adapter.py` | translates ground-truth boxes into the six cells: vision-LLM adapter (numbered boxes drawn on the image), experimental DentVLM spotlight adapter, resumable per-dataset run |
| `llama_runtime.py` | llama.cpp build, one-time GGUF conversion of the Hugging Face checkpoint, GGUF download, server process (with image-token flags) |
| `report_writer.py` | dentist report: dense structured findings per image (tasks, cells, multiplicity, extra tasks, not-assessed findings), report-writer prompts, verification of the reply against the input, one repair turn, Markdown rendering, resumable run |
| `experiments.py` | the experiment table: DEFAULTS, merging and validation of each configuration, per-experiment paths, the runner/adapter/report-writer of one experiment |
| `llm_api.py` | hosted-model access shared by the runner, the adapter and the report writer: provider registry, key lookup (environment variable or Kaggle secret), client construction |
| `main_notebook.ipynb` | Kaggle runner; the experiments to run and compare are Cell 3, the ranking is Cell 11 |
| `test_dental_pipeline.py`, `test_location_adapter.py`, `test_report_writer.py`, `test_location_scoring.py`, `test_llm_api.py`, `test_experiments.py` | offline tests with fake models (`python -m unittest -q`) |

## Small evaluation comparisons (Cells 11 and 12)

Cell 11 saves, and Cell 12 displays, compact diagnostics with supporting image IDs
in `<output_root>/<experiment>/<dataset>/evaluation/evaluation.json` and matching CSV files.
The existing finding, count and location scoring rules are preserved.

| Table | DentVLM-specific comparison |
| --- | --- |
| `stage_changes` | First saved phrasing vs whole-image vote, and whole-image vs crop outcomes. Includes corrected errors, new errors, unchanged, unresolved and not-assessed outcomes; overall and per finding. |
| `phrasing_votes` | Agreement, disagreement, ties, and unresolved phrasings per task. Available when multiple phrasing answers were saved. |
| `region_vote_comparison` | Union vs majority using identical saved rationale answers and the existing vote/OR rules. Only for rationale mode with multiple saved phrasings; crop locations do not use this vote. |
| `parse_recovery` | First-pass, recovered, unresolved questions by task/stage, plus correctness where ground truth supports it. Each phrasing is a separate question. |
| `call_usage` | Recorded analyzer completions, tokens and latency, split into first attempts and parse retries. Missing usage is unavailable, with recorded-call denominators; transport attempts are not separate saved completions. |
| `case_breakdown` | Trained/untrained task support, instance counts, other findings, named/true cells, boundary-crossing boxes, and location-truth sources. Unasked findings remain not assessed. |

First-phrasing and region-vote replay use saved answers **after any parse repairs**;
they do not simulate retries OFF or change predictions. Crown and bridge are
combined with the existing OR rule before finding scoring. Their individual retry
answers cannot be graded from the merged restoration label, so correctness is
unavailable for those tasks; extra tasks without benchmark labels are likewise
unscored. A recovered parse can still be wrong. Named-cell multiplicity is never
treated as a tooth count; count metrics require the optional count question.

Cell 11 compares the experiments of Cell 3 automatically: every experiment with a
complete set of results for a dataset is scored paired against the first one and
written to `<output_root>/comparison/<dataset>/`. `run_comparison` reports each
experiment's knobs (`phrasings`, `region_vote`, `location`, `count_question`,
`ask_untrained`, `extra_tasks`, `parse_retries`), coverage, metrics and recorded
usage; `run_changes` contains paired outcomes and image IDs. Every selected image
must exist in each run with identical image hashes and a consistent saved
protocol; an experiment still missing images is left out of the paired table (its
own row stays in the leaderboard). Extra unselected images are ignored. Location
comparisons require matching saved cell-side conventions, so the paired table
leaves location out and the leaderboard scores it per experiment. All runs use the
same supplied ground truth.

Paired F1 uses only findings asked and resolved in both runs. Newly assessed and
unresolved transitions remain separate, so enabling untrained tasks cannot be
counted as repairing old errors. Coverage is scored/expected **asked** checks;
not-assessed checks have their own column. Count and location metrics retain each
run's true-positive subset. Case groups describe associations, not causal effects;
empty denominators are unavailable. Reload the updated project imports and rerun
Cell 11 with ground truth/location adaptation already loaded; no model calls are
made. Older artifacts without task answers or retry metadata omit those analyses.

## Calls per image

13 whole-image calls, the same for every image (one per task; crown and bridge
are separate tasks). Optional knobs in `dental_pipeline.Protocol`:

* `phrasings=3`: ask three verbatim wordings per task and vote (majority for
  yes/no; `region_vote="union"` is the paper's matching voting, `"majority"`
  its majority voting). 39 calls per image. In-distribution.
* `count_question=True`: one count call per positive countable finding.
  Out-of-distribution.
* `location="crops"`: the primary question for every task on each of the six
  cell crops, whatever the whole image answered (kept as `whole_image`). A task
  is present when any cell answers Yes, so a finding missed with the model's
  attention spread over the whole image can be recovered in a cell, and the
  cells answering Yes are its regions. 13 + 6 x 13 = 91 calls per image.
  Cropped panoramics are outside the model's image distribution, so this is a
  comparison, not the default.
* `location="none"`: presence only.

## Runtime settings that matter

* The model is published only as bf16 safetensors (Hugging Face
  `ZJU-AI4H/DentVLM`, gated with automatic approval, CC BY-NC 4.0). Accept the
  license once, store a token as a Kaggle secret, and let Cell 6 convert it once
  with llama.cpp's converter (Q8_0 language model, f16 vision projector, about
  9.5 GB kept, 17 GB scratch). Keep the two files in a private Kaggle dataset
  or your own Hugging Face repo for later sessions.
* `--image-max-tokens 8192` mirrors the authors' `max_pixels` of 8192 x 28 x 28;
  llama.cpp would otherwise cap Qwen2-VL images at 4096 tokens. No token floor.
  The paper's ablation found a 1024 x 1024 bound best for disease tasks;
  an experiment with `"image_max_tokens": 1369` reproduces it for an A/B.
* `--ctx-size 16384`, the authors' maximum input length.
* `max_tokens 512` (the authors' output cap), temperature 0, `repeat_penalty
  1.05` as in their inference script. Line 1 is present even when a rationale
  is cut off, so truncated replies are still graded.
* No system prompt is sent; the chat template injects Qwen's default one, which
  the authors' vLLM script sets explicitly.
* Radiographs must be JPEG, PNG, or BMP (what llama.cpp can decode).
* `backend="api"` in an experiment sends the same questions to a hosted vision model
  instead, for a controlled comparison (see "Hosted models").

## Hosted models

Cell 3 has one `PROVIDERS` registry containing each provider's base URL and API
key. The keys come from environment variables or Kaggle Secrets (Add-ons >
Secrets), and unused providers may have no key. The small `analyzer`,
`adapter` and `reporter` role dictionaries of an experiment then select any provider and exact model, e.g.
`{"provider": "openrouter", "model": "qwen/qwen3-vl-235b-a22b-thinking"}`.
Model-specific options such as `token_param`, `temperature`, and OpenRouter
routing under `request_options` stay with the role. `VisionRunner.from_api`,
`LLMAdapter.from_api` and `ReportWriter.from_api` build the clients; run manifests
record the public role configuration, never the provider key. Transport
errors, rate limits and 5xx replies are retried by the client with backoff; a
bad request or key fails at once.

## Dentist report

DentVLM's answers are a dozen narrow facts per image (one yes/no per task,
regions named in rationales, a multiplicity); a dentist reads one report. Cell
14 sends the findings of each image to a text LLM (the `reporter` role of the
experiment, a spec dict like `analyzer` and `adapter`; it never sees the image) and
saves a classified report in the dentist's language (`report_language`).
`report_writer.py` does it in three fixed steps, shaped by what DentVLM
actually produces:

1. **Structured input.** `structured_findings` condenses a result JSON into one
   dense object: the 14 benchmark findings plus the three extra DentVLM tasks
   (residual crown, insufficient eruption space, calculus), in seven sections
   (restorations and prostheses, endodontic, caries, periodontal, periapical,
   teeth and eruption, appliances and hardware). Every entry carries an
   explicit `status` (`present`, `absent`, `unparseable`, or `not_assessed`
   for findings DentVLM has no task for), the task or tasks that decided it
   with their verbatim question and parsed answer (so the writer can say that
   the bridge, not the crown, answered Yes), every dental-arch region with an
   explicit value (`named` / `not_named` from the rationale, or
   `present` / `absent` / `unparseable` from cell crops), the regions the
   finding was located in on the patient's side, the multiplicity, the
   optional out-of-distribution count, a `trained` flag for zero-shot
   questions, and a `detection` note for the crop comparison. A legend, the
   analyzer's method and its limitations go with it, so nothing is implicit
   and nothing is null. The rationale text itself stays out unless
   `include_rationale` is set: by default the report rests on the same parsed
   answers the evaluation scores.
2. **One call, fixed prompt.** `SYSTEM_PROMPT` and `USER_PROMPT` ask for a
   radiology-style report as JSON: a title and localized headings, one entry
   per finding with its status copied and a statement in the dentist's
   language, an impression with pathology before treatment history, the
   unparseable findings under "not assessable", and limitations. The writer
   may reword and organise; it may not add, drop, soften or upgrade a finding,
   estimate a count, name a tooth, report a region the model did not name as
   free of the finding, or give a diagnosis, severity or advice. Untrained
   questions, crop-only detections and experimental counts must be called
   what they are.
3. **Verification and rendering.** `verify_report` checks the reply against
   the input: every finding exactly once, in its section, with its status
   unchanged, no unknown finding, impression and limitations present, "not
   assessable" naming the unparseable findings and nothing else. A failing
   reply goes back once with the list of problems (`REPAIR_PROMPT`); a reply
   that still fails is saved with its problems and the deterministic
   `dentist_report` takes its place in the Markdown, so the failure is visible
   and the dentist still gets a summary. Verified JSON is rendered to Markdown
   deterministically (sections in a fixed order; ● present, ○ absent,
   ? unparseable, – not assessed).

Reports resume like the other loops: one `.json` (structured input, prompt,
every attempt with its problems, the verified report, the Markdown) and one
`.md` per image under `<dataset>/reports/<model>-<language>/reports/`, with a
manifest that hashes the writer settings, the prompts and the language.
`summarize_reports` counts how many reports verified at once, after a repair,
or fell back. Reports are for reading and are not scored: Cell 11 stays the
measure of the analyzer. The language is not verified; read one report before
trusting a batch.

## Datasets

* **UMFIH 14-class set** (Zenodo 15487430, CC BY 4.0 with a non-commercial
  note): the exact ontology, YOLO boxes. Score the 100-image internal test split
  and the 180 external-validation images separately.
* **DENTEX** (Hugging Face `ibrahimhamamci/DENTEX`, CC BY-NC-SA): FDI quadrant
  and tooth labels for caries, periapical lesion, and impacted tooth, which
  give exact cell truth through the model's own tooth-region mapping. Use the
  fully labeled train split (`training_data/quadrant-enumeration-disease`, 705
  images) and the validation split (`validation_triple.json`, 50 images).
  DentVLM's authors used only the 242-image official test split for their
  external validation, so both are held out. That test split ships as raw
  LabelMe files with unmapped Turkish labels and is not supported.

Each dataset is scored on the findings it annotates and the model was asked
about; unannotated findings are not counted as negatives.

## Location truth

Ground truth is numeric (boxes); DentVLM reports six cells. Scoring location
means deciding which cells each true box occupies, and fixed image windows do
that badly: the midline, the canine line and the occlusal plane move with
patient positioning and the shape of the arch. DentVLM's authors built their
own location labels anatomically (box, nearest teeth, tooth-region mapping;
Methods 4.2), and `location_truth` picks how this project does it:

* `"llm"` (recommended): `location_adapter.LLMAdapter` draws numbered boxes on
  the radiograph, burns the FDI quadrant names into the corners, and asks a
  strong vision model through any OpenAI-compatible API (GPT-5, Gemini's
  compatibility endpoint, NIM, ...) for the dental-arch units each box
  occupies: FDI quadrant x anterior (incisors and canine) / posterior
  (premolars, molars and behind), plus the FDI tooth positions. Units are the
  finest division the six cells are made of, so the mapping onto cells is
  deterministic (`dental_pipeline.unit_cell`) and follows the same
  `LEFT_IS_IMAGE_LEFT` reading as the model's own words. One call per image
  (chunked above `max_boxes_per_call` boxes), strict JSON back, with bounded
  retries and the configured location failure policy. The model is the `adapter` role of the experiment (see "Hosted models");
  for reasoning models set `token_param` to `max_completion_tokens` and leave
  `temperature` at `None`.
* `"fdm"` (experimental): DentVLM itself. It has no question about a marked
  region, so the task is split into one in-distribution question per box: a
  full-frame "spotlight" copy that shows only the box and a margin, the
  finding's own Table S7 question, and the descriptor DentVLM writes in its
  rationale. The truth then lives in the model's own convention, but masked
  panoramics are outside its image distribution; boxes it answers "No" to, or
  findings without a DentVLM task, fall back to the windows.
* `"geometry"`: the fixed cell windows (a box counts in every window holding at
  least a quarter of its area; the windows overlap on the canine line and the
  occlusal plane). No model calls.

The adapter runs once per dataset and adapter (Cell 10), independently of the
model run, and resumes: one JSON per image under
`<dataset>/location_truth/<adapter>/boxes` with the raw reply, the units, the
cells, the windows' answer and the source that placed the box; the drawn or
spotlighted images are kept under `.../drawn` for audit. DENTEX boxes carry FDI
tooth numbers, which give exact cells, so on a DENTEX dataset Cell 10 also
prints the adapter's and the windows' agreement with that exact truth
(`dental_eval.truth_agreement`): the check that the adapter is worth its calls.
`evaluation.json` records under `summary.location_truth` how many true boxes
each source placed.

## Evaluation outputs

`<output_root>/<experiment>/<dataset>/evaluation/` holds `presence.csv` (TP, FP, TN, FN,
unparseable, sensitivity, specificity, PPV, F1 per finding, with a
`trained_task` flag), `whole_image.csv` (the same table for the whole-image
answers alone with `location="crops"`: read the two side by side to see what
the cells recovered and what it cost in specificity), `region_presence.csv`
(presence per finding and cell: every cell of every asked image, whatever the
whole image answered, against the cells the true boxes occupy, as TP, FP, TN,
FN, unparseable, sensitivity, specificity, PPV and F1; one cell per image, so
several boxes in a cell are one presence; with `location="rationale"` a named
cell is the prediction and an unnamed cell counts as not predicted, with
`"crops"` each cell's own answer counts and an unparseable one is an excluded
cell; an image whose true boxes could not be placed is left out and counted
under `location_truth_excluded`), `regions.csv` (per-cell TP, FP, TN, FN,
exact-set match, Jaccard, unlocalized rate, over the localized true positives
only), `counts.csv` (only when the count question was asked), `per_image.csv`,
and `evaluation.json` with a summary: micro and macro F1, complete-case rate,
mean false alarms per image, the presence-per-cell micro numbers under
`region_presence` (the leaderboard's `region_f1`), and the list of findings not
assessed. `regions.csv` and `region_presence.csv` answer different questions:
the first scores where a detected finding was placed, the second whether each
cell was called correctly at all, absent findings included. Both need the
location truth, so they follow `evaluate_location`.

## Caveats

* The paper reports DentVLM's own location IoU at about 38% on its test set, so
  cell-level location is a coarse signal; the presence answer is the reliable
  part. Location truth from the LLM adapter is itself a model output: read the
  DENTEX agreement numbers and spot-check the drawn images before trusting the
  region rows.
* Caries and calculus are the weakest panoramic tasks in the paper (about 63%
  and 62%); implants, bridges and crowns the strongest (above 90%).
* Ground-truth boxes are per instance while the model names regions, so the
  region count is a lower bound on the number of occurrences.
* Apical surgery, root resorption, and furcation have very few positives in
  UMFIH and no DentVLM task; their rows are not statistically meaningful even
  with `ask_untrained=True`.
* The weights are CC BY-NC 4.0: research use only.
## Optional location scoring

In an experiment, set `evaluate_location = True` (default) to score locations,
or `False` to skip location scoring and location-truth adapter calls. Finding
scores and total-count scores remain enabled; inference, counting questions,
and saved predictions are unchanged. Presence per cell follows this switch.
Re-run Cell 3, Cell 10, and Cell 11 to evaluate existing results with this setting;
no inference rerun is required. Cell 13 also skips its side check when disabled.
The report records `summary.evaluate_location`. Re-exporting a report with location
scoring disabled removes its previous location CSVs so stale metrics are not shown.
