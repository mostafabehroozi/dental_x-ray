# DentalGPT panoramic findings pipeline

A small wrapper that gets findings, counts, and coarse locations out of
DentalGPT (a 7B dental vision-language model) on panoramic radiographs, while
sending it only question shapes it was trained and evaluated on. Everything else
(task decomposition, location, aggregation, scoring) happens in Python.

## Experiments

The notebook runs a list of configurations, not one. Cell 3 holds one dictionary
per experiment: a name, plus the knobs that experiment changes. Everything it
does not mention comes from the `SHARED` dictionary above it, and then from
`experiments.DEFAULTS`:

```python
EXPERIMENTS = xp.build([
    {"name": "base"},
    {"name": "separate-questions", "question_form": "separate"},
    {"name": "whole-image-only", "presence_level": "overall", "count_level": "overall"},
    {"name": "presence-only", "counting": False},
    {"name": "gemini", "analyzer": {"provider": "gemini", "model": "gemini-3-pro"}},
    {"name": "dentalgpt-local", "backend": "local"},
], shared=SHARED)
```

Anything in `experiments.DEFAULTS` may vary per experiment: the backend (local
DentalGPT through llama.cpp or a hosted model), the analyzer, adapter and
reporter models, the six protocol knobs, the retry budgets, the location truth,
the report language, and the local checkpoint, context size and image-token cap.
A dictionary knob (`analyzer`, `adapter`, `reporter`) merges key by key, so
changing the model keeps the provider and its request options; every other knob
is replaced. An unknown knob name is an error rather than a silent default, and
so are a duplicate name, an invalid value, and `location_truth="fdm"` without
the local backend.

Every experiment writes into `<output_root>/<name>/<dataset>/`, with its
resolved configuration saved as `<output_root>/<name>/experiment.json`, so two
configurations never share a run directory and each of them resumes on its own.
Cell 8 runs them one at a time (an experiment that fails is reported and the
sweep continues with the next one, and local experiments restart the llama.cpp
server only when their server settings differ). Cell 9 translates the
ground-truth boxes once per adapter instead of once per experiment. Cell 10 then
scores every experiment and ranks them:

* `<output_root>/leaderboard.csv`: one row per experiment and dataset - support,
  TP/TN/FP/FN, coverage, sensitivity, specificity, PPV, micro/macro F1, false
  alarms, count accuracy, location accuracy and calls per image. Each row is
  scored against that experiment's own location truth.
* `<output_root>/overview/`: joined tables across every experiment: the overall
  leaderboard, one row per experiment and finding, one row per experiment and
  diagnostic situation, and the full experiment x situation x finding drill-down.
* `<output_root>/comparison/<dataset>/`: the same experiments compared **paired**
  on the same images against the first one - raw confusion counts, coverage,
  `paired_f1_delta`, checks corrected and worsened, newly resolved/unresolved,
  recorded calls and tokens.

Cells 11 to 13 look at one experiment at a time: `INSPECT_EXPERIMENT` selects it
for the per-finding tables, for one image's raw answers, and for the dentist
report (one report call per image, so it defaults to the inspected experiment).

Running several experiments multiplies model calls. `"limit"` in `DATASETS`
keeps a first sweep cheap, and every cell resumes, so a sweep can be extended,
or an experiment added, without recomputing what is already saved.

## Why it looks like this

DentalGPT (arXiv 2512.11558) was fine-tuned from Qwen2.5-VL-7B and then trained
with reinforcement learning on multiple-choice questions. The only panoramic
skill the paper measures is one condition per question, answered True/False
(Figure 7, 84% accuracy). Counting appears once, as a tooth count with a prose
answer (Figure 9). Location is never asked in the paper, but the panoramic
benchmark it was scored on (MMOral-OPG-Bench, arXiv 2509.09254) asks "in which
jaw" and counts "in the lower jaw", and the model itself, in Figure 9, counts
per jaw and walks "the right upper quadrant ... the left lower quadrant" by
name. So:

* **Presence** uses the Figure 7 wording verbatim, one finding per call, on the
  whole image and, with regions on, again for every finding in every region.
  The whole-image answers are kept as a separate result and never decide which
  regional questions are asked, so a finding the model misses with its
  attention spread over the whole image can be recovered when the question is
  narrowed to one region and one finding.
* **Counts** use the Figure 9 wording for fillings and the same "How many teeth
  ..." shape for the other tooth-anchored findings. Findings whose boxes are
  regions or devices (bone loss, furcation, apical surgery, appliances, plates)
  are presence-only.
* **Regions** are the two jaws ("the upper jaw", benchmark-verbatim) or the four
  FDI quadrants ("the upper right quadrant", the model's own words, patient's
  side). Nothing finer is ever named. A region is put to the model either in
  words on the whole image (the image stays in distribution, no seams for
  counts) or as a crop with the verbatim whole-image question (the question
  stays verbatim, the picture does not).
* **Bounded parse recovery.** The notebook sets `parse_retries = 1`: one extra
  attempt for an unparseable answer, on the same image/model with an output-format
  reminder. `0` means warn only. Each failure prints the full prompt and response.
  All attempts and the recovery summary are saved. A valid presence decision is
  retained when only its count needs repair. Exhausted fields remain `None`, never
  a forced zero or negative. TP/FP/TN/FN and per-image recall exclude unresolved
  findings; complete-case rate excludes images with unresolved findings.
  `expected_finding_checks = scored_finding_checks + excluded_unparseable_checks`.
  This means confusion-table totals can still differ when final coverage differs.
  The recovery policy is hashed into the manifest; give the changed setting a new
  experiment name (its own directory) rather than reusing one. Direct Python `Protocol()` keeps retries off unless specified.
* **Visible failure control.** `api_call_retries` retries transient API failures without hidden SDK
  retries. `location_parse_retries` controls location-format retries and
  `location_failure_policy` selects `geometry`, `exclude`, or `error`. Failure-only console blocks
  print the full prompt and response; saved JSON keeps every attempt. Invalid resumed artifacts stop
  with `ARTIFACT ERROR`. A hashed control that changed under an existing experiment name
  stops the run instead of mixing two configurations.

The only public weights are the GGUF conversion of `DentalGPT-7B-1026`. That
checkpoint may predate the reinforcement-learning stage, and the exact sentence
the authors appended to request `<think>/<answer>` tags is unpublished. A short
probe therefore decides once per local experiment whether the suffix is needed
("plain" or "tagged"), and the run manifest records the choice, so a resumed
experiment keeps it.

## Two levels

`dental_pipeline.Protocol`, built from the six question knobs of an experiment, has:

| Knob | Values | Meaning |
| --- | --- | --- |
| `counting` | `True`, `False` | `False`: no count question of any kind. The model only decides presence, on the whole image and, with `presence_level="region"`, in every region, so a finding's result is its presence and its region set; `count_level` and `question_form` then have no effect and every question is the bare Figure 7 question on both backends (the manifest records no count wording). The evaluation keeps the presence tables, adds presence per region (`region_presence.csv`, see "Evaluation outputs") and leaves the count tables out, so a class with several boxes in one image or one region is scored once, as present. |
| `presence_level` | `overall`, `region` | `overall`: presence from the whole-image question only. `region`: the same question for every finding in every region, region by region, independent of the whole-image answers (kept under `whole_image`). A finding is present when any region answers A and absent only when every region answers B; the region set is the regions that answer A. |
| `count_level` | `overall`, `region` | `overall`: one whole-image count per positive countable finding. `region`: one count per region, asked right after a region answers A when `presence_level="region"`, else in every region for every countable finding; the finding's count is the sum, and a region count above zero also localizes the finding. A region count of 0 is a valid answer. |
| `region_scheme` | `quadrant`, `arch` | UR, UL, LL, LR (patient-side FDI names) or upper, lower. |
| `region_prompt` | `words`, `crop` | `words`: "Kindly evaluate if the condition 'X' is present in the upper right quadrant of this image." and "How many teeth in the upper right quadrant have ..." on the whole image. `crop`: the whole-image questions on the region crop. |
| `question_form` | `separate`, `combined` | `separate`: the Figure 7 presence question, then the Figure 9-shaped count question for a positive countable finding (DentalGPT's shapes). `combined`: for hosted models, one presence-and-count question wherever the separate form would ask both in the same scope (see "Combined presence-and-count question"); the five presence-only findings keep the bare question. |

The whole-image wording is byte-identical in every configuration; the region
wording only fills a scope slot of the same sentence (`REGION_PHRASES`,
`COUNT_TEMPLATES`). Quadrant words are the patient's sides, so "the upper right
quadrant" is the image-left window; `QUADRANT_WORDS_ARE_PATIENT_SIDE` records
that reading and the DENTEX side check (below) confirms it.

## Calls per image

R = regions in the scheme (4 or 2), Pc = positive countable findings, Rp =
region-finding pairs that answered A for a countable finding.

| `presence_level` / `count_level` | Calls | Example (Pc=4, Rp=6, quadrants) |
| --- | --- | --- |
| overall / overall | 14 + Pc | 18 |
| region / overall | 14 + R x 14 + Pc | 74 |
| overall / region | 14 + R x 9 | 50 |
| region / region | 14 + R x 14 + Rp | 76 |

The regional calls never depend on what the whole image answered, so an
all-negative image needs 14, 70, 50 or 70 calls. With `region_prompt="words"`
every call reuses the cached image prefix; with crops the loop is region-major,
so each crop's prefix is built once. With `counting=False` there is no count
call at all: 14 calls with `presence_level="overall"`, 14 + R x 14 with
`"region"`, whatever the image holds and whichever backend answers.

With `question_form="combined"` the count calls disappear. Rf = findings a
region answered A for while the whole image did not (only `region / overall`
asks a count for those):

| `presence_level` / `count_level` | Calls | Example (quadrants) |
| --- | --- | --- |
| overall / overall | 14 | 14 |
| region / overall | 14 + R x 14 + Rf | 70 + Rf |
| overall / region | 14 + R x 9 | 50 |
| region / region | 14 + R x 14 | 70 |

## Combined presence-and-count question (hosted models)

DentalGPT is asked one fact per call because that is the shape it was trained
on. A capable hosted model does not need the split: with
`question_form="combined"` (Cell 3's default for `backend="api"`) each of the
nine countable findings is asked one presence-and-count question per scope, and
the reply ends in two fixed lines:

```text
Answer: A. True
Count: 3
```

or `Answer: B. False` / `Count: 0`. The five presence-only findings keep the
bare Figure 7 question, and the local model keeps the separate questions. The
prompt (`dental_pipeline.COMBINED_QUESTION`) is longer than anything sent to
DentalGPT, but the two task sentences inside it are the ones the separate form
sends, verbatim: the Figure 7 sentence (whole image, or with the region named)
and the Figure 9-shaped count sentence with the same scope. Around them it
states the display convention (the patient's right is on the image's left),
defines the finding and its counting unit in one line each (`DEFINITIONS`; edit
there only), pins the scope (the whole radiograph, a crop, or one quadrant or
jaw named both anatomically and as an image half, `SCOPE_NOTES`), and forbids
the two inconsistent pairs (A with 0, B with more than 0). The model may reason
first; only the last `Answer:` and `Count:` lines are read, with the lenient
rules of the separate form as a fallback.

The combined question replaces a presence question exactly where the separate
form would have followed it with a count question in the same scope, so the
results keep the same fields:

* `count_level="overall"`: the whole-image question of a countable finding is
  combined and yields `whole_image` and the count; the regional questions stay
  presence-only. A finding the whole image answered B but a region answered A
  is present by the regional rule and still gets the Figure 9 whole-image count
  question, the only extra call the form ever makes.
* `count_level="region"`: the whole-image questions stay presence-only and the
  regional question of a countable finding is combined. A region's count is
  stored when it answers A (`presence_level="region"`), or for every region,
  0 with a B (`presence_level="overall"`), as the separate form stores them.

Grading never guesses: a reply without a readable letter yields nothing; a
count that contradicts the letter is unparseable while the letter stands, and
shows up under `count_unparseable`. Every combined call records the raw pair
and whether it was consistent (`calls[*].parsed`), so a rejected count stays
visible. The quadrant words of the combined form are always the patient's and
the scope note names the image half, so `QUADRANT_WORDS_ARE_PATIENT_SIDE` does
not apply to it; a low `side_agreement` on a combined run means the model
ignored the scope note, and crops are the remedy. Force `"separate"` on the API
backend for a prompt-for-prompt comparison with DentalGPT; the manifest hash
follows the form and the combined wording, so the two never mix in one run
directory.

## Files

| File | Role |
| --- | --- |
| `dental_pipeline.py` | prompts and region wording, answer extraction, crops, model runner, probe, `Protocol`, resumable run loop, dentist summary |
| `dental_eval.py` | ground-truth loaders (UMFIH YOLO, DENTEX with FDI labels), location truth (adapted, FDI, or fixed windows), metrics incl. presence per region, per-region counts and the side check, CSV/JSON export |
| `dental_analysis.py` | offline regional changes, parse recovery, case breakdowns, and paired saved-run comparisons |
| `location_adapter.py` | translates ground-truth boxes into the region windows: vision-LLM adapter (numbered boxes drawn on the image), experimental DentalGPT multiple-choice adapter, resumable per-dataset run |
| `llama_runtime.py` | llama.cpp build, GGUF download, server process (with image-token flags) |
| `report_writer.py` | dentist report: dense structured findings per image, report-writer prompts, verification of the reply against the input, one repair turn, Markdown rendering, resumable run |
| `experiments.py` | the experiment table: DEFAULTS, merging and validation of each configuration, per-experiment paths, the runner/adapter/report-writer of one experiment, the probe decision |
| `llm_api.py` | hosted-model access shared by the runner, the adapter and the report writer: provider registry, key lookup (environment variable or Kaggle secret), client construction |
| `main_notebook.ipynb` | Kaggle runner; the experiments to run and compare are Cell 3, the ranking is Cell 10 |
| `test_dental_pipeline.py`, `test_location_adapter.py`, `test_report_writer.py`, `test_location_scoring.py`, `test_llm_api.py`, `test_experiments.py` | offline tests with fake models (`python -m unittest -q`) |

## Runtime settings that matter

* `--image-max-tokens 6144`: llama.cpp otherwise caps Qwen2.5-VL images at 4096
  tokens and silently downscales a full-size panoramic below training resolution.
* No `--image-min-tokens` floor, so crops of small panoramics stay at native
  size, as they would under the Hugging Face processor.
* `--ctx-size 16384` so image tokens, question, and a long `<think>` fit.
* `max_tokens 4096`, temperature 0, `repeat_penalty 1.05` (the same value as the
  backbone's generation config; llama.cpp applies it over the last 64 tokens).
* Replies cut off by `max_tokens` without a closed `<answer>` are graded as
  unparseable, never as an answer.
* No system prompt is sent; the GGUF chat template injects Qwen's default one,
  matching the authors' published inference snippet.
* Use Q6_K or Q8_0 weights with the f16 projector for reported numbers.
* Radiographs must be JPEG, PNG, or BMP (what llama.cpp can decode).
* `"backend": "api"` in an experiment sends the same prompts to a hosted vision
  model instead, for a controlled comparison against DentalGPT (see "Hosted
  models"); local and hosted experiments can sit in the same list.

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
bad request or key fails at once. With the API backend `mode="auto"` resolves
to `"plain"` without a probe: the `<think>/<answer>` suffix is a DentalGPT
training artifact. `question_form="auto"` (the default) resolves to `"combined"`
on this backend, so the nine countable findings are asked presence and count in
one question (see "Combined presence-and-count question"), and to `"separate"`
on the local one; set it explicitly for a prompt-for-prompt comparison.

## Dentist report

The analyzer's answers are dozens of narrow facts per image; a dentist reads
one report. Cell 13 sends the findings of each image to a text LLM (the
`reporter` role of the experiment, a spec dict like `analyzer` and `adapter`; it
never sees the image) and saves a classified report in the dentist's language
(`report_language`). `report_writer.py` does it in three fixed steps:

1. **Structured input.** `structured_findings` condenses a result JSON into one
   dense object: all 14 findings in seven sections (restorations and
   prostheses, endodontic, caries, periodontal, periapical, teeth and eruption,
   appliances and hardware), each with an explicit `status` (`present`,
   `absent`, `unparseable`), the answer in every region, the count and the
   per-region counts (numbers, or the words `not_asked`, `incomplete`,
   `unparseable`, `not_countable`), the regions the finding was located in,
   and a `detection` note saying whether the whole-image and the regional
   answers agree. Region names are spelled out anatomically on the patient's
   side. A legend, the analyzer's method and its limitations go with it, so
   nothing is implicit and nothing is null.
2. **One call, fixed prompt.** `SYSTEM_PROMPT` and `USER_PROMPT` ask for a
   radiology-style report as JSON: a title and localized headings, one entry
   per finding with its status copied and a statement in the dentist's
   language, an impression with pathology before treatment history, the
   unparseable findings under "not assessable", and limitations. The writer
   may reword and organise; it may not add, drop, soften or upgrade a finding,
   estimate a count, name a tooth, or give a diagnosis, severity or advice. A
   regional-only detection must be called a weaker signal.
3. **Verification and rendering.** `verify_report` checks the reply against
   the input: every finding exactly once, in its section, with its status
   unchanged, no unknown finding, impression and limitations present, "not
   assessable" naming the unparseable findings and nothing else. A failing
   reply goes back once with the list of problems (`REPAIR_PROMPT`); a reply
   that still fails is saved with its problems and the deterministic
   `dentist_report` takes its place in the Markdown, so the failure is visible
   and the dentist still gets a summary. Verified JSON is rendered to Markdown
   deterministically (sections in a fixed order; ● present, ○ absent,
   ? unparseable).

Reports resume like the other loops: one `.json` (structured input, prompt,
every attempt with its problems, the verified report, the Markdown) and one
`.md` per image under `<dataset>/reports/<model>-<language>/reports/`, with a
manifest that hashes the writer settings, the prompts and the language.
`summarize_reports` counts how many reports verified at once, after a repair,
or fell back. Reports are for reading and are not scored: Cell 10 stays the
measure of the analyzer. The language is not verified; read one report before
trusting a batch.

## Datasets

* **UMFIH 14-class set** (Zenodo 15487430, CC BY 4.0 with a non-commercial
  note): the exact ontology, YOLO boxes. Score the 100-image internal test split
  and the 180 external-validation images separately.
* **DENTEX** (Hugging Face `ibrahimhamamci/DENTEX`, CC BY-NC-SA): FDI
  quadrant labels for caries, periapical lesion, and impacted tooth. Gives exact
  quadrant ground truth for those three findings. Use the fully labeled train
  split (`training_data/quadrant-enumeration-disease`, 705 images) and the
  validation split (`validation_triple.json`, 50 images). The DentalGPT paper
  does not list the detection sets it trained on, so treat DENTEX as held-out
  with that caveat. The 250-image test split is raw LabelMe files with unmapped
  Turkish labels and is not supported.

Each dataset is scored on the findings it annotates; unannotated findings are
not counted as negatives. With two datasets the notebook also prints a pooled
table over the shared findings.

## Location truth

Ground truth is numeric (boxes); the pipeline localizes a finding as the set of
regions that answer True (or count above zero). Scoring location means deciding
which region windows each true box occupies, and fixed image fractions do that
badly: the midline and the occlusal plane move with patient positioning and the
shape of the arch. `location_truth` in Cell 3 picks how it is done:

* `"llm"` (recommended): `location_adapter.LLMAdapter` draws numbered boxes on
  the radiograph, burns the FDI quadrant names into the corners, and asks a
  strong vision model through any OpenAI-compatible API (GPT-5, Gemini's
  compatibility endpoint, NIM, ...) for the dental-arch units each box
  occupies: FDI quadrant x anterior (incisors and canine) / posterior
  (premolars, molars and behind), plus the FDI tooth positions. Units are the
  finest division the quadrant and arch windows are made of, so the mapping
  onto the pipeline's names is deterministic (`dental_pipeline.unit_region`),
  and the same output serves the six-cell vocabulary of the DentVLM branch.
  One call per image (chunked above `max_boxes_per_call` boxes), strict JSON
  back, with bounded retries and the configured location failure policy. The model is the `adapter` role in Cell 3
  (see "Hosted models"); for reasoning models set `token_param` to
  `max_completion_tokens` and leave `temperature` at `None`.
* `"fdm"` (experimental): DentalGPT itself. It was trained with reinforcement
  learning on multiple-choice questions, so the task is split into two short
  questions per box in the Figure 7 shape, on a copy of the image with only
  that box drawn in red: which jaw (upper / lower / both) and which side of the
  image (left / right / both). Sides are asked as image sides so the model never
  resolves the patient-side convention; the quadrant follows in Python. The
  probe's `<think>/<answer>` mode is reused. Drawn boxes are outside the
  model's training images; unparseable answers follow `location_failure_policy`.
* `"geometry"`: the fixed crop windows (a box counts in every window holding at
  least a quarter of its area; the windows overlap on the midline and the
  occlusal plane). No model calls.

The adapter runs once per dataset and adapter (Cell 9), independently of the
model run, and resumes: one JSON per image under
`<dataset>/location_truth/<adapter>/boxes` with the raw reply, the units, the
quadrants, the windows' answer and the source that placed the box; the drawn
images are kept under `.../drawn` for audit. DENTEX boxes carry FDI quadrant
labels, which are exact, so on a DENTEX dataset Cell 9 also prints the
adapter's and the windows' agreement with that truth
(`dental_eval.truth_agreement`): the check that the adapter is worth its calls.
`evaluation.json` records under `summary.location_truth` how many true boxes
each source placed. Arch-level scoring derives from the same quadrants. For
per-region counts each true box is counted once, in the first window (in UR,
UL, LL, LR order) that holds it.

## Evaluation outputs

`<output_root>/<experiment>/<dataset>/evaluation/` holds `presence.csv` (TP, FP, TN, FN,
unparseable, sensitivity, specificity, PPV, F1 per finding, with a
`paper_covered` flag), `whole_image.csv` (the same table for the whole-image
answers alone when `presence_level="region"`: read the two side by side to see
what the regional pass recovered and what it cost in specificity),
`counts.csv` (exact, within-1, MAE on true positives, and a strict MAE that
scores misses as zero), `region_counts.csv` (the same per finding and region
when `count_level="region"`; a region that answered B to presence counts as 0
in the strict MAE), `region_presence.csv` (presence per finding and region:
every region answer of every image, whatever the whole image said, against the
regions the true boxes occupy, as TP, FP, TN, FN, unparseable, sensitivity,
specificity, PPV and F1; one cell per region, so several boxes in a region are
one presence and an unparseable region answer is one excluded cell; an image
whose true boxes could not be placed is left out and counted under
`location_truth_excluded`), `regions.csv` (per-region TP, FP, TN, FN, exact-set
match, Jaccard, unlocalized rate, `from_counts` for regions derived from
counts, and `pred_all_regions_rate` next to `truth_all_regions_rate`),
`per_image.csv`, and `evaluation.json` with a summary: the protocol, micro and
macro F1, complete-case rate, mean false alarms per image, the whole-image
micro numbers under `whole_image`, the presence-per-region micro numbers under
`region_presence` (the leaderboard's `region_f1`), and, for the quadrant
scheme, `side_agreement`. Both count tables are absent from a `counting=False`
run. `regions.csv` and `region_presence.csv` answer different questions: the
first scores where a detected finding was placed (true positives only), the
second whether each region was called correctly at all, absent findings
included. Both need the location truth, so they follow `evaluate_location`.
`case_condition_breakdown.csv` joins each diagnostic situation with each
applicable finding, including support, TP/TN/FP/FN, unparseable answers,
presence metrics, count metrics and location metrics. It is the detailed source
for the cross-experiment situation/finding view described below.

Two diagnostics decide whether word-based regions are being read:

* `side_agreement`: how often a quadrant the model answered for holds a true
  box on that image side (only findings whose true boxes all lie on one side
  count). Far above 50% confirms the patient-side reading; far below means
  `QUADRANT_WORDS_ARE_PATIENT_SIDE` should be set to `False` (the words move
  to the mirrored windows) and the run repeated. DENTEX is the clean test: its
  FDI quadrant labels are exact.
* `pred_all_regions_rate` far above `truth_all_regions_rate` means the model
  answered A in every region whenever the whole image was positive, i.e. it
  ignored the region clause; switch to `region_prompt="crop"` for that finding
  set.

### Small ablation-style reports (Cells 10 and 11)

Cell 10 saves, and Cell 11 displays, compact diagnostics and saves their full rows, including
supporting image IDs, in `evaluation.json` and matching CSV files:

* `stage_changes`: whole-image to regional outcomes, overall and per finding.
  FN -> TP means a recovered miss; TN -> FP means a new false alarm. Reverse,
  unchanged, and unresolved transitions are retained. Missing legacy whole-image
  fields are skipped, not treated as negative predictions.
* `parse_recovery`: first-pass, recovered, and unresolved **fields** of recorded
  questions, by stage. Presence and count are separate, so an unusable count
  does not hide a valid presence decision. Recovered answers are checked against
  truth; parsing successfully does not imply correctness. Regional correctness
  is unavailable when location scoring is off or a true box was excluded.
* `call_usage`: recorded analyzer completions, tokens and latency, split into
  first attempts and parse retries. Combined fields do not duplicate call costs.
  Missing metadata is unavailable, with recorded-call denominators beside usage;
  transport retry attempts are not stored as individual completions.
* `case_breakdown`: one/multiple boxes for a finding, absent findings with/without
  other annotated findings, 0/1-2/3+ finding types, and (when enabled) one/multiple
  true regions, boundary-crossing boxes, and location-truth sources. Mixed and
  excluded truth sources remain visible. These are descriptive groups, not
  causal effects or severity grades. Counts/location use the evaluator's existing
  true-positive subsets; empty denominators are unavailable.
* `case_condition_breakdown`: the same situations split by finding. This keeps
  the aggregate situation table compact while retaining the full drill-down in
  CSV/JSON and in Cell 11 for the selected experiment.

Cell 10 also writes `<output_root>/overview/` with four joined tables:

* `experiment_overview.csv`: one compact row per experiment and dataset.
* `finding_comparison.csv`: every experiment x finding row.
* `situation_comparison.csv`: every experiment x diagnostic situation row.
* `situation_finding_comparison.csv`: every experiment x situation x finding row.

The notebook presents detection, counting, and location as separate compact
blocks so disabled or inapplicable metrics do not create a mostly empty table.
The full exports remain available for filtering and audit. With
`evaluate_location=False`, location rows and location-based situations are
intentionally unavailable; regional inference, whole-image-to-region changes,
finding scores and total counts remain available.

Cell 10 compares the experiments of Cell 3 automatically: every experiment with
a complete set of results for a dataset is scored paired against the first one
and written to `<output_root>/comparison/<dataset>/`. `run_comparison.csv`
includes each experiment's settings, coverage, accuracy and count metrics and
recorded usage; `run_changes.csv` retains paired transitions and image IDs.
Every run must cover the selected ground-truth images with matching image hashes
and consistent saved protocol/mode; an experiment still missing images is left
out of the paired table (its own row stays in the leaderboard). Extra unselected
images are ignored. All runs are rescored against the same supplied ground
truth. Paired F1 uses only checks resolved by both runs; newly
resolved/unresolved counts and full coverage are separate. Location is scored
per experiment in the leaderboard, not in the paired table, because experiments
may use different location truth, and different region schemes have different
localization difficulty.

Use the updated project files and rerun Cell 10 with ground truth and any needed
location adaptation already loaded; these reports make no model calls. Saved
attempts describe the executed questions, not a simulated retries-OFF run.
Older runs without attempt metadata cannot produce recovery rows. Generation
changes still need their own experiment (their own name and directory).

## Caveats

* Ten of the 14 findings are outside the paper's evaluated panoramic labels;
  read the `paper_covered` column before comparing to the paper's 84%.
* Jaw wording is benchmark-verbatim; quadrant wording rests on the model's own
  narration and on FDI numbering, and no benchmark question uses the word.
  Read the side agreement and the region tables on DENTEX before trusting
  quadrant rows.
* Location truth from the LLM adapter is itself a model output: read the DENTEX
  agreement numbers and spot-check the drawn images before trusting the region
  rows.
* Ground-truth boxes are per instance while the model counts teeth, so counts
  for crowns/bridges and multi-box fillings carry definitional error; region
  counts inherit it.
* Every region is asked about every finding, so one region false alarm makes
  the finding present: compare `whole_image.csv` with `presence.csv` before
  reading the regional numbers as an improvement.
* With `region_prompt="crop"` and `count_level="region"`, the windows overlap
  by 10% of the width and 20% of the height, so teeth on the seams can be
  counted twice; use words for region counts.
* Apical surgery, root resorption, and furcation have very few positives in
  UMFIH; their rows are not statistically meaningful.
* The combined form's definitions and counting rules (`DEFINITIONS`) are
  radiographic conventions written for this ontology, not the UMFIH annotation
  guide (bridge pontics are not counted, for one); check them against it before
  comparing count rows of a combined run with those of a separate run.
## Optional location scoring

In notebook Cell 3, set `evaluate_location = True` (default) to score locations,
or `False` to skip location scoring and location-truth adapter calls. Finding
scores and total-count scores remain enabled; inference, counting questions,
and saved predictions are unchanged. Regional-count metrics, presence per region and the side check also follow this switch.
Re-run Cell 3, Cell 9, and Cell 10 to evaluate existing results with this setting;
no inference rerun is required.
The report records `summary.evaluate_location`. Re-exporting a report with location
scoring disabled removes its previous location CSVs so stale metrics are not shown.
