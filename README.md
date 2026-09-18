# DentVLM panoramic findings pipeline

A small wrapper that gets findings, their regions, and an occupied-region count
(in how many distinct regions a finding is reported, never how many teeth) out
of DentVLM (a 7B dental vision-language model) on panoramic radiographs, while
sending it only questions it was trained and evaluated on. Everything else
(task decomposition, aggregation, counting, scoring) happens in Python.

## Experiments

The notebook runs a list of configurations, not one. Cell 3 holds one dictionary
per experiment: a name, plus the knobs that experiment changes. Everything it
does not mention comes from the `SHARED` dictionary above it, and then from
`experiments.DEFAULTS`:

```python
EXPERIMENTS = xp.build([
    {"name": "base"},
    {"name": "three-phrasings", "phrasings": 3},
    {"name": "six-regions", "location": "regions"},
    {"name": "gemini", "analyzer": {"provider": "gemini", "model": "gemini-3-pro"}},
    {"name": "dentvlm-local", "backend": "local"},
], shared=SHARED)
```

Anything in `experiments.DEFAULTS` may vary per experiment: the backend (local
DentVLM through llama.cpp or a hosted model), the analyzer, adapter and reporter
models, the six protocol knobs, the retry budgets, the location truth, the
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

* `<output_root>/leaderboard.csv`: one row per experiment and dataset with the
  protocol knobs, raw TP/TN/FP/FN, scored/expected/annotated denominators,
  coverage, detection rates, six-cell location diagnostics,
  side agreement, and calls per image. Each row is scored against that
  experiment's own location truth.
* `<output_root>/overview/`: joined experiment, finding, situation,
  situation-by-finding, stage, phrasing, union/majority replay, parse-recovery,
  and call-usage tables. Cell 11 shows compact decision columns; these CSVs keep
  the full diagnostic rows and supporting image IDs.
* `<output_root>/comparison/<dataset>/`: the same experiments compared **paired**
  on the same images against the first one - `paired_f1_delta`, checks corrected
  and worsened, newly unresolved, recorded calls and tokens.

Cells 12 to 14 look at one experiment at a time: `INSPECT_EXPERIMENT` selects it
for the per-finding tables and saved diagnostics, for one image's raw answers,
and for the dentist report (one report call per image, so it defaults to the
inspected experiment).

Running several experiments increases the number of logical questions. `"limit"`
in `DATASETS` controls the selected images, and every cell resumes, so a sweep
can be extended or an experiment added without recomputing saved results.

Local DentVLM experiments share an exact response cache under
`<output_root>/_response_cache`. This is especially useful here: the base question
is also phrasing 1 of the three-phrasing run and the whole-image stage of the
region run. A reply is reused only
when the converted model files, llama.cpp binary and server settings, complete
request, prompt, generation settings, and image bytes match. Changed
phrasings, question wording, token budgets, models, or runtime settings miss the cache.
Set `reuse_local_responses=False` only when independently repeating identical
deterministic calls is itself part of the experiment.

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
* **Location** is never asked by default. The nine descriptors are read from the
  rationale by exact match, exactly as the authors compute their IoU, and mapped
  onto six dental-arch cells (upper/lower x left/anterior/right). `location="regions"`
  asks instead, region by region, using those same descriptor strings inside the
  task's own question and never cropping the image.
* **Multiplicity** is the number of distinct cells a finding is reported in
  (0 to 6): the size of the deduplicated region set, never the number of
  boxes, teeth, mentions or Yes answers, reported as "in N region(s)". DentVLM
  is never asked to count: it has no count task, so the model is asked
  presence only and the count is derived from the regions with an explicit
  status (`dental_pipeline.count_block`): `resolved` (an accepted No is 0; a
  Yes with every region read is the number named), `partial` (region
  questions: some regions Yes, at least one unreadable, so the confirmed
  regions are a lower bound), `unlocated` (reported, but no region named: the
  total is unavailable, not zero) and `unresolved`. The evaluation scores each
  finding as present or absent per image and per cell (`region_presence.csv`),
  so a class that occurs several times in a cell is scored once, and compares
  the count with the number of distinct cells the true boxes occupy
  (`occupied_regions.csv`, the `counting` knob; see "Optional location
  scoring and counting").
* **Findings without a DentVLM task** (furcation involvement, apical surgery,
  root resorption, orthodontic appliances, surgical plates) are not asked and
  are reported as "not assessed by this model". `ask_untrained`
  asks them anyway and scores them under `trained_task=False`; the paper's
  zero-shot accuracy on untrained diseases is 52-64%.
* **Prosthetic restoration** (crowns or bridges in the benchmark) is the OR of
  the prosthetic crown and prosthetic bridge tasks, regions merged.
* **Bounded parse recovery.** The notebook sets `parse_retries = 1`: one extra
  attempt per unparseable answer, on the same image/model with a reminder to put
  Yes/No on line 1 and retain the rationale/location.
  Every failed attempt prints the full prompt and response;
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
* **Nothing fails silently, and no failure takes the sweep with it** (`run_monitor.py`). Every stage
  runs its items behind a guard: a failed image, dataset, experiment, smoke test, adapted image,
  scored run or report prints its error and full traceback, is recorded in the session ledger with
  its scope, and the loop continues with the next item; the run directory keeps a `failures.json`,
  and each cell ends with a `[HEALTH]` line grouping what failed by reason. Three failures in a row
  stop that dataset instead, because that is a dead server or a rejected key rather than a bad
  image. Only identity errors (a saved artifact that belongs to another run, a changed hashed
  control) still stop at once, and `Ctrl-C` is never swallowed. The failed items are simply absent
  from the results, so a rerun resumes them and the evaluation reports them as `missing_results`.
* **Dense monitoring while it runs.** One line per image carries the Yes/No/unclear counts, which
  findings were found, which were never asked (no trained task), any aggregation warning, the call
  and cache counts and an ETA; one `[DONE]` line per dataset carries the totals, the token counts
  and every retry the stage paid for. Single model calls are counted rather than listed, and printed
  only when worth reading (slow, truncated, empty) — `CALL_LOG` in Cell 3 switches that to `"each"`
  for debugging one image, or `"off"`. Failures are the exception to density and always print
  completely. Before any model call, Cell 7 prints what each benchmark holds and flags what is
  unusable in it: images with no label file, label files with no image, annotated images missing
  from disk.

Left and right follow the model's own convention (Supplementary Table S6): its
"left posterior region" is FDI quadrants 1 and 4, the patient's right, which is
the left side of a panoramic as displayed. `dental_pipeline.LEFT_IS_IMAGE_LEFT`
records that reading, the dentist summary translates cells to the patient's
side, and the notebook checks the convention against DENTEX boxes.

## Files

| File | Role |
| --- | --- |
| `dental_pipeline.py` | task table and verbatim questions, answer and region extraction, protocol knobs, model runner, resumable run loop, dentist summary |
| `dental_eval.py` | ground-truth loaders (UMFIH YOLO, DENTEX with FDI tooth numbers), location truth (adapted, FDI, or fixed windows), metrics incl. presence per cell and occupied-region counts (one primary cell per true box), side-convention check, CSV/JSON export |
| `dental_analysis.py` | offline phrasing/vote and region comparisons, recovery, case breakdowns, paired saved-run comparisons |
| `response_cache.py` | immutable, content-addressed reuse of exact local DentVLM responses across compatible experiments |
| `location_adapter.py` | translates ground-truth boxes into the six cells: vision-LLM adapter (numbered boxes drawn on the image), per-image area adapter (the cell areas of that radiograph, then geometry), experimental DentVLM spotlight adapter, resumable per-dataset run |
| `llama_runtime.py` | llama.cpp build, one-time GGUF conversion of the Hugging Face checkpoint, GGUF download, server process (with image-token flags) |
| `report_writer.py` | dentist report: dense structured findings per image (tasks, cells, multiplicity, extra tasks, not-assessed findings), report-writer prompts, verification of the reply against the input, one repair turn, Markdown rendering, resumable run |
| `experiments.py` | the experiment table: DEFAULTS, merging and validation of each configuration, per-experiment paths, the runner/adapter/report-writer of one experiment |
| `llm_api.py` | hosted-model access shared by the runner, the adapter, the report writer and the parser: provider registry, key lookup (environment variable or Kaggle secret), client construction, visible API and parse retries |
| `llm_parser.py` | reading what the models wrote: the strict readers, the optional parser LLM behind them, one mode per stage plus a global override, the prompts, and the record of every parse |
| `run_monitor.py` | the console and failure side of a run: dense per-item progress with ETA, call counters with a print policy, the failure ledger and the guard that keeps a loop alive |
| `main_notebook.ipynb` | Kaggle runner; the experiments to run and compare are Cell 3, the ranking is Cell 11 |
| `test_dental_pipeline.py`, `test_location_adapter.py`, `test_report_writer.py`, `test_location_scoring.py`, `test_region_counting.py`, `test_llm_api.py`, `test_llm_parser.py`, `test_experiments.py`, `test_response_cache.py`, `test_run_monitor.py` | offline tests with fake models (`python -m unittest -q`) |

## Small evaluation comparisons (Cells 11 and 12)

Cell 11 saves, and Cell 12 displays, compact diagnostics with supporting image IDs
in `<output_root>/<experiment>/<dataset>/evaluation/evaluation.json` and matching CSV files.
The existing finding and location scoring rules are preserved.

| Table | DentVLM-specific comparison |
| --- | --- |
| `stage_changes` | First saved phrasing vs whole-image vote, and whole-image vs region-question outcomes. Includes corrected errors, new errors, unchanged, unresolved and not-assessed outcomes; overall and per finding. |
| `phrasing_votes` | Agreement, disagreement, ties, and unresolved phrasings per task. Available when multiple phrasing answers were saved. |
| `region_vote_comparison` | Union vs majority using identical saved rationale answers and the existing vote/OR rules, re-aggregated with `dental_pipeline._finding` so the regions are merged first and counted after (never a sum over wordings); the count metrics come with it. Only for rationale mode with multiple saved phrasings; region-question locations do not use this vote. |
| `parse_recovery` | First-pass, recovered, unresolved questions by task/stage, plus correctness where ground truth supports it. Each phrasing is a separate question. |
| `call_usage` | Recorded analyzer completions, tokens and latency, split into first attempts and parse retries. Logical calls, actual inference calls and cache hits are separate, with inference-only token/latency totals. Missing usage is unavailable; transport attempts are not separate saved completions. |
| `case_breakdown` | Trained/untrained task support, instance counts, other findings, named/true cells, boundary-crossing boxes, and location-truth sources. Unasked findings remain not assessed. |
| `case_condition_breakdown` | The same situations split by finding, including assessment status, raw confusion counts, coverage, and six-cell location metrics. |

Cell 11 also writes the joined versions to `<output_root>/overview/` as
`experiment_overview`, `finding_comparison`, `situation_comparison`,
`situation_finding_comparison`, `stage_comparison`, `phrasing_comparison`,
`vote_replay_comparison`, `parse_recovery_comparison`, and
`call_usage_comparison`. Findings without an active DentVLM task appear as
`not_assessed`; their TP/TN/FP/FN cells are blank rather than misleading zeros.

First-phrasing and region-vote replay use saved answers **after any parse repairs**;
they do not simulate retries OFF or change predictions. Crown and bridge are
combined with the existing OR rule before finding scoring. Their individual retry
answers cannot be graded from the merged restoration label, so correctness is
unavailable for those tasks; extra tasks without benchmark labels are likewise
unscored. A recovered parse can still be wrong. An occupied-region count is
never treated as a tooth count.

Cell 11 compares the experiments of Cell 3 automatically: every experiment with a
complete set of results for a dataset is scored paired against the first one and
written to `<output_root>/comparison/<dataset>/`. `run_comparison` reports each
experiment's knobs (`phrasings`, `region_vote`, `location`, `ask_untrained`,
`extra_tasks`, `parse_retries`), coverage, metrics and recorded
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
not-assessed checks have their own column. Location metrics retain each
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
* `location="regions"`: the primary question for every task asked once per
  dental-arch region, with the region named inside the question and the whole
  uncropped image sent every time: "Based on the imaging analysis, does the
  patient have caries in the left posterior region of the lower dentition?".
  The added words are one of the model's own nine location descriptors
  (`dental_pipeline.CELL_DESCRIPTORS`, inverted from the scorer's `DESCRIPTORS`
  so the two can never drift apart), so the region is asked in the vocabulary
  and the left/right convention the model was trained to write, and the
  question keeps its verbatim shape. Every region is asked whatever the whole
  image answered (kept as `whole_image`). A task is present when any region
  answers Yes, so a finding missed with the model's attention spread over the
  whole image can be recovered in a region, and the regions answering Yes are
  its regions. 13 + 13 x 6 = 91 calls per image. The model was not fine-tuned
  on region-restricted questions, so this is a comparison, not the default;
  cropping the panoramic instead would change the image distribution as well,
  which is why the image is never cropped.
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
`adapter`, `reporter` and `parser` role dictionaries of an experiment then select any provider and exact model, e.g.
`{"provider": "openrouter", "model": "qwen/qwen3-vl-235b-a22b-thinking"}`.
Model-specific options such as `token_param`, `temperature`, and OpenRouter
routing under `request_options` stay with the role. `VisionRunner.from_api`,
`LLMAdapter.from_api`, `AreaAdapter.from_api`, `ReportWriter.from_api` and
`ParserModel.from_api` build the
clients; run manifests record the public role configuration, never the provider key. Transport
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
   `present` / `absent` / `unparseable` from the region questions), the regions the
   finding was located in on the patient's side, the multiplicity, a
   `trained` flag for zero-shot questions, and a `detection` note for the
   region comparison. A legend, the
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
   estimate a number of teeth, name a tooth, report a region the model did not
   name as free of the finding, or give a diagnosis, severity or advice.
   Untrained questions and region-only detections must be called what they
   are.
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

**Agreement between wordings (`vote_agreement`, off).** With `phrasings > 1`
every task is answered several times and voted, but the report was only ever
given the decision, so `region_vote="union"` reached it with a region three
wordings named and a region one wording named looking equally supported. With
the reporter's `vote_agreement` knob on, each finding and each of its tasks
also carries an `agreement` block read from the saved answers: how many of the
readable answers reported the recorded status (`2/3`), how many wordings were
asked, how many answers were unreadable, whether the readable answers tied, and
per region how many of the answers that reported the finding named that region,
counted over those answers and never added together (`lower-left 3/3` stays
apart from `upper-right 1/3`). Each count comes with the fixed phrase to use for
it - consistently identified, moderately supported, weakly supported / not
consistent, not consistent (tie), or not measured - and the prompt gains two
passages: what the counts are, and the rule to quote them as they stand, keep
the regions apart, name unreadable answers and ties, and invent no percentage,
probability or confidence. `verify_report` rejects a reply quoting a count the
data does not hold, so `3/3` cannot be claimed where two answers were readable,
and the Markdown prints the counts under each finding from the data itself.
These counts measure how stable the model is under rewording; they are not a
probability, not medical certainty and not diagnostic confidence, and both the
prompt and the report's limitations say so. Nothing else moves: the answers,
the union/majority decision and Cell 11 are untouched, and with the knob off the
structured input, the prompt, the report and the Markdown are exactly what they
were.

Reports resume like the other loops: one `.json` (structured input, prompt,
every attempt with its problems, the verified report, the Markdown) and one
`.md` per image under `<dataset>/reports/<model>-<language>/reports/`, with a
manifest that hashes the writer settings, the prompts and the language.
`summarize_reports` counts how many reports verified at once, after a repair,
or fell back. Reports are for reading and are not scored: Cell 11 stays the
measure of the analyzer. The language is not verified; read one report before
trusting a batch.

## Reading what the models wrote

Every number this project produces comes from reading text a model wrote. The
readers are strict: `\byes\b` / `\bno\b` on line 1, the nine location
descriptors matched verbatim, `json.loads` on the adapter's boxes and on the
report, `(\d+)/(\d+)` for a quoted vote. Strict readers fail in two directions.
They reject a reply a dentist would understand at once ("Caries is evident in
the lower left quadrant" names no descriptor, so the scorer sees no region at
all), and they accept text whose meaning is not the matched string (a rationale
that names a region only to rule it out).

`llm_parser.py` puts a second reader behind them: a text LLM (the `parser` role)
whose only job is to say what an existing reply means. It never sees a
radiograph, never sees ground truth, and never decides anything clinical.

**Ten stages, each with its own mode.**

| stage | default in manual mode | why |
| --- | --- | --- |
| `whole_image_decision` | `code_then_llm` | line 1 is reliable when it reads at all |
| `region_decision` | `code_then_llm` | the same reader on the same replies |
| `rationale_location` | `llm` | a paraphrase looks to the strict reader exactly like "no region named" |
| `saved_answer_reconstruction` | `code_then_llm` | only the replies the run could not read are re-read |
| `spotlight_decision` | `code_then_llm` | the same line-1 reader |
| `spotlight_location` | `llm` | a missed descriptor silently moves a box to the fixed windows |
| `location_json` | `code_then_llm` | valid JSON that passes the schema check is exactly right |
| `report_json` | `code_then_llm` | a reply that loads is the report; the model repairs the rest |
| `report_fidelity` | `llm` | a report can pass every structural check and still say more than the data does |
| `vote_fraction` | `code_then_llm` | "2/3" reads exactly; "2 out of 3" is an explicit failure the model can read |

`"code_then_llm"` is the default wherever the strict reader is right whenever it
succeeds, so a model is only paid for after an explicit failure: an ambiguous or
missing decision, invalid or incomplete JSON, a schema mismatch, a missing entry,
truncation. `"llm"` is the default wherever the strict reader can succeed while
losing the meaning, because consulting it first would hide the very thing the
second reader is there to catch.

The location prompt is what makes the occupied-region count trustworthy when
the parser reads a rationale: it lists every region the reply places the
finding in, a region named twice once, a "both the upper and lower" phrase as
two, a region mentioned only to deny the finding or to describe another finding
not at all, and it refuses to expand a broad site ("the posterior teeth", "the
lower jaw") into several regions - such a site is unresolved, never a guess.
Its tooth-number table is generated from `dental_pipeline.unit_cell`, the same
mapping the scorer and the adapters use, so FDI quadrant 1 reads as DentVLM's
"left" exactly as `fdi_cell` does (prompt version 2; version 1 stated the
quadrants the other way round). The side convention is part of the parser
settings, so a flipped `LEFT_IS_IMAGE_LEFT` is a different reading.

**One switch over all ten.** `parser_mode` in Cell 3:

| value | effect |
| --- | --- |
| `"code"` | strict readers only. The shipped default: no parser key, no extra call, and byte-identical decisions to every run made before this existed |
| `"llm"` | the parser model reads every stage |
| `"code_then_llm"` | the strict reader everywhere, the model only after a real failure |
| `None` | manual: every stage follows its own mode in `parser_modes` |

The global setting overrides every stage whenever it is not `None`. The resolved
modes are printed with the experiment table, saved in `experiment.json`, and
hashed into the run manifest, so resuming a directory with a different parser
model, prompt or mode is refused rather than filling one result set with two
readings of the same replies. Reconstructing a saved answer is guarded the same
way: an accepted value is never read again, and a replay whose parser
configuration does not match the run's reads with code alone and says so.

**Nothing unreadable becomes a finding.** When both readers fail, the result is
the unresolved state the strict reader would have left: an unreadable decision
stays unresolved (never "No"), an unreadable location stays unresolved (never
the empty set, which would claim the model named no region), and an unresolved
fidelity check adds no problem and hides none. The evaluation already counts an
unresolved location as `region_unparseable` and excludes it; the report says
"the location could not be read" rather than listing six regions as not named.

**What is not routed through a model.** OpenAI response envelopes, YOLO and
DENTEX annotations, run manifests, response-cache artifacts, configuration
dictionaries and saved-artifact integrity checks. Those are machine formats with
one correct reading, and a model could only make them less reliable.

**What is recorded.** Per parse: the original text, the parser's input and reply,
the parsed value, the selected and the resolved mode, whether the strict reader
succeeded or caused the fallback, the model, latency, tokens, retries, cache
status and the final failure reason. Parser calls are counted under their own
role (`CallLog("parser")`), never mixed into the analyzer, adapter or reporter
totals, and Cells 11 and 12 show one row per stage: how often each reader
answered, how often it fell back, and what stayed unresolved. Identical parser
requests are reused from `<output_root>/_parser_cache`.

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
* `"areas"`: the same kind of hosted model, asked once per image about the
  radiograph itself with nothing drawn on it: where do this patient's six cells
  lie? It answers one normalized area `[x1, y1, x2, y2]` per cell, accepted only
  as a complete valid set (every cell once, four numbers each, inside `[0, 1]`,
  positive width and height), and ordinary geometry then places the boxes: each
  one goes to the area covering the greatest fraction of it, or, when no area
  touches it, to the nearest area. Equal scores fall to cell order, so the same
  box always lands in the same cell, and every box records the rule that placed
  it with the coverages or distances behind the decision. The model never sees a
  finding box, so it cannot classify a finding; it only moves the boundaries that
  positioning, arch shape, centring and missing teeth move. The cells are named
  to it in DentVLM's own descriptors under the current `LEFT_IS_IMAGE_LEFT`
  reading, and that flag is part of the adapter configuration, so flipping it
  asks again rather than re-labelling saved areas. One call per image whatever
  the number of boxes, read by code (no parser stage); the areas, the reply and a
  marked copy of the image (cells labelled, true boxes in white) are saved. Same
  `adapter` role and failure policy as `"llm"`.
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
`<dataset>/location_truth/<adapter>/boxes` with the raw reply, the units (or the
areas and the assignment behind each box), the cells, the windows' answer and the
source that placed the box; the drawn, spotlighted or marked images are kept
under `.../drawn` for audit. DENTEX boxes carry FDI
tooth numbers, which give exact cells, so on a DENTEX dataset Cell 10 also
prints the adapter's and the windows' agreement with that exact truth
(`dental_eval.truth_agreement`): the check that the adapter is worth its calls.
`evaluation.json` records under `summary.location_truth` how many true boxes
each source placed.

## Evaluation outputs

`<output_root>/<experiment>/<dataset>/evaluation/` holds `presence.csv` (TP, FP, TN, FN,
unparseable, sensitivity, specificity, PPV, F1 per finding, with a
`trained_task` flag), `whole_image.csv` (the same table for the whole-image
answers alone with `location="regions"`: read the two side by side to see what
the region questions recovered and what they cost in specificity), `region_presence.csv`
(presence per finding and cell: every cell of every asked image, whatever the
whole image answered, against the cells the true boxes occupy, as TP, FP, TN,
FN, unparseable, sensitivity, specificity, PPV and F1; one cell per image, so
several boxes in a cell are one presence; with `location="rationale"` a named
cell is the prediction and an unnamed cell counts as not predicted, with
`"regions"` each region question's own answer counts and an unparseable one is an excluded
cell; an image whose true boxes could not be placed is left out and counted
under `location_truth_excluded`), `regions.csv` (per-cell TP, FP, TN, FN,
exact-set match, Jaccard, unlocalized rate, over the localized true positives
only), `per_image.csv`,
and `evaluation.json` with a summary: the saved protocol, micro and macro F1,
complete-case rate, mean false alarms per image, the presence-per-cell micro
numbers under `region_presence`, left/right side agreement when location is
enabled, and the list of findings not assessed. `regions.csv` and
`region_presence.csv` answer different questions:
the first scores where a detected finding was placed, the second whether each
cell was called correctly at all, absent findings included. Both need the
location truth, so they follow `evaluate_location`.

`occupied_regions.csv` (the `counting` knob) compares, per finding and image,
the number of distinct cells the model reported the finding in with the number
of distinct cells its true boxes occupy. Every true box is placed in exactly one
cell (`dental_eval.box_primary_region`: the adapted, FDI or fixed-window cell;
when several cells hold a box, the one holding the largest share of it, ties to
cell order), and the cells of one class are deduplicated, so three caries boxes
in one cell and two in another are a truth count of 2, and the same prediction
`{A, B}` is exact. A case is scored only when both sides are resolved: an
accepted No is a predicted 0, so true negatives (exact 0), false alarms
(overcount) and missed findings (undercount) all count; a Yes without a named
region (`unlocated`), an unreadable decision or location (`unresolved`), a
partial region set from the region questions (`partial`) and a finding whose
true boxes could not be placed (`truth_incomplete`) are excluded under that
reason and never become a zero. The table gives `exact_rate`, `mae`,
`overcount_rate` and `undercount_rate` over the scored cases, the expected /
scored / excluded denominators with the excluded cases by reason, and
`exact_rate_of_expected`, the end-to-end success over every expected check, so
an unresolved prediction is charged against the run rather than hidden. In the
region comparison the table has two rows per finding: `presence` (the region
questions, the run's answer) and `whole_image` (the rationale stage alone).
`straddling_truth_boxes` says how many scored true boxes several cells held:
the location tables keep every cell such a box touches, the count target keeps
one, so those are the cases where the two families can disagree. A count is
compared with a count only; a right count with wrong cells (truth `{A, B}`,
prediction `{A, C}`) is exact here and shows its mistake in `regions.csv`
(TP 1, FP 1, FN 1, TN 3).

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
## Optional location scoring and counting

Three settings are independent: how findings are asked (`location`:
`"rationale"`, `"regions"` or `"none"`), whether occupied-region counts are
scored and reported (`counting`), and whether location correctness is scored
(`evaluate_location`). Neither switch changes a question, a saved answer or the
run manifest: both are evaluation and report settings, so an existing run is
re-scored by re-running Cells 3, 10 and 11 with no inference.

| `counting` | `evaluate_location` | what is scored |
| --- | --- | --- |
| off | off | image-level presence only; no true box is placed and no adapter runs |
| on | off | presence and `occupied_regions.csv`; region identity is ignored in the count comparison, but the regions are still extracted and the truth still adapted (with a hosted `location_truth`, Cell 10 says the adapted truth serves "counts only") |
| off | on | presence, `region_presence.csv` and `regions.csv`; no count table, no multiplicity in the report |
| on | on | all three families |

With `counting` off the count leaves everything: the evaluation tables and
summary, the leaderboard columns, the deterministic summary ("regions: ..."
instead of "in N region(s): ...") and the dentist report, whose structured
input, legend and prompt then carry no multiplicity at all. With it on, the
report gets each present finding's multiplicity as the number of regions it was
reported in ("reported in two regions", never "two lesions" or "two teeth"),
or the words that say why there is none: "at least N" for a partial set, "not
stated" when the model named no region, "unresolved" when its location could
not be read. `counting` with `location="none"` is refused at build time: there
is no region evidence to count. The same truth mapping (`location_truth`)
serves both families, so toggling `evaluate_location` never changes the count
target. Re-exporting an evaluation with a switch off removes that family's CSVs
so stale metrics are not shown; the same rule retires `counts.csv`, the count
question of the other branch, which this branch never writes.

The finding schema of a saved result carries the count (`region_count`,
`count_status`, `unresolved_regions`, and the whole-image stage's regions next
to the authoritative ones). It is versioned in the run manifest
(`findings_version` 2), so a directory written before it is refused on resume
rather than filled with two schemas; rerun it into a new directory (local
replies come back from the response cache). Scoring such an older directory
still works: the count is derived from its presence and regions, and in the
region comparison the regions left unresolved are reconstructed from its saved
calls.
