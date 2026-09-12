# DentalGPT panoramic findings pipeline

A small wrapper that gets findings, counts, and coarse locations out of
DentalGPT (a 7B dental vision-language model) on panoramic radiographs, while
sending it only question shapes it was trained and evaluated on. Everything else
(task decomposition, location, aggregation, scoring) happens in Python.

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
* **No JSON, no paraphrase retries, no forced zeros.** One greedy call per
  question. Unparseable answers are recorded as such and excluded from the
  per-finding TP/FP/TN/FN tables, never converted into a negative there. The
  per-image complete-case rate and recall are strict: a true finding whose
  answer was unparseable counts as not caught.

The only public weights are the GGUF conversion of `DentalGPT-7B-1026`. That
checkpoint may predate the reinforcement-learning stage, and the exact sentence
the authors appended to request `<think>/<answer>` tags is unpublished. A short
probe therefore decides once per run whether the suffix is needed ("plain" or
"tagged"), and the run manifest records the choice.

## Two levels

`dental_pipeline.Protocol` (Cell 3) has four knobs:

| Knob | Values | Meaning |
| --- | --- | --- |
| `PRESENCE_LEVEL` | `overall`, `region` | `overall`: presence from the whole-image question only. `region`: the same question for every finding in every region, region by region, independent of the whole-image answers (kept under `whole_image`). A finding is present when any region answers A and absent only when every region answers B; the region set is the regions that answer A. |
| `COUNT_LEVEL` | `overall`, `region` | `overall`: one whole-image count per positive countable finding. `region`: one count per region, asked right after a region answers A when `PRESENCE_LEVEL="region"`, else in every region for every countable finding; the finding's count is the sum, and a region count above zero also localizes the finding. A region count of 0 is a valid answer. |
| `REGION_SCHEME` | `quadrant`, `arch` | UR, UL, LL, LR (patient-side FDI names) or upper, lower. |
| `REGION_PROMPT` | `words`, `crop` | `words`: "Kindly evaluate if the condition 'X' is present in the upper right quadrant of this image." and "How many teeth in the upper right quadrant have ..." on the whole image. `crop`: the whole-image questions on the region crop. |

The whole-image wording is byte-identical in every configuration; the region
wording only fills a scope slot of the same sentence (`REGION_PHRASES`,
`COUNT_TEMPLATES`). Quadrant words are the patient's sides, so "the upper right
quadrant" is the image-left window; `QUADRANT_WORDS_ARE_PATIENT_SIDE` records
that reading and the DENTEX side check (below) confirms it.

## Calls per image

R = regions in the scheme (4 or 2), Pc = positive countable findings, Rp =
region-finding pairs that answered A for a countable finding.

| `PRESENCE_LEVEL` / `COUNT_LEVEL` | Calls | Example (Pc=4, Rp=6, quadrants) |
| --- | --- | --- |
| overall / overall | 14 + Pc | 18 |
| region / overall | 14 + R x 14 + Pc | 74 |
| overall / region | 14 + R x 9 | 50 |
| region / region | 14 + R x 14 + Rp | 76 |

The regional calls never depend on what the whole image answered, so an
all-negative image needs 14, 70, 50 or 70 calls. With `REGION_PROMPT="words"`
every call reuses the cached image prefix; with crops the loop is region-major,
so each crop's prefix is built once.

## Files

| File | Role |
| --- | --- |
| `dental_pipeline.py` | prompts and region wording, answer extraction, crops, model runner, probe, `Protocol`, resumable run loop, dentist summary |
| `dental_eval.py` | ground-truth loaders (UMFIH YOLO, DENTEX with FDI labels), location truth (adapted, FDI, or fixed windows), metrics incl. per-region counts and the side check, CSV/JSON export |
| `location_adapter.py` | translates ground-truth boxes into the region windows: vision-LLM adapter (numbered boxes drawn on the image), experimental DentalGPT multiple-choice adapter, resumable per-dataset run |
| `llama_runtime.py` | llama.cpp build, GGUF download, server process (with image-token flags) |
| `report_writer.py` | dentist report: dense structured findings per image, report-writer prompts, verification of the reply against the input, one repair turn, Markdown rendering, resumable run |
| `llm_api.py` | hosted-model access shared by the runner, the adapter and the report writer: provider registry, key lookup (environment variable or Kaggle secret), client construction |
| `main_notebook.ipynb` | Kaggle runner; edit Cell 3 only |
| `test_dental_pipeline.py`, `test_location_adapter.py`, `test_report_writer.py`, `test_location_scoring.py`, `test_llm_api.py` | offline tests with fake models (`python -m unittest -q`) |

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
* `BACKEND="api"` in Cell 3 sends the same prompts to a hosted vision model
  instead, for a controlled comparison against DentalGPT (see "Hosted models").

## Hosted models

Cell 3 has one `PROVIDERS` registry containing each provider's base URL and API
key. The keys come from environment variables or Kaggle Secrets (Add-ons >
Secrets), and unused providers may have no key. The small `ANALYZER`,
`ADAPTER` and `REPORTER` role dictionaries then select any provider and exact model, e.g.
`{"provider": "openrouter", "model": "qwen/qwen3-vl-235b-a22b-thinking"}`.
Model-specific options such as `token_param`, `temperature`, and OpenRouter
routing under `request_options` stay with the role. `VisionRunner.from_api`,
`LLMAdapter.from_api` and `ReportWriter.from_api` build the clients; run manifests
record the public role configuration, never the provider key. Transport
errors, rate limits and 5xx replies are retried by the client with backoff; a
bad request or key fails at once. With the API backend `MODE="auto"` resolves
to `"plain"` without a probe: the `<think>/<answer>` suffix is a DentalGPT
training artifact.

## Dentist report

The analyzer's answers are dozens of narrow facts per image; a dentist reads
one report. Cell 15 sends the findings of each image to a text LLM (the
`REPORTER` role in Cell 3, a spec dict like `ANALYZER` and `ADAPTER`; it never
sees the image) and saves a classified report in the dentist's language
(`REPORT_LANGUAGE`). `report_writer.py` does it in three fixed steps:

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
or fell back. Reports are for reading and are not scored: Cell 13 stays the
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
shape of the arch. `LOCATION_TRUTH` in Cell 3 picks how it is done:

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
  back, one retry when the reply is incomplete, and a box the model cannot
  place falls back to the windows. The model is the `ADAPTER` role in Cell 3
  (see "Hosted models"); for reasoning models set `token_param` to
  `max_completion_tokens` and leave `temperature` at `None`.
* `"fdm"` (experimental): DentalGPT itself. It was trained with reinforcement
  learning on multiple-choice questions, so the task is split into two short
  questions per box in the Figure 7 shape, on a copy of the image with only
  that box drawn in red: which jaw (upper / lower / both) and which side of the
  image (left / right / both). Sides are asked as image sides so the model never
  resolves the patient-side convention; the quadrant follows in Python. The
  probe's `<think>/<answer>` mode is reused. Drawn boxes are outside the
  model's training images, and an unparseable answer leaves the box to the
  windows.
* `"geometry"`: the fixed crop windows (a box counts in every window holding at
  least a quarter of its area; the windows overlap on the midline and the
  occlusal plane). No model calls.

The adapter runs once per dataset and adapter (Cell 12), independently of the
model run, and resumes: one JSON per image under
`<dataset>/location_truth/<adapter>/boxes` with the raw reply, the units, the
quadrants, the windows' answer and the source that placed the box; the drawn
images are kept under `.../drawn` for audit. DENTEX boxes carry FDI quadrant
labels, which are exact, so on a DENTEX dataset Cell 12 also prints the
adapter's and the windows' agreement with that truth
(`dental_eval.truth_agreement`): the check that the adapter is worth its calls.
`evaluation.json` records under `summary.location_truth` how many true boxes
each source placed. Arch-level scoring derives from the same quadrants. For
per-region counts each true box is counted once, in the first window (in UR,
UL, LL, LR order) that holds it.

## Evaluation outputs

`<OUTPUT_DIR>/<dataset>/evaluation/` holds `presence.csv` (TP, FP, TN, FN,
unparseable, sensitivity, specificity, PPV, F1 per finding, with a
`paper_covered` flag), `whole_image.csv` (the same table for the whole-image
answers alone when `PRESENCE_LEVEL="region"`: read the two side by side to see
what the regional pass recovered and what it cost in specificity),
`counts.csv` (exact, within-1, MAE on true positives, and a strict MAE that
scores misses as zero), `region_counts.csv` (the same per finding and region
when `COUNT_LEVEL="region"`; a region that answered B to presence counts as 0
in the strict MAE), `regions.csv` (per-region TP, FP, TN, FN, exact-set match,
Jaccard, unlocalized rate, `from_counts` for regions derived from counts, and
`pred_all_regions_rate` next to `truth_all_regions_rate`), `per_image.csv`,
and `evaluation.json` with a summary: the protocol, micro and macro F1,
complete-case rate, mean false alarms per image, the whole-image micro numbers
under `whole_image`, and, for the quadrant scheme, `side_agreement`.

Two diagnostics decide whether word-based regions are being read:

* `side_agreement`: how often a quadrant the model answered for holds a true
  box on that image side (only findings whose true boxes all lie on one side
  count). Far above 50% confirms the patient-side reading; far below means
  `QUADRANT_WORDS_ARE_PATIENT_SIDE` should be set to `False` (the words move
  to the mirrored windows) and the run repeated. DENTEX is the clean test: its
  FDI quadrant labels are exact.
* `pred_all_regions_rate` far above `truth_all_regions_rate` means the model
  answered A in every region whenever the whole image was positive, i.e. it
  ignored the region clause; switch to `REGION_PROMPT="crop"` for that finding
  set.

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
* With `REGION_PROMPT="crop"` and `COUNT_LEVEL="region"`, the windows overlap
  by 10% of the width and 20% of the height, so teeth on the seams can be
  counted twice; use words for region counts.
* Apical surgery, root resorption, and furcation have very few positives in
  UMFIH; their rows are not statistically meaningful.
## Optional location scoring

In notebook Cell 3, set `EVALUATE_LOCATION = True` (default) to score locations,
or `False` to skip location scoring and location-truth adapter calls. Finding
scores and total-count scores remain enabled; inference, counting questions,
and saved predictions are unchanged. Regional-count metrics and the side check also follow this switch.
Re-run Cell 3, Cell 12, and Cell 13 to evaluate existing results with this setting;
no inference rerun is required.
The report records `summary.evaluate_location`. Re-exporting a report with location
scoring disabled removes its previous location CSVs so stale metrics are not shown.
