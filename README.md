# DentVLM panoramic findings pipeline

A small wrapper that gets findings, their regions, and a multiplicity signal out
of DentVLM (a 7B dental vision-language model) on panoramic radiographs, while
sending it only questions it was trained and evaluated on. Everything else
(task decomposition, aggregation, scoring) happens in Python.

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
  behind `Protocol(count_question=True)` as an explicitly out-of-distribution
  experiment.
* **Findings without a DentVLM task** (furcation involvement, apical surgery,
  root resorption, orthodontic appliances, surgical plates) are not asked and
  are reported as "not assessed by this model". `Protocol(ask_untrained=True)`
  asks them anyway and scores them under `trained_task=False`; the paper's
  zero-shot accuracy on untrained diseases is 52-64%.
* **Prosthetic restoration** (crowns or bridges in the benchmark) is the OR of
  the prosthetic crown and prosthetic bridge tasks, regions merged.
* **No JSON, no region wording, no paraphrase retries, no forced zeros.** One
  greedy call per question. Unparseable answers are recorded as such and
  excluded from the per-finding TP/FP/TN/FN tables, never converted into a
  negative there. The per-image complete-case rate and recall are strict.

Left and right follow the model's own convention (Supplementary Table S6): its
"left posterior region" is FDI quadrants 1 and 4, the patient's right, which is
the left side of a panoramic as displayed. `dental_pipeline.LEFT_IS_IMAGE_LEFT`
records that reading, the dentist summary translates cells to the patient's
side, and the notebook checks the convention against DENTEX boxes.

## Files

| File | Role |
| --- | --- |
| `dental_pipeline.py` | task table and verbatim questions, answer and region extraction, protocol knobs, model runner, resumable run loop, dentist summary |
| `dental_eval.py` | ground-truth loaders (UMFIH YOLO, DENTEX with FDI tooth numbers), location truth (adapted, FDI, or fixed windows), metrics, side-convention check, CSV/JSON export |
| `location_adapter.py` | translates ground-truth boxes into the six cells: vision-LLM adapter (numbered boxes drawn on the image), experimental DentVLM spotlight adapter, resumable per-dataset run |
| `llama_runtime.py` | llama.cpp build, one-time GGUF conversion of the Hugging Face checkpoint, GGUF download, server process (with image-token flags) |
| `llm_api.py` | hosted-model access shared by the runner and the adapter: provider table, key lookup (environment variable or Kaggle secret), client construction |
| `main_notebook.ipynb` | Kaggle runner; edit Cell 3 only |
| `test_dental_pipeline.py`, `test_location_adapter.py`, `test_llm_api.py` | offline tests with fake models (`python -m unittest -q`) |

## Calls per image

13 whole-image calls, the same for every image (one per task; crown and bridge
are separate tasks). Optional knobs in `dental_pipeline.Protocol`:

* `phrasings=3`: ask three verbatim wordings per task and vote (majority for
  yes/no; `region_vote="union"` is the paper's matching voting, `"majority"`
  its majority voting). 39 calls per image. In-distribution.
* `count_question=True`: one count call per positive countable finding.
  Out-of-distribution.
* `location="crops"`: the primary question on six cell crops per positive task,
  scored on the same cells for comparison. Cropped panoramics are outside the
  model's image distribution.
* `location="none"`: presence only.

## Runtime settings that matter

* The model is published only as bf16 safetensors (Hugging Face
  `ZJU-AI4H/DentVLM`, gated with automatic approval, CC BY-NC 4.0). Accept the
  license once, store a token as a Kaggle secret, and let Cell 6 convert it
  with llama.cpp's converter (Q8_0 language model, f16 vision projector, about
  9.5 GB kept, 17 GB scratch). Keep the two files in a private Kaggle dataset
  or your own Hugging Face repo for later sessions.
* `--image-max-tokens 8192` mirrors the authors' `max_pixels` of 8192 x 28 x 28;
  llama.cpp would otherwise cap Qwen2-VL images at 4096 tokens. No token floor.
  The paper's ablation found a 1024 x 1024 bound best for disease tasks;
  `IMAGE_MAX_TOKENS = 1369` reproduces it for an A/B.
* `--ctx-size 16384`, the authors' maximum input length.
* `max_tokens 512` (the authors' output cap), temperature 0, `repeat_penalty
  1.05` as in their inference script. Line 1 is present even when a rationale
  is cut off, so truncated replies are still graded.
* No system prompt is sent; the chat template injects Qwen's default one, which
  the authors' vLLM script sets explicitly.
* Radiographs must be JPEG, PNG, or BMP (what llama.cpp can decode).
* `BACKEND="api"` in Cell 3 sends the same questions to a hosted vision model
  instead, for a controlled comparison (see "Hosted models").

## Hosted models

The model behind `BACKEND="api"` and the location-truth adapter are both
described by a small spec dict in Cell 3, e.g.
`{"provider": "openrouter", "model": "qwen/qwen3-vl-235b-a22b-thinking"}`.
`llm_api.PROVIDERS` holds the endpoints (`openai`, `nvidia`, `openrouter`,
`gemini`) and the name of the key each one reads: an environment variable, else
the Kaggle secret of the same name (`OPENAI_API_KEY`, `NVIDIA_API_KEY`,
`OPENROUTER_API_KEY`, `GEMINI_API_KEY`; Add-ons > Secrets). Optional keys of a
spec: `api_key` (paste one for a quick test, never commit it), `base_url` and
`api_key_env` (any other OpenAI-compatible endpoint), `token_param` and
`temperature` (`"max_completion_tokens"` and `None` for OpenAI reasoning
models), `request_options` (extra request fields, e.g. OpenRouter routing under
`extra_body`). `VisionRunner.from_api` and `LLMAdapter.from_api` build the
clients; run manifests record `llm_api.public(spec)`, never the key. Transport
errors, rate limits and 5xx replies are retried by the client with backoff; a
bad request or key fails at once.

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
Methods 4.2), and `LOCATION_TRUTH` in Cell 3 picks how this project does it:

* `"llm"` (recommended): `location_adapter.LLMAdapter` draws numbered boxes on
  the radiograph, burns the FDI quadrant names into the corners, and asks a
  strong vision model through any OpenAI-compatible API (GPT-5, Gemini's
  compatibility endpoint, NIM, ...) for the dental-arch units each box
  occupies: FDI quadrant x anterior (incisors and canine) / posterior
  (premolars, molars and behind), plus the FDI tooth positions. Units are the
  finest division the six cells are made of, so the mapping onto cells is
  deterministic (`dental_pipeline.unit_cell`) and follows the same
  `LEFT_IS_IMAGE_LEFT` reading as the model's own words. One call per image
  (chunked above `max_boxes_per_call` boxes), strict JSON back, one retry when
  the reply is incomplete, and a box the model cannot place falls back to the
  windows. The model is a spec in Cell 3 (`ADAPTER_API`, see "Hosted models");
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

The adapter runs once per dataset and adapter (Cell 12), independently of the
model run, and resumes: one JSON per image under
`<dataset>/location_truth/<adapter>/boxes` with the raw reply, the units, the
cells, the windows' answer and the source that placed the box; the drawn or
spotlighted images are kept under `.../drawn` for audit. DENTEX boxes carry FDI
tooth numbers, which give exact cells, so on a DENTEX dataset Cell 12 also
prints the adapter's and the windows' agreement with that exact truth
(`dental_eval.truth_agreement`): the check that the adapter is worth its calls.
`evaluation.json` records under `summary.location_truth` how many true boxes
each source placed.

## Evaluation outputs

`<OUTPUT_DIR>/<dataset>/evaluation/` holds `presence.csv` (TP, FP, TN, FN,
unparseable, sensitivity, specificity, PPV, F1 per finding, with a
`trained_task` flag), `regions.csv` (per-cell TP, FP, TN, FN, exact-set match,
Jaccard, unlocalized rate), `counts.csv` (only when the count question was
asked), `per_image.csv`, and `evaluation.json` with a summary: micro and macro
F1, complete-case rate, mean false alarms per image, and the list of findings
not assessed.

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
