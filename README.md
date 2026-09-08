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
answer (Figure 9). Location is never asked. So:

* **Presence** uses the Figure 7 wording verbatim, one finding per call.
* **Counts** use the Figure 9 wording for fillings and the same "How many teeth
  ..." shape for the other tooth-anchored findings. Findings whose boxes are
  regions or devices (bone loss, furcation, apical surgery, appliances, plates)
  are presence-only.
* **Location** is never put into words. For each positive, the same presence
  question is sent to four overlapping quadrant crops; the quadrant set is
  whichever crops answer True. Quadrant is the honest ceiling for this model.
* **No JSON, no region wording, no paraphrase retries, no forced zeros.** One
  greedy call per question. Unparseable answers are recorded as such and
  excluded from the per-finding TP/FP/TN/FN tables, never converted into a
  negative there. The per-image complete-case rate and recall are strict: a
  true finding whose answer was unparseable counts as not caught.

The only public weights are the GGUF conversion of `DentalGPT-7B-1026`. That
checkpoint may predate the reinforcement-learning stage, and the exact sentence
the authors appended to request `<think>/<answer>` tags is unpublished. A short
probe therefore decides once per run whether the suffix is needed ("plain" or
"tagged"), and the run manifest records the choice.

## Files

| File | Role |
| --- | --- |
| `dental_pipeline.py` | prompts, answer extraction, quadrant crops, model runner, probe, resumable run loop, dentist summary |
| `dental_eval.py` | ground-truth loaders (UMFIH YOLO, DENTEX), quadrant geometry, metrics, CSV/JSON export |
| `llama_runtime.py` | llama.cpp build, GGUF download, server process (with image-token flags) |
| `main_notebook.ipynb` | Kaggle runner; edit Cell 3 only |
| `test_dental_pipeline.py` | offline tests with a fake model (`python -m unittest -q test_dental_pipeline`) |

## Calls per image

14 presence calls, plus one count call per positive countable finding, plus
four crop calls per positive when `LOCATION="quadrant"`. A typical image with
four or five positives needs about 35 calls; an all-negative image needs 14.

## Runtime settings that matter

* `--image-max-tokens 6144`: llama.cpp otherwise caps Qwen2.5-VL images at 4096
  tokens and silently downscales a full-size panoramic below training resolution.
* No `--image-min-tokens` floor, so quadrant crops of small panoramics stay at
  native size, as they would under the Hugging Face processor.
* `--ctx-size 16384` so image tokens, question, and a long `<think>` fit.
* `max_tokens 4096`, temperature 0, `repeat_penalty 1.05` (the same value as the
  backbone's generation config; llama.cpp applies it over the last 64 tokens).
* Replies cut off by `max_tokens` without a closed `<answer>` are graded as
  unparseable, never as an answer.
* No system prompt is sent; the GGUF chat template injects Qwen's default one,
  matching the authors' published inference snippet.
* Use Q6_K or Q8_0 weights with the f16 projector for reported numbers.
* Radiographs must be JPEG, PNG, or BMP (what llama.cpp can decode).
* `BACKEND="api"` in Cell 3 sends the same prompts to an OpenAI-compatible
  vision API instead, for a controlled comparison against DentalGPT.

## Datasets

* **UMFIH 14-class set** (Zenodo 15487430, CC BY 4.0 with a non-commercial
  note): the exact ontology, YOLO boxes. Score the 100-image internal test split
  and the 180 external-validation images separately.
* **DENTEX** (Hugging Face `ibrahimhamamci/DENTEX`, CC BY-NC-SA): FDI
  quadrant labels for caries, periapical lesion, and impacted tooth. Gives real
  quadrant ground truth for those three findings. Use the fully labeled train
  split (`training_data/quadrant-enumeration-disease`, 705 images) and the
  validation split (`validation_triple.json`, 50 images); DentalGPT never saw
  DENTEX, so both are held-out. The 250-image test split is raw LabelMe files
  with unmapped Turkish labels and is not supported.

Each dataset is scored on the findings it annotates; unannotated findings are
not counted as negatives. Location truth is scored against the same overlapping
crop windows the model saw: a box counts in every window that holds at least a
quarter of its area. With two datasets the notebook also prints a pooled
table over the shared findings.

## Evaluation outputs

`<OUTPUT_DIR>/<dataset>/evaluation/` holds `presence.csv` (TP, FP, TN, FN,
unparseable, sensitivity, specificity, PPV, F1 per finding, with a
`paper_covered` flag), `counts.csv` (exact, within-1, MAE on true positives and a
strict MAE that scores misses as zero), `regions.csv` (per-crop TP, FP, TN, FN,
exact-set match, Jaccard, unlocalized rate), `per_image.csv`, and
`evaluation.json` with a summary: micro and macro F1, complete-case rate, mean
false alarms per image.

## Caveats

* Ten of the 14 findings are outside the paper's evaluated panoramic labels;
  read the `paper_covered` column before comparing to the paper's 84%.
* Ground-truth boxes are per instance while the model counts teeth, so counts
  for crowns/bridges and multi-box fillings carry definitional error.
* Apical surgery, root resorption, and furcation have very few positives in
  UMFIH; their rows are not statistically meaningful.
