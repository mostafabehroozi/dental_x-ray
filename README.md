# Dental expert-model reporting pipeline

The pipeline treats the image-capable dental foundational model as a replaceable **dental expert model**. The current deployment uses DentalGPT through a llama.cpp multimodal server, but reporting and evaluation code do not depend on that model name or version.

## Runtime model pair

- `DentalGPT-7B-1026.Q6_K.gguf`
- `DentalGPT-7B-1026.mmproj-f16.gguf`

The model and projector are downloaded from `mradermacher/DentalGPT-7B-1026-GGUF`.

## Architecture

`panoramic image -> dental expert model -> multi-level text observations -> dentist-report synthesis -> evaluation adaptation`

The expert-model layer answers independent broad, family, atomic, and location questions. Python preserves those source observations and normalizes atomic `PRESENT/ABSENT/UNCERTAIN` answers. When enabled, the text-only orchestrator runs two separate phases with the same configured model:

1. **Dentist report:** comprehensively combines all useful supported content, removes redundancy, and preserves locations, uncertainty, conflicts, limitations, and relevant negative findings.
2. **Evaluation adaptation report:** maps the completed dentist report into the closed evaluation ontology without changing deterministic atomic statuses.

Neither orchestration phase receives the radiograph.

## Modes

- `BASIC`: four broad independent passes.
- `DISEASE_HIERARCHY`: broad + family + one atomic question per condition.
- `DISEASE_AND_LOCATION`: hierarchy plus three coarse-to-fine location questions for each PRESENT/UNCERTAIN candidate.

## Kaggle

Run `main_notebook.ipynb`. It installs Python dependencies, builds llama.cpp for CUDA SM 60 (Tesla P100), downloads the two GGUF files, starts the local server, then constructs the pipeline.

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
