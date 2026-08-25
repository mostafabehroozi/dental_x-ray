# DentalGPT-7B-1026 GGUF integration

This rewrite replaces the old Transformers/BitsAndBytes runner with a llama.cpp multimodal server.

## Runtime model pair

- `DentalGPT-7B-1026.Q6_K.gguf`
- `DentalGPT-7B-1026.mmproj-f16.gguf`

The model and projector are downloaded from `mradermacher/DentalGPT-7B-1026-GGUF`.

## Architecture

`panoramic image -> llama.cpp + mmproj -> DentalGPT text observations -> Python pipeline -> optional text-only orchestrator`

The DentalGPT layer never generates the final JSON schema directly. It stays close to its reasoning/text behavior; Python normalizes atomic `PRESENT/ABSENT/UNCERTAIN` answers and an optional stronger text model can format the final report.

## Modes

- `BASIC`: four broad independent passes.
- `DISEASE_HIERARCHY`: broad + family + one atomic question per condition.
- `DISEASE_AND_LOCATION`: hierarchy plus three coarse-to-fine location questions for each PRESENT/UNCERTAIN candidate.

## Kaggle

Run `main_notebook.ipynb`. It installs Python dependencies, builds llama.cpp for CUDA SM 60 (Tesla P100), downloads the two GGUF files, starts the local server, then constructs the pipeline.

The optional external orchestrator is disabled by default so DentalGPT can be tested fully locally first.

For a lower-memory/faster development run, switch to `DentalGPT-7B-1026.Q4_K_M.gguf` + `DentalGPT-7B-1026.mmproj-Q8_0.gguf`.

The notebook pins llama.cpp to `b10516` rather than building an arbitrary future `master`, because multimodal APIs are evolving quickly.
