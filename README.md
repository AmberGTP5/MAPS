# MAPS Code Release

This directory contains the code used for the MAPS paper:

- `mDPO`: margin-aware preference learning for embodied planning
- `VGS`: verifier-guided repair during inference
- `App-200`: benchmark construction, training-data generation, and executable evaluation

## Repository Layout

- `01_data_generation_4phase/`
  Data construction and pair generation scripts, grouped by paper stage.
- `02_mdpo_training/`
  mDPO, SFT, vanilla DPO, SimPO, and ORPO training scripts and YAML configs.
- `03_app_inference/`
  App-200 benchmark inference, retry/search evaluation, and RWARE transfer scripts.
- `app/`
  Shared runtime modules for planning, verifier-based search, LLM routing, and robot tools.
- `data/`
  Canonical runnable mirror used by several shell scripts. This folder is also the default location for generated training artifacts.
- `configs/`
  Extra training configs.
- `framework_minimal/`
  Slimmed-down code pack kept for figure-oriented packaging. It is not the main reproducibility entry point.

The `01_`, `02_`, and `03_` folders are the easiest places to read the code by pipeline stage. The root-level `app/` and `data/` folders are kept because several legacy scripts expect that layout.

## Environment Setup

Use Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you want to run the YAML-based SFT/DPO/mDPO training commands, install LLaMA-Factory separately and make sure `llamafactory-cli` is available in your shell.

## Configuration

Create a local runtime config from the template:

```bash
cp config/config.example.toml config/config.toml
```

The code supports two common runtime modes:

1. OpenAI-compatible API routing

```bash
export OPENROUTER_API_KEY=...
export OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
export OPENROUTER_MODEL=google/gemini-2.5-flash
```

2. Local Hugging Face model routing

```bash
export ROBOT_LOCAL_MODEL_PATH=/path/to/base-model
export ROBOT_LOCAL_ADAPTER_PATH=/path/to/lora-adapter
export ROBOT_LOCAL_DTYPE=bfloat16
export ROBOT_LOCAL_DEVICE_MAP=auto
```

The `app/llm.py` runtime also supports role-specific overrides such as `ROBOT_PLANNER_LOCAL_MODEL_PATH`, `ROBOT_PLANNER_MODEL`, and `ROBOT_PLANNER_BASE_URL`.

## Expected Data Inputs

This code release is separate from the dataset release.

- App-200 benchmark tasks:
  `../datasets/app_benchmark_tasks_random_200.json`
- mDPO pairwise data:
  `../datasets/app200_mdpo_pairs_v3/`

Most scripts accept explicit input paths. Generated files are usually written under the repo-local `data/` folder unless you override the output path.

## 1. Build Pairwise Preference Data

The full data pipeline is split into generation, validation, filtering, and export.

```bash
cd 01_data_generation_4phase
python phase1_generate_openrouter.py
python phase2_validate.py
python phase3_filter.py
python generate_optimized_mdpo_data_fixed.py \
  --input dpo_dataset_filtered.jsonl \
  --output-prefix dpo_margin_dataset_v3 \
  --split-by task \
  --stratify-by robot_profile \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 42
cd ..
python 02_mdpo_training/prepare_llamafactory_data_fixed.py \
  --input-prefix 01_data_generation_4phase/dpo_margin_dataset_v3 \
  --output-dir data/pvr_mdpo_optimized \
  --stage both
```

After the last step, the YAML training configs expect the following directories:

- `data/pvr_mdpo_optimized/sft/`
- `data/pvr_mdpo_optimized/dpo/`

## 2. Train SFT, Vanilla DPO, and mDPO

The paper-stage training configs live in `02_mdpo_training/`.

```bash
cd 02_mdpo_training
llamafactory-cli train train_sft_warmup.yaml
llamafactory-cli train train_dpo_vanilla_fixed.yaml
llamafactory-cli train train_mdpo_optimized_fixed.yaml
```

The default YAML files assume:

- base model at `./models/Qwen2.5-7B-Instruct`
- SFT output at `./saves/qwen25_7b/lora/pvr_sft_optimized`
- DPO output at `./saves/qwen25_7b/lora/pvr_dpo_vanilla`
- mDPO output at `./saves/qwen25_7b/lora/pvr_mdpo_optimized`

Edit the YAML paths if your local layout is different.

There is also a standalone PyTorch trainer:

```bash
python train_margin_dpo.py --help
```

## 3. Run App-200 Benchmark Inference

`run_app_benchmark.py` uses the shared runtime in `app/` and supports greedy, retry, and verifier-guided search modes.

```bash
python 03_app_inference/run_app_benchmark.py \
  --tasks-file ../datasets/app_benchmark_tasks_random_200.json \
  --output-dir ./app_benchmark_results \
  --modes greedy retry search \
  --capture-stdout
```

This script reads its planner backend from `config/config.toml` and environment overrides handled by `app/llm.py`.

## 4. Evaluate Priors Under Greedy, Retry, and Search

Use `run_prior_retry_search_eval.py` when you already have a held-out jsonl evaluation split and want matched comparisons across inference modes.

```bash
python 03_app_inference/run_prior_retry_search_eval.py \
  --input-file /path/to/eval.jsonl \
  --output-dir ./outputs/pvr_eval \
  --label mdpo \
  --model-name-or-path /path/to/base-model \
  --adapter-path /path/to/lora \
  --trust-remote-code
```

## 5. Run RWARE Transfer Scripts

The RWARE scripts require `gymnasium` and `rware`.

```bash
python 03_app_inference/run_rware_generalization.py \
  --env-id rware-tiny-2ag-easy-v2 \
  --episodes 200
```

```bash
python 03_app_inference/run_rware_app_modes.py \
  --env-id rware-tiny-2ag-easy-v2 \
  --modes greedy retry search \
  --episodes 200
```

## Outputs

- Logs are written under `logs/`.
- Generated training artifacts are usually written under `data/`.
- Benchmark and evaluation outputs are written to the `--output-dir` you provide.

## Notes

- The benchmark task JSON and the mDPO pairwise dataset are different assets. The benchmark file is for executable evaluation. The pairwise jsonl files are for preference learning.
- Some shell scripts were originally written for a server layout and use environment variables such as `PROJECT_DIR`, `CONDA_ENV`, and `SAVE_DIR`. Adjust these paths before running them unchanged.
- The files in `framework_minimal/` are optional packaging artifacts and are not required for the main training or inference workflow.
