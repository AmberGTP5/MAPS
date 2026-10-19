#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pvr

cd /root/autodl-tmp/robots-planning/data

export CUDA_VISIBLE_DEVICES=0

OUT_DIR="/root/autodl-tmp/robots-planning/data/pvr_eval_hardonly_1000_test"
mkdir -p "${OUT_DIR}"

python eval_generate_plans.py \
  --input-file /root/autodl-tmp/robots-planning/data/dpo_hardonly_1000_test.jsonl \
  --output-file "${OUT_DIR}/base_generated.jsonl" \
  --model-name-or-path /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
  --dtype bfloat16 \
  --device-map auto \
  --temperature 0.0 \
  --max-new-tokens 1024 \
  --trust-remote-code

python eval_score_generations.py \
  --input-file "${OUT_DIR}/base_generated.jsonl" \
  --text-field generated_text \
  --output-prefix "${OUT_DIR}/base"
