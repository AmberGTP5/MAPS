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
  --output-file "${OUT_DIR}/orpo_generated_4096.jsonl" \
  --model-name-or-path /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
  --adapter-path /root/autodl-tmp/robots-planning/saves/qwen25_7b/lora/hardonly_1000_orpo \
  --dtype bfloat16 \
  --device-map auto \
  --temperature 0.0 \
  --max-new-tokens 4096 \
  --trust-remote-code

python eval_score_generations.py \
  --input-file "${OUT_DIR}/orpo_generated_4096.jsonl" \
  --text-field generated_text \
  --output-prefix "${OUT_DIR}/orpo_4096"
