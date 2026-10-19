#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pvr

cd /root/autodl-tmp/robots-planning/data

GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"

OUT_DIR="/root/autodl-tmp/robots-planning/data/pvr_eval_llama31_8b_hardonly_1000_test"
mkdir -p "${OUT_DIR}"

INPUT_FILE="/root/autodl-tmp/robots-planning/data/dpo_hardonly_1000_test.jsonl"
OUT_PART0="${OUT_DIR}/mdpo_generated_4096_part0.jsonl"
OUT_PART1="${OUT_DIR}/mdpo_generated_4096_part1.jsonl"
OUT_MERGED="${OUT_DIR}/mdpo_generated_4096.jsonl"

TOTAL_LINES="$(wc -l < "${INPUT_FILE}")"
if [[ "${TOTAL_LINES}" -le 0 ]]; then
  echo "empty eval input: ${INPUT_FILE}"
  exit 1
fi

SHARD0_COUNT=$(( (TOTAL_LINES + 1) / 2 ))
SHARD1_START="${SHARD0_COUNT}"
SHARD1_COUNT=$(( TOTAL_LINES - SHARD0_COUNT ))

CUDA_VISIBLE_DEVICES="${GPU_A}" python eval_generate_plans.py \
  --input-file "${INPUT_FILE}" \
  --output-file "${OUT_PART0}" \
  --model-name-or-path /root/autodl-tmp/models/Llama-3.1-8B-Instruct \
  --adapter-path /root/autodl-tmp/robots-planning/saves/llama31_8b/lora/hardonly_1000_mdpo \
  --dtype bfloat16 \
  --device-map auto \
  --temperature 0.0 \
  --max-new-tokens 4096 \
  --start-index 0 \
  --max-samples "${SHARD0_COUNT}" \
  --trust-remote-code \
  > "${OUT_DIR}/mdpo_gen_gpu${GPU_A}.log" 2>&1 &
PID0=$!

if [[ "${SHARD1_COUNT}" -gt 0 ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_B}" python eval_generate_plans.py \
    --input-file "${INPUT_FILE}" \
    --output-file "${OUT_PART1}" \
    --model-name-or-path /root/autodl-tmp/models/Llama-3.1-8B-Instruct \
    --adapter-path /root/autodl-tmp/robots-planning/saves/llama31_8b/lora/hardonly_1000_mdpo \
    --dtype bfloat16 \
    --device-map auto \
    --temperature 0.0 \
    --max-new-tokens 4096 \
    --start-index "${SHARD1_START}" \
    --max-samples "${SHARD1_COUNT}" \
    --trust-remote-code \
    > "${OUT_DIR}/mdpo_gen_gpu${GPU_B}.log" 2>&1 &
  PID1=$!
else
  PID1=""
fi

wait "${PID0}"
if [[ -n "${PID1}" ]]; then
  wait "${PID1}"
fi

[[ -f "${OUT_PART0}" ]] || { echo "missing shard-0 output"; exit 1; }
if [[ "${SHARD1_COUNT}" -gt 0 ]]; then
  [[ -f "${OUT_PART1}" ]] || { echo "missing shard-1 output"; exit 1; }
fi

cat "${OUT_PART0}" > "${OUT_MERGED}"
if [[ "${SHARD1_COUNT}" -gt 0 ]]; then
  cat "${OUT_PART1}" >> "${OUT_MERGED}"
fi

python eval_score_generations.py \
  --input-file "${OUT_MERGED}" \
  --text-field generated_text \
  --output-prefix "${OUT_DIR}/mdpo_4096"
