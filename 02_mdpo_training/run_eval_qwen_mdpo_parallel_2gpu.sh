#!/usr/bin/env bash
set -euo pipefail

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

PROJECT_DIR="${PROJECT_DIR:-/root/autodl-tmp/robots-planning}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/data}"
CONDA_ENV="${CONDA_ENV:-pvr}"

MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-7B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH:-${PROJECT_DIR}/saves/qwen25_7b/lora/hardonly_1000_mdpo_ctx8192}"
EVAL_INPUT="${EVAL_INPUT:-${DATA_DIR}/dpo_hardonly_1000_test.jsonl}"
OUT_DIR="${OUT_DIR:-${DATA_DIR}/pvr_eval_hardonly_1000_test_ctx8192}"
LABEL="${LABEL:-mdpo_ctx8192_4096}"

DTYPE="${DTYPE:-bfloat16}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
DEVICE_MAP="${DEVICE_MAP:-auto}"

ENABLE_LLM_SECOND_PASS="${ENABLE_LLM_SECOND_PASS:-0}"
LLM_TARGET="${LLM_TARGET:-all_samples}"
LLM_MODEL="${LLM_MODEL:-google/gemini-2.5-flash}"
LLM_API_KEY="${LLM_API_KEY:-${OPENROUTER_API_KEY:-}}"
LLM_BASE_URL="${LLM_BASE_URL:-${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}}"
LLM_MAX_RECHECKS="${LLM_MAX_RECHECKS:-0}"
LLM_TEMPERATURE="${LLM_TEMPERATURE:-0.0}"
LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-2048}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
cd "${DATA_DIR}"

[[ -f "${EVAL_INPUT}" ]] || { log "missing input: ${EVAL_INPUT}"; exit 1; }
[[ -d "${MODEL_PATH}" ]] || { log "missing model: ${MODEL_PATH}"; exit 1; }
[[ -d "${ADAPTER_PATH}" ]] || { log "missing adapter: ${ADAPTER_PATH}"; exit 1; }
mkdir -p "${OUT_DIR}"

TOTAL_LINES="$(wc -l < "${EVAL_INPUT}")"
if [[ "${TOTAL_LINES}" -le 0 ]]; then
  log "empty eval input: ${EVAL_INPUT}"
  exit 1
fi

SHARD0_COUNT=$(( (TOTAL_LINES + 1) / 2 ))
SHARD1_START="${SHARD0_COUNT}"
SHARD1_COUNT=$(( TOTAL_LINES - SHARD0_COUNT ))

OUT0="${OUT_DIR}/${LABEL}_generated_part0.jsonl"
OUT1="${OUT_DIR}/${LABEL}_generated_part1.jsonl"
OUT_MERGED="${OUT_DIR}/${LABEL}_generated.jsonl"
OUT_PREFIX="${OUT_DIR}/${LABEL}"

log "total=${TOTAL_LINES}, shard0=${SHARD0_COUNT}, shard1=${SHARD1_COUNT}"
log "launch shard-0 on GPU0"
CUDA_VISIBLE_DEVICES=0 python eval_generate_plans.py \
  --input-file "${EVAL_INPUT}" \
  --output-file "${OUT0}" \
  --model-name-or-path "${MODEL_PATH}" \
  --adapter-path "${ADAPTER_PATH}" \
  --dtype "${DTYPE}" \
  --device-map "${DEVICE_MAP}" \
  --temperature "${TEMPERATURE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --start-index 0 \
  --max-samples "${SHARD0_COUNT}" \
  --trust-remote-code \
  > "${OUT_DIR}/${LABEL}_gen_gpu0.log" 2>&1 &
PID0=$!

if [[ "${SHARD1_COUNT}" -gt 0 ]]; then
  log "launch shard-1 on GPU1"
  CUDA_VISIBLE_DEVICES=1 python eval_generate_plans.py \
    --input-file "${EVAL_INPUT}" \
    --output-file "${OUT1}" \
    --model-name-or-path "${MODEL_PATH}" \
    --adapter-path "${ADAPTER_PATH}" \
    --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}" \
    --temperature "${TEMPERATURE}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --start-index "${SHARD1_START}" \
    --max-samples "${SHARD1_COUNT}" \
    --trust-remote-code \
    > "${OUT_DIR}/${LABEL}_gen_gpu1.log" 2>&1 &
  PID1=$!
else
  PID1=""
fi

wait "${PID0}"
if [[ -n "${PID1}" ]]; then
  wait "${PID1}"
fi

[[ -f "${OUT0}" ]] || { log "missing shard0 output"; exit 1; }
if [[ "${SHARD1_COUNT}" -gt 0 ]]; then
  [[ -f "${OUT1}" ]] || { log "missing shard1 output"; exit 1; }
fi

cat "${OUT0}" > "${OUT_MERGED}"
if [[ "${SHARD1_COUNT}" -gt 0 ]]; then
  cat "${OUT1}" >> "${OUT_MERGED}"
fi

log "merged generation saved: ${OUT_MERGED}"

SCORE_CMD=(
  python eval_score_generations.py
  --input-file "${OUT_MERGED}"
  --text-field generated_text
  --output-prefix "${OUT_PREFIX}"
)

if [[ "${ENABLE_LLM_SECOND_PASS}" == "1" ]]; then
  [[ -n "${LLM_API_KEY}" ]] || { log "ENABLE_LLM_SECOND_PASS=1 but API key is empty"; exit 1; }
  SCORE_CMD+=(
    --llm-second-pass
    --llm-target "${LLM_TARGET}"
    --llm-model "${LLM_MODEL}"
    --llm-api-key "${LLM_API_KEY}"
    --llm-base-url "${LLM_BASE_URL}"
    --llm-max-rechecks "${LLM_MAX_RECHECKS}"
    --llm-temperature "${LLM_TEMPERATURE}"
    --llm-max-tokens "${LLM_MAX_TOKENS}"
  )
fi

log "start scoring"
"${SCORE_CMD[@]}"

log "done"
log "report: ${OUT_PREFIX}_report.json"
