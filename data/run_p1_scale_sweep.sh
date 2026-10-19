#!/usr/bin/env bash
set -euo pipefail

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

die() {
  log "ERROR: $*"
  exit 1
}

PROJECT_DIR="${PROJECT_DIR:-/root/autodl-tmp/robots-planning}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/data}"
CONDA_ENV="${CONDA_ENV:-pvr}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_DIR}/models/Qwen2.5-7B-Instruct}"
SCALE_LIST="${SCALE_LIST:-375 1000}"

# Fixed held-out eval set for P1 fair comparison (same across all scales/methods).
EVAL_INPUT="${EVAL_INPUT:-${DATA_DIR}/eval_id344_unseen.jsonl}"

# 0/1 switches
DO_TRAIN="${DO_TRAIN:-1}"
DO_EVAL="${DO_EVAL:-1}"

# optional second-pass LLM audit
ENABLE_LLM_SECOND_PASS="${ENABLE_LLM_SECOND_PASS:-0}"
LLM_TARGET="${LLM_TARGET:-all_samples}"
LLM_MODEL="${LLM_MODEL:-google/gemini-2.5-flash}"
LLM_API_KEY="${LLM_API_KEY:-${OPENROUTER_API_KEY:-}}"
LLM_BASE_URL="${LLM_BASE_URL:-${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}}"
LLM_MAX_RECHECKS="${LLM_MAX_RECHECKS:-0}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
cd "${PROJECT_DIR}"

[[ -d "${MODEL_PATH}" ]] || die "model path not found: ${MODEL_PATH}"
[[ -f "${EVAL_INPUT}" ]] || die "eval input not found: ${EVAL_INPUT}"

if [[ "${DO_TRAIN}" == "1" ]]; then
  log "==== P1-1 training starts ===="
  for scale in ${SCALE_LIST}; do
    log "[scale=${scale}] train SFT"
    llamafactory-cli train "${DATA_DIR}/train_sft_p1_scale_${scale}.yaml"

    log "[scale=${scale}] train vanilla DPO"
    llamafactory-cli train "${DATA_DIR}/train_dpo_vanilla_p1_scale_${scale}.yaml"

    log "[scale=${scale}] train mDPO"
    llamafactory-cli train "${DATA_DIR}/train_mdpo_p1_scale_${scale}.yaml"
  done
fi

if [[ "${DO_EVAL}" == "1" ]]; then
  log "==== P1-1 eval starts ===="
  for scale in ${SCALE_LIST}; do
    OUT_DIR="${DATA_DIR}/pvr_eval_p1_scale_${scale}"
    SAVE_DIR="${PROJECT_DIR}/saves/qwen25_7b/lora"

    export PROJECT_DIR DATA_DIR MODEL_PATH EVAL_INPUT OUT_DIR SAVE_DIR
    export SFT_ADAPTER="${SAVE_DIR}/p1_scale_${scale}_sft"
    export DPO_VANILLA_ADAPTER="${SAVE_DIR}/p1_scale_${scale}_dpo_vanilla"
    export MDPO_ADAPTER="${SAVE_DIR}/p1_scale_${scale}_mdpo"
    export RUN_BASE=0 RUN_SFT=1 RUN_DPO_VANILLA=1 RUN_MDPO=1
    export ENABLE_LLM_SECOND_PASS LLM_TARGET LLM_MODEL LLM_API_KEY LLM_BASE_URL LLM_MAX_RECHECKS

    log "[scale=${scale}] unified eval"
    bash "${DATA_DIR}/run_eval_three.sh"
  done
fi

log "[OK] P1 scale sweep finished."
