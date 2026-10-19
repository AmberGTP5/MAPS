#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data"

TASKS_FILE="${TASKS_FILE:-${DATA_DIR}/app_benchmark_tasks.json}"
RUN_TAG="${RUN_TAG:-local}"
REPORT_PREFIX="${REPORT_PREFIX:-app_local_prior_closure}"
MAX_RETRIES="${MAX_RETRIES:-2}"
SEARCH_ITERATIONS="${SEARCH_ITERATIONS:-96}"
SEARCH_DEPTH="${SEARCH_DEPTH:-18}"
PLANNER_MAX_TOKENS="${PLANNER_MAX_TOKENS:-2048}"
PLANNER_TEMPERATURE="${PLANNER_TEMPERATURE:-0.0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/models/Qwen2.5-7B-Instruct}"
SFT_ADAPTER="${SFT_ADAPTER:-/root/autodl-tmp/robots-planning/saves/qwen25_7b/lora/hardonly_1000_sft/checkpoint-68}"
DPO_ADAPTER="${DPO_ADAPTER:-/root/autodl-tmp/robots-planning/saves/qwen25_7b/lora/hardonly_1000_dpo_vanilla/checkpoint-102}"
MDPO_ADAPTER="${MDPO_ADAPTER:-/root/autodl-tmp/robots-planning/saves/qwen25_7b/lora/hardonly_1000_mdpo/checkpoint-102}"

if [[ ! -f "${TASKS_FILE}" ]]; then
  echo "[ERR] Missing tasks file: ${TASKS_FILE}" >&2
  exit 1
fi
if [[ ! -d "${BASE_MODEL}" ]]; then
  echo "[ERR] Missing base model directory: ${BASE_MODEL}" >&2
  exit 1
fi

run_one() {
  local tag="$1"
  local adapter_path="$2"
  local out_dir="${DATA_DIR}/app_benchmark_${RUN_TAG}_${tag}"

  export CUDA_VISIBLE_DEVICES
  export ROBOT_PLANNER_LOCAL_MODEL_PATH="${BASE_MODEL}"
  export ROBOT_PLANNER_LOCAL_DTYPE="bfloat16"
  export ROBOT_PLANNER_LOCAL_DEVICE_MAP="auto"
  export ROBOT_PLANNER_MAX_TOKENS="${PLANNER_MAX_TOKENS}"
  export ROBOT_PLANNER_TEMPERATURE="${PLANNER_TEMPERATURE}"

  if [[ -n "${adapter_path}" ]]; then
    if [[ ! -d "${adapter_path}" ]]; then
      echo "[ERR] Missing adapter directory for ${tag}: ${adapter_path}" >&2
      exit 1
    fi
    export ROBOT_PLANNER_LOCAL_ADAPTER_PATH="${adapter_path}"
  else
    unset ROBOT_PLANNER_LOCAL_ADAPTER_PATH || true
  fi

  echo "[RUN] tag=${tag} out_dir=${out_dir}"
  python "${DATA_DIR}/run_app_benchmark.py" \
    --tasks-file "${TASKS_FILE}" \
    --output-dir "${out_dir}" \
    --modes greedy retry search \
    --max-retries "${MAX_RETRIES}" \
    --search-iterations "${SEARCH_ITERATIONS}" \
    --search-depth "${SEARCH_DEPTH}" \
    --dry-run \
    --capture-stdout
}

run_one "base" ""
run_one "sft" "${SFT_ADAPTER}"
run_one "dpo" "${DPO_ADAPTER}"
run_one "mdpo" "${MDPO_ADAPTER}"

echo "[RUN] build closure report"
python "${DATA_DIR}/build_app_local_prior_closure.py" \
  --run base=app_benchmark_"${RUN_TAG}"_base \
  --run sft=app_benchmark_"${RUN_TAG}"_sft \
  --run dpo=app_benchmark_"${RUN_TAG}"_dpo \
  --run mdpo=app_benchmark_"${RUN_TAG}"_mdpo \
  --report-prefix "${REPORT_PREFIX}"

echo "[OK] closure done"
