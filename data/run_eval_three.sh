#!/usr/bin/env bash
set -euo pipefail

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

die() {
  log "ERROR: $*"
  exit 1
}

check_file() {
  [[ -f "$1" ]] || die "file not found: $1"
}

check_dir() {
  [[ -d "$1" ]] || die "directory not found: $1"
}

PROJECT_DIR="${PROJECT_DIR:-/root/autodl-tmp/robots-planning}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/data}"
CONDA_ENV="${CONDA_ENV:-pvr}"

MODEL_PATH="${MODEL_PATH:-${PROJECT_DIR}/models/Qwen2.5-7B-Instruct}"
EVAL_INPUT="${EVAL_INPUT:-${DATA_DIR}/dpo_mdpo_optimized_test.jsonl}"
OUT_DIR="${OUT_DIR:-${DATA_DIR}/pvr_eval_mdpo_compare}"
SAVE_DIR="${SAVE_DIR:-${PROJECT_DIR}/saves/qwen25_7b/lora}"
SFT_ADAPTER="${SFT_ADAPTER:-${SAVE_DIR}/pvr_sft_optimized}"
DPO_VANILLA_ADAPTER="${DPO_VANILLA_ADAPTER:-${SAVE_DIR}/pvr_dpo_vanilla}"
MDPO_ADAPTER="${MDPO_ADAPTER:-${SAVE_DIR}/pvr_mdpo_optimized}"

DTYPE="${DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
ENABLE_LLM_SECOND_PASS="${ENABLE_LLM_SECOND_PASS:-0}"
LLM_TARGET="${LLM_TARGET:-all_samples}"
LLM_MODEL="${LLM_MODEL:-google/gemini-2.5-flash}"
LLM_API_KEY="${LLM_API_KEY:-${OPENROUTER_API_KEY:-}}"
LLM_BASE_URL="${LLM_BASE_URL:-${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}}"
LLM_MAX_RECHECKS="${LLM_MAX_RECHECKS:-0}"
LLM_TEMPERATURE="${LLM_TEMPERATURE:-0.0}"
LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-2048}"

RUN_BASE="${RUN_BASE:-1}"
RUN_SFT="${RUN_SFT:-1}"
RUN_DPO_VANILLA="${RUN_DPO_VANILLA:-1}"
RUN_MDPO="${RUN_MDPO:-1}"

log "=========================================="
log "Unified eval: base / SFT / vanilla DPO / mDPO"
log "=========================================="
log "MODEL_PATH=${MODEL_PATH}"
log "EVAL_INPUT=${EVAL_INPUT}"
log "OUT_DIR=${OUT_DIR}"
log "SFT_ADAPTER=${SFT_ADAPTER}"
log "DPO_VANILLA_ADAPTER=${DPO_VANILLA_ADAPTER}"
log "MDPO_ADAPTER=${MDPO_ADAPTER}"
log "RUN_BASE=${RUN_BASE} RUN_SFT=${RUN_SFT} RUN_DPO_VANILLA=${RUN_DPO_VANILLA} RUN_MDPO=${RUN_MDPO}"
log "ENABLE_LLM_SECOND_PASS=${ENABLE_LLM_SECOND_PASS}"
log "=========================================="

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
cd "${DATA_DIR}"

check_file "eval_generate_plans.py"
check_file "eval_score_generations.py"
check_file "${EVAL_INPUT}"
check_dir "${MODEL_PATH}"
mkdir -p "${OUT_DIR}"
export OUT_DIR

run_one() {
  local label="$1"
  local adapter="$2"

  if [[ -n "${adapter}" ]]; then
    check_dir "${adapter}"
  fi

  local generated_file="${OUT_DIR}/${label}_generated.jsonl"
  local prefix="${OUT_DIR}/${label}"

  local gen_cmd=(
    python eval_generate_plans.py
    --input-file "${EVAL_INPUT}"
    --output-file "${generated_file}"
    --model-name-or-path "${MODEL_PATH}"
    --dtype "${DTYPE}"
    --device-map "${DEVICE_MAP}"
    --temperature "${TEMPERATURE}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --max-samples "${MAX_SAMPLES}"
    --trust-remote-code
  )
  if [[ -n "${adapter}" ]]; then
    gen_cmd+=(--adapter-path "${adapter}")
  fi

  log "Generate (${label})"
  "${gen_cmd[@]}"

  log "Score (${label})"
  local score_cmd=(
    python eval_score_generations.py
    --input-file "${generated_file}"
    --text-field generated_text
    --output-prefix "${prefix}"
  )
  if [[ "${ENABLE_LLM_SECOND_PASS}" == "1" ]]; then
    [[ -n "${LLM_API_KEY}" ]] || die "ENABLE_LLM_SECOND_PASS=1 but LLM_API_KEY/OPENROUTER_API_KEY is empty."
    score_cmd+=(
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
  "${score_cmd[@]}"
}

if [[ "${RUN_BASE}" == "1" ]]; then
  run_one "base" ""
fi
if [[ "${RUN_SFT}" == "1" ]]; then
  run_one "sft" "${SFT_ADAPTER}"
fi
if [[ "${RUN_DPO_VANILLA}" == "1" ]]; then
  run_one "dpo_vanilla" "${DPO_VANILLA_ADAPTER}"
fi
if [[ "${RUN_MDPO}" == "1" ]]; then
  run_one "mdpo" "${MDPO_ADAPTER}"
fi

log "Build compact comparison report"
python - <<'PY'
import json
import os
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
labels = ("base", "sft", "dpo_vanilla", "mdpo")
rows = []

for label in labels:
    report = out_dir / f"{label}_report.json"
    if not report.exists():
        continue
    payload = json.loads(report.read_text(encoding="utf-8"))
    overall = payload.get("overall", {})
    llm_block = payload.get("llm_second_pass") or {}
    llm_stats = llm_block.get("stats") or {}
    rows.append(
        {
            "model": label,
            "parse_ok": overall.get("parse_rate"),
            "schema_pass": overall.get("schema_pass_rate"),
            "executable_plan": overall.get("executable_plan_rate"),
            "exact_delivery": overall.get("exact_delivery_rate"),
            "penalty_score": overall.get("avg_penalty_score"),
            "valid_plan_count": overall.get("valid_plan_count"),
            "n": overall.get("count"),
            "llm_recheck_attempted": llm_stats.get("attempted"),
            "llm_review_returned": llm_stats.get("returned_review"),
            "llm_pass_yes": llm_stats.get("llm_pass_yes"),
            "llm_goal_yes": llm_stats.get("goal_reachability_yes"),
            "llm_terminal_yes": llm_stats.get("terminal_delivery_yes"),
            "llm_symbolic_disagreement": llm_stats.get("symbolic_disagreement_count"),
        }
    )

summary_path = out_dir / "comparison_summary.json"
summary_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(rows, ensure_ascii=False, indent=2))
print(f"[saved] {summary_path}")
PY

log "[OK] Evaluation finished."
