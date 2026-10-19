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
EVAL_INPUT="${EVAL_INPUT:-${DATA_DIR}/eval_id344_unseen.jsonl}"
OUT_DIR="${OUT_DIR:-${DATA_DIR}/pvr_eval_p1_preference_sota_1000}"

DO_TRAIN="${DO_TRAIN:-1}"
DO_EVAL="${DO_EVAL:-1}"

ENABLE_LLM_SECOND_PASS="${ENABLE_LLM_SECOND_PASS:-0}"
LLM_TARGET="${LLM_TARGET:-all_samples}"
LLM_MODEL="${LLM_MODEL:-google/gemini-2.5-flash}"
LLM_API_KEY="${LLM_API_KEY:-${OPENROUTER_API_KEY:-}}"
LLM_BASE_URL="${LLM_BASE_URL:-${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}}"
LLM_MAX_RECHECKS="${LLM_MAX_RECHECKS:-0}"
LLM_TEMPERATURE="${LLM_TEMPERATURE:-0.0}"
LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-2048}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
cd "${PROJECT_DIR}"

[[ -d "${MODEL_PATH}" ]] || die "model path not found: ${MODEL_PATH}"
[[ -f "${EVAL_INPUT}" ]] || die "eval input not found: ${EVAL_INPUT}"
mkdir -p "${OUT_DIR}"
export OUT_DIR

if [[ "${DO_TRAIN}" == "1" ]]; then
  log "Train ORPO (scale=1000, hard-only)"
  llamafactory-cli train "${DATA_DIR}/train_orpo_p1_scale_1000.yaml"

  log "Train SimPO (scale=1000, hard-only)"
  llamafactory-cli train "${DATA_DIR}/train_simpo_p1_scale_1000.yaml"
fi

run_eval_one() {
  local label="$1"
  local adapter="$2"
  local generated_file="${OUT_DIR}/${label}_generated.jsonl"
  local prefix="${OUT_DIR}/${label}"

  [[ -d "${adapter}" ]] || die "adapter not found: ${adapter}"

  log "Generate (${label})"
  python "${DATA_DIR}/eval_generate_plans.py" \
    --input-file "${EVAL_INPUT}" \
    --output-file "${generated_file}" \
    --model-name-or-path "${MODEL_PATH}" \
    --adapter-path "${adapter}" \
    --dtype bfloat16 \
    --device-map auto \
    --temperature 0.0 \
    --max-new-tokens 2048 \
    --max-samples 0 \
    --trust-remote-code

  local score_cmd=(
    python "${DATA_DIR}/eval_score_generations.py"
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

  log "Score (${label})"
  "${score_cmd[@]}"
}

if [[ "${DO_EVAL}" == "1" ]]; then
  SAVE_DIR="${PROJECT_DIR}/saves/qwen25_7b/lora"
  run_eval_one "orpo" "${SAVE_DIR}/p1_scale_1000_orpo"
  run_eval_one "simpo" "${SAVE_DIR}/p1_scale_1000_simpo"
fi

log "Build compact summary"
python - <<'PY'
import json
import os
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
rows = []
for label in ("orpo", "simpo"):
    report_path = out_dir / f"{label}_report.json"
    if not report_path.exists():
        continue
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    overall = payload.get("overall", {})
    rows.append(
        {
            "model": label,
            "count": overall.get("count"),
            "parse_rate": overall.get("parse_rate"),
            "schema_pass_rate": overall.get("schema_pass_rate"),
            "executable_plan_rate": overall.get("executable_plan_rate"),
            "exact_delivery_rate": overall.get("exact_delivery_rate"),
            "avg_penalty_score": overall.get("avg_penalty_score"),
        }
    )

summary = out_dir / "comparison_summary_p1_preference_sota.json"
summary.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(rows, ensure_ascii=False, indent=2))
print(f"[saved] {summary}")
PY

log "[OK] P1-2 preference SOTA run finished."
