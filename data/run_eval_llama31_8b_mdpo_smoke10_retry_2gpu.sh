#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pvr

cd /root/autodl-tmp/robots-planning/data

OUT_DIR="/root/autodl-tmp/robots-planning/data/pvr_eval_llama31_8b_hardonly_1000_test_smoke10_retry"
mkdir -p "${OUT_DIR}"

rm -f \
  "${OUT_DIR}/mdpo_retry_generated_part0.jsonl" \
  "${OUT_DIR}/mdpo_retry_generated_part1.jsonl" \
  "${OUT_DIR}/mdpo_retry_generated_10.jsonl" \
  "${OUT_DIR}/mdpo_retry_10_scored.jsonl" \
  "${OUT_DIR}/mdpo_retry_10_report.json" \
  "${OUT_DIR}/mdpo_retry_10_report.md" \
  "${OUT_DIR}/gpu0.log" \
  "${OUT_DIR}/gpu1.log"

COMMON_ARGS=(
  --input-file /root/autodl-tmp/robots-planning/data/dpo_hardonly_1000_test.jsonl
  --model-name-or-path /root/autodl-tmp/models/Llama-3.1-8B-Instruct
  --adapter-path /root/autodl-tmp/robots-planning/saves/llama31_8b/lora/hardonly_1000_mdpo
  --dtype bfloat16
  --device-map auto
  --temperature 0.0
  --max-new-tokens 2048
  --prompt-mode raw
  --retry-on-invalid 1
  --retry-if-nosolution
  --retry-if-parse-fail
  --retry-if-empty-plan
  --retry-if-schema-fail
  --trust-remote-code
)

CUDA_VISIBLE_DEVICES=0 python eval_generate_plans.py \
  "${COMMON_ARGS[@]}" \
  --output-file "${OUT_DIR}/mdpo_retry_generated_part0.jsonl" \
  --start-index 0 \
  --max-samples 5 \
  > "${OUT_DIR}/gpu0.log" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python eval_generate_plans.py \
  "${COMMON_ARGS[@]}" \
  --output-file "${OUT_DIR}/mdpo_retry_generated_part1.jsonl" \
  --start-index 5 \
  --max-samples 5 \
  > "${OUT_DIR}/gpu1.log" 2>&1 &
PID1=$!

wait "${PID0}"
wait "${PID1}"

cat "${OUT_DIR}/mdpo_retry_generated_part0.jsonl" "${OUT_DIR}/mdpo_retry_generated_part1.jsonl" > "${OUT_DIR}/mdpo_retry_generated_10.jsonl"

python eval_score_generations.py \
  --input-file "${OUT_DIR}/mdpo_retry_generated_10.jsonl" \
  --text-field generated_text \
  --output-prefix "${OUT_DIR}/mdpo_retry_10"

python - << 'PY'
import json
from pathlib import Path
p = Path("/root/autodl-tmp/robots-planning/data/pvr_eval_llama31_8b_hardonly_1000_test_smoke10_retry/mdpo_retry_10_report.json")
obj = json.loads(p.read_text(encoding="utf-8"))
print(json.dumps(obj.get("overall", {}), ensure_ascii=False, indent=2))
PY
