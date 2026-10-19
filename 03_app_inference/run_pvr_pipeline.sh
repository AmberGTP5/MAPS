#!/usr/bin/env bash
set -euo pipefail

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

die() {
  log "ERROR: $*"
  exit 1
}

check_dir() {
  [[ -d "$1" ]] || die "directory not found: $1"
}

check_file() {
  [[ -f "$1" ]] || die "file not found: $1"
}

PROJECT_DIR="${PROJECT_DIR:-/root/autodl-tmp/robots-planning}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/data}"
CLAUDE_DIR="${CLAUDE_DIR:-${PROJECT_DIR}/data_claude}"
CONDA_ENV="${CONDA_ENV:-pvr}"

# Lock defaults to avoid accidental drift (for example stale env set to 800).
export PVR_PHASE1_NUM_SAMPLES="${PVR_PHASE1_NUM_SAMPLES:-500}"
export PVR_PHASE1_MODEL="${PVR_PHASE1_MODEL:-google/gemini-2.5-flash}"
export PVR_PHASE1_TOPK_CANDIDATES="${PVR_PHASE1_TOPK_CANDIDATES:-4}"
export PVR_PHASE1_TOPK_PATHS_PER_ROBOT="${PVR_PHASE1_TOPK_PATHS_PER_ROBOT:-3}"
export PVR_PHASE1_MIN_SOFT_UTILITY_GAP="${PVR_PHASE1_MIN_SOFT_UTILITY_GAP:-2.0}"
export PVR_PHASE1_REJECTED_ATTEMPTS="${PVR_PHASE1_REJECTED_ATTEMPTS:-3}"
export PVR_PHASE1_TASK_DEDUP_ATTEMPTS="${PVR_PHASE1_TASK_DEDUP_ATTEMPTS:-80}"

[[ -n "${OPENROUTER_API_KEY:-}" ]] || die "OPENROUTER_API_KEY is not set."

GEN_SCRIPT="${CLAUDE_DIR}/generate_optimized_mdpo_data_fixed.py"
PREP_SCRIPT="${CLAUDE_DIR}/prepare_llamafactory_data_fixed.py"
if [[ ! -f "${GEN_SCRIPT}" ]]; then
  GEN_SCRIPT="${DATA_DIR}/generate_optimized_mdpo_data_fixed.py"
fi
if [[ ! -f "${PREP_SCRIPT}" ]]; then
  PREP_SCRIPT="${DATA_DIR}/prepare_llamafactory_data_fixed.py"
fi

check_dir "${PROJECT_DIR}"
check_dir "${DATA_DIR}"
check_file "${DATA_DIR}/phase1_generate_openrouter.py"
check_file "${DATA_DIR}/phase2_validate.py"
check_file "${DATA_DIR}/phase3_filter.py"
check_file "${GEN_SCRIPT}"
check_file "${PREP_SCRIPT}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
cd "${DATA_DIR}"

log "=========================================="
log "PVR 4-phase data pipeline (Claude-aligned)"
log "=========================================="
log "PROJECT_DIR=${PROJECT_DIR}"
log "DATA_DIR=${DATA_DIR}"
log "PHASE1_MODEL=${PVR_PHASE1_MODEL}"
log "PHASE1_NUM_SAMPLES=${PVR_PHASE1_NUM_SAMPLES}"
log "TOPK_CANDIDATES=${PVR_PHASE1_TOPK_CANDIDATES}"
log "TOPK_PATHS_PER_ROBOT=${PVR_PHASE1_TOPK_PATHS_PER_ROBOT}"
log "MIN_SOFT_UTILITY_GAP=${PVR_PHASE1_MIN_SOFT_UTILITY_GAP}"
log "=========================================="

log "Step 1/5: phase1_generate_openrouter.py"
python phase1_generate_openrouter.py

log "Step 2/5: phase2_validate.py"
python phase2_validate.py

log "Step 3/5: phase3_filter.py"
python phase3_filter.py

log "Step 4/5: generate_optimized_mdpo_data_fixed.py"
python "${GEN_SCRIPT}" \
  --input "${DATA_DIR}/dpo_dataset_filtered.jsonl" \
  --output-prefix "${DATA_DIR}/dpo_mdpo_optimized" \
  --split-by task \
  --stratify-by robot_profile \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 42 \
  --no-format-augmentation

log "Step 5/5: prepare_llamafactory_data_fixed.py"
python "${PREP_SCRIPT}" \
  --input-prefix "${DATA_DIR}/dpo_mdpo_optimized" \
  --output-dir "${DATA_DIR}/pvr_mdpo_optimized" \
  --stage both

log "Final quick stats"
python - <<'PY'
import json
from collections import Counter
from pathlib import Path

data_dir = Path(".")
paths = {
    "phase1": data_dir / "dpo_dataset_phase1.jsonl",
    "validated": data_dir / "dpo_dataset_validated.jsonl",
    "filtered": data_dir / "dpo_dataset_filtered.jsonl",
    "train": data_dir / "dpo_mdpo_optimized_train.jsonl",
    "val": data_dir / "dpo_mdpo_optimized_val.jsonl",
    "test": data_dir / "dpo_mdpo_optimized_test.jsonl",
}

def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.open("r", encoding="utf-8"))

for name, path in paths.items():
    print(f"{name:>10}: {line_count(path)}")

filtered = paths["filtered"]
if filtered.exists():
    c = Counter()
    for line in filtered.open("r", encoding="utf-8"):
        row = json.loads(line)
        c[row.get("pair_type", "hard_invalid")] += 1
    total = sum(c.values())
    ratio = (c.get("soft_suboptimal", 0) / total) if total else 0.0
    print(f" pair_type: {dict(c)}")
    print(f" soft_ratio: {ratio:.3f}")
PY

log "[OK] Data pipeline finished."
