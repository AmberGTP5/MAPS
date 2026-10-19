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
CONDA_ENV="${CONDA_ENV:-pvr}"
SAVE_DIR="${SAVE_DIR:-${PROJECT_DIR}/saves/qwen25_7b/lora}"

SFT_DATA_DIR="${SFT_DATA_DIR:-${PROJECT_DIR}/data/pvr_mdpo_optimized/sft}"
DPO_DATA_DIR="${DPO_DATA_DIR:-${PROJECT_DIR}/data/pvr_mdpo_optimized/dpo}"

# Prefer Claude configs, fallback to data configs.
CONFIG_DIR="${CONFIG_DIR:-${PROJECT_DIR}/data_claude}"
if [[ ! -d "${CONFIG_DIR}" ]]; then
  CONFIG_DIR="${PROJECT_DIR}/data"
fi

RUN_SFT="${RUN_SFT:-1}"
RUN_DPO_VANILLA="${RUN_DPO_VANILLA:-1}"
RUN_MDPO="${RUN_MDPO:-1}"

SFT_CONFIG="${CONFIG_DIR}/train_sft_warmup.yaml"
VANILLA_CONFIG="${CONFIG_DIR}/train_dpo_vanilla_fixed.yaml"
MDPO_CONFIG="${CONFIG_DIR}/train_mdpo_optimized_fixed.yaml"

log "=========================================="
log "Training pipeline: SFT / vanilla DPO / mDPO"
log "=========================================="
log "PROJECT_DIR=${PROJECT_DIR}"
log "CONFIG_DIR=${CONFIG_DIR}"
log "SFT_DATA_DIR=${SFT_DATA_DIR}"
log "DPO_DATA_DIR=${DPO_DATA_DIR}"
log "RUN_SFT=${RUN_SFT} RUN_DPO_VANILLA=${RUN_DPO_VANILLA} RUN_MDPO=${RUN_MDPO}"
log "=========================================="

[[ "${RUN_SFT}" == "1" || "${RUN_DPO_VANILLA}" == "1" || "${RUN_MDPO}" == "1" ]] || \
  die "At least one of RUN_SFT/RUN_DPO_VANILLA/RUN_MDPO must be 1."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
cd "${PROJECT_DIR}"

check_file "${SFT_DATA_DIR}/train.jsonl"
check_file "${SFT_DATA_DIR}/val.jsonl"
check_file "${SFT_DATA_DIR}/dataset_info.json"
check_file "${DPO_DATA_DIR}/train.jsonl"
check_file "${DPO_DATA_DIR}/val.jsonl"
check_file "${DPO_DATA_DIR}/dataset_info.json"

if [[ "${RUN_SFT}" == "1" ]]; then
  check_file "${SFT_CONFIG}"
  log "Start SFT warm-up"
  llamafactory-cli train "${SFT_CONFIG}"
fi

SFT_CKPT="${SAVE_DIR}/pvr_sft_optimized"
if [[ "${RUN_DPO_VANILLA}" == "1" || "${RUN_MDPO}" == "1" ]]; then
  check_dir "${SFT_CKPT}"
fi

if [[ "${RUN_DPO_VANILLA}" == "1" ]]; then
  check_file "${VANILLA_CONFIG}"
  log "Start vanilla DPO"
  llamafactory-cli train "${VANILLA_CONFIG}"
fi

if [[ "${RUN_MDPO}" == "1" ]]; then
  check_file "${MDPO_CONFIG}"
  log "Start mDPO"
  llamafactory-cli train "${MDPO_CONFIG}"
fi

log "=========================================="
log "Training finished"
log "SFT checkpoint        : ${SFT_CKPT}"
log "vanilla DPO checkpoint: ${SAVE_DIR}/pvr_dpo_vanilla"
log "mDPO checkpoint       : ${SAVE_DIR}/pvr_mdpo_optimized"
log "=========================================="
