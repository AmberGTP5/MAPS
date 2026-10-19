#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pvr

cd /root/autodl-tmp/robots-planning

if [[ ! -f data/dataset_info.json ]]; then
  echo "[ERROR] data/dataset_info.json not found."
  exit 1
fi

if [[ ! -f configs/lf_dpo_qwen7b_v3.yaml ]]; then
  echo "[ERROR] configs/lf_dpo_qwen7b_v3.yaml not found."
  exit 1
fi

llamafactory-cli train configs/lf_dpo_qwen7b_v3.yaml
