#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pvr

cd /root/autodl-tmp/robots-planning

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

exec llamafactory-cli train train_dpo_vanilla_hardonly_1000.yaml
