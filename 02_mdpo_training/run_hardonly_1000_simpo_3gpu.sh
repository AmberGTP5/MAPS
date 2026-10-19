#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pvr

cd /root/autodl-tmp/robots-planning

export CUDA_VISIBLE_DEVICES=0,1,2
export FORCE_TORCHRUN=1
export NPROC_PER_NODE=3
export MASTER_PORT=29625
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec llamafactory-cli train train_simpo_hardonly_1000_3gpu.yaml
