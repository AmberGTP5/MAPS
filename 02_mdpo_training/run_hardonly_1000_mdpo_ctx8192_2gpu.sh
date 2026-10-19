#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pvr

cd /root/autodl-tmp/robots-planning

export CUDA_VISIBLE_DEVICES=0,1
export FORCE_TORCHRUN=1
export NPROC_PER_NODE=2
export MASTER_PORT=29643
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec llamafactory-cli train train_mdpo_hardonly_1000_ctx8192_2gpu.yaml
