#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pvr

cd /root/autodl-tmp/robots-planning

export CUDA_VISIBLE_DEVICES=0,1
export FORCE_TORCHRUN=1
export NPROC_PER_NODE=2
export MASTER_PORT=29631
export OMP_NUM_THREADS=8

exec llamafactory-cli train train_sft_llama31_8b_hardonly_1000_2gpu.yaml
