#!/usr/bin/env bash
set -euo pipefail

while true; do
  echo "===== $(date) ====="
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
  df -h /root/autodl-tmp /
  sleep 600
done
