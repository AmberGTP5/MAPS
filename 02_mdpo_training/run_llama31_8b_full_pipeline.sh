#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pvr

cd /root/autodl-tmp/robots-planning

LOG_DIR="./logs"
mkdir -p "${LOG_DIR}"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  echo "[$(timestamp)] $*"
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    log "Missing required file: ${path}"
    exit 1
  fi
}

wait_for_pid_exit() {
  local pid="$1"
  log "Waiting for SFT PID ${pid} to finish..."
  while kill -0 "${pid}" 2>/dev/null; do
    sleep 60
  done
  log "Observed SFT PID ${pid} exit."
}

wait_for_adapter() {
  local dir="$1"
  local safetensors="${dir}/adapter_model.safetensors"
  local bin="${dir}/adapter_model.bin"
  log "Checking adapter output in ${dir}"
  for _ in $(seq 1 20); do
    if [[ -f "${safetensors}" || -f "${bin}" ]]; then
      log "Adapter checkpoint is present."
      return 0
    fi
    sleep 15
  done
  log "Adapter checkpoint did not appear in ${dir}"
  exit 1
}

run_parallel_eval_pair() {
  local left_gpu="$1"
  local left_script="$2"
  local left_log="$3"
  local right_gpu="$4"
  local right_script="$5"
  local right_log="$6"

  log "Launching parallel eval: ${left_script} on GPU ${left_gpu}, ${right_script} on GPU ${right_gpu}"
  GPU_ID="${left_gpu}" bash "./${left_script}" > "${LOG_DIR}/${left_log}" 2>&1 &
  local left_pid=$!
  GPU_ID="${right_gpu}" bash "./${right_script}" > "${LOG_DIR}/${right_log}" 2>&1 &
  local right_pid=$!

  wait "${left_pid}"
  wait "${right_pid}"
  log "Parallel eval pair finished."
}

run_single_eval() {
  local gpu="$1"
  local script="$2"
  local log_name="$3"
  log "Launching eval ${script} on GPU ${gpu}"
  GPU_ID="${gpu}" bash "./${script}" > "${LOG_DIR}/${log_name}" 2>&1
  log "Eval ${script} finished."
}

run_train_stage() {
  local script="$1"
  local log_name="$2"
  log "Launching training ${script}"
  bash "./${script}" > "${LOG_DIR}/${log_name}" 2>&1
  log "Training ${script} finished."
}

SFT_PID="${1:-2550}"
SFT_ADAPTER_DIR="./saves/llama31_8b/lora/hardonly_1000_sft"
DPO_ADAPTER_DIR="./saves/llama31_8b/lora/hardonly_1000_dpo_vanilla"
MDPO_ADAPTER_DIR="./saves/llama31_8b/lora/hardonly_1000_mdpo"

require_file "./run_eval_llama31_8b_base_4096.sh"
require_file "./run_eval_llama31_8b_sft_4096.sh"
require_file "./run_eval_llama31_8b_dpo_vanilla_4096.sh"
require_file "./run_eval_llama31_8b_mdpo_4096.sh"
require_file "./run_llama31_8b_hardonly_1000_dpo_vanilla_2gpu.sh"
require_file "./run_llama31_8b_hardonly_1000_mdpo_2gpu.sh"

wait_for_pid_exit "${SFT_PID}"
wait_for_adapter "${SFT_ADAPTER_DIR}"

run_parallel_eval_pair 0 "run_eval_llama31_8b_base_4096.sh" "llama31_8b_base_eval.log" 1 "run_eval_llama31_8b_sft_4096.sh" "llama31_8b_sft_eval.log"

run_train_stage "run_llama31_8b_hardonly_1000_dpo_vanilla_2gpu.sh" "llama31_8b_hardonly_1000_dpo_vanilla.log"
wait_for_adapter "${DPO_ADAPTER_DIR}"
run_single_eval 0 "run_eval_llama31_8b_dpo_vanilla_4096.sh" "llama31_8b_dpo_vanilla_eval.log"

run_train_stage "run_llama31_8b_hardonly_1000_mdpo_2gpu.sh" "llama31_8b_hardonly_1000_mdpo.log"
wait_for_adapter "${MDPO_ADAPTER_DIR}"
run_single_eval 0 "run_eval_llama31_8b_mdpo_4096.sh" "llama31_8b_mdpo_eval.log"

log "Full Llama-3.1-8B pipeline completed."
log "Reports expected under /root/autodl-tmp/robots-planning/data/pvr_eval_llama31_8b_hardonly_1000_test"
