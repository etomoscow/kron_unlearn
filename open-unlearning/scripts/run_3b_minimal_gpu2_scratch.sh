#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-2}"
MODEL_ID="${MODEL_ID:-Llama-3.2-3B-Instruct}"
BASE_MODEL="${BASE_MODEL:-open-unlearning/tofu_${MODEL_ID}_full}"
FORGET_SPLIT="${FORGET_SPLIT:-forget10}"
HOLDOUT_SPLIT="${HOLDOUT_SPLIT:-holdout10}"
RETAIN_SPLIT="${RETAIN_SPLIT:-retain90}"
RETAIN_LOGS="${RETAIN_LOGS:-saves/eval/tofu_${MODEL_ID}_${RETAIN_SPLIT}/TOFU_EVAL.json}"

wait_for_gpu() {
  while true; do
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU_ID}")
    if [[ "${used}" -lt 5000 ]]; then
      break
    fi
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] waiting for GPU ${GPU_ID}; used=${used} MiB"
    sleep 180
  done
}

mkdir -p .cache/triton .cache/torch_extensions .cache/hf .cache/xdg logs
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TRITON_CACHE_DIR="${PWD}/.cache/triton"
export TORCH_EXTENSIONS_DIR="${PWD}/.cache/torch_extensions"
export HF_HOME="${PWD}/.cache/hf"
export HF_HUB_DISABLE_XET=1
export XDG_CACHE_HOME="${PWD}/.cache/xdg"
export WANDB_DISABLED=true

if [[ ! -s "${RETAIN_LOGS}" ]]; then
  echo "Missing retain logs: ${RETAIN_LOGS}" >&2
  exit 2
fi

wait_for_gpu

MODEL_ID="${MODEL_ID}" \
BASE_MODEL="${BASE_MODEL}" \
MODEL_TORCH_DTYPE=float32 \
FORGET_SPLIT="${FORGET_SPLIT}" \
HOLDOUT_SPLIT="${HOLDOUT_SPLIT}" \
RETAIN_SPLIT="${RETAIN_SPLIT}" \
RETAIN_LOGS="${RETAIN_LOGS}" \
TRAINERS=NPO \
STEP_BUDGETS="50 100" \
SEEDS="0 1 2" \
INIT_MODES=scratch \
RUN_TAG="3b_minimal_v2" \
GPU_ID="${GPU_ID}" \
  bash scripts/kforge_week2_init_experiment.sh

echo "3B scratch minimal queue complete"
