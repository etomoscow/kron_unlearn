#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-1}"
MODEL_ID="${MODEL_ID:-Llama-3.2-3B-Instruct}"
BASE_MODEL="${BASE_MODEL:-open-unlearning/tofu_${MODEL_ID}_full}"
FORGET_SPLIT="${FORGET_SPLIT:-forget10}"
HOLDOUT_SPLIT="${HOLDOUT_SPLIT:-holdout10}"
RETAIN_SPLIT="${RETAIN_SPLIT:-retain90}"
RETAIN_LOGS="${RETAIN_LOGS:-saves/eval/tofu_${MODEL_ID}_${RETAIN_SPLIT}/TOFU_EVAL.json}"
INIT_RUN="${INIT_RUN:-KFORGE_TOFU_forget10_R2_M1_B32_S0p45_kron_retain_cfix_v2_lam0p01_3b_minimal}"
INIT_DIR="saves/unlearn/${INIT_RUN}"
TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"

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

if [[ ! -s "${INIT_DIR}/config.json" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python src/train.py --config-name=unlearn experiment=unlearn/tofu/kforge \
    model="${MODEL_ID}" \
    forget_split="${FORGET_SPLIT}" retain_split="${RETAIN_SPLIT}" \
    model.model_args.pretrained_model_name_or_path="${BASE_MODEL}" \
    model.model_args.attn_implementation=sdpa \
    model.model_args.torch_dtype=float32 \
    model.tokenizer_args.pretrained_model_name_or_path="${BASE_MODEL}" \
    retain_logs_path="${RETAIN_LOGS}" \
    trainer.args.report_to=none trainer.args.do_eval=false \
    trainer.method_args.edit_variant=wiener_v2 \
    trainer.method_args.lambda_tradeoff=0.01 \
    trainer.method_args.max_target_modules=1 \
    trainer.method_args.max_calibration_batches=32 \
    trainer.method_args.target_modules_regex='.*mlp\.down_proj$' \
    trainer.method_args.rank=2 \
    trainer.method_args.strength=0.45 \
    trainer.method_args.factor_mode=kron \
    trainer.method_args.use_retain_fisher=true \
    task_name="${INIT_RUN}" hydra.run.dir="outputs/${INIT_RUN}" \
    2>&1 | tee "logs/${INIT_RUN}.log"
fi

MODEL_ID="${MODEL_ID}" \
BASE_MODEL="${BASE_MODEL}" \
KFORGE_INIT_MODEL="${INIT_DIR}" \
KFORGE_INIT_TAG=kforge_s045 \
MODEL_TORCH_DTYPE=float32 \
FORGET_SPLIT="${FORGET_SPLIT}" \
HOLDOUT_SPLIT="${HOLDOUT_SPLIT}" \
RETAIN_SPLIT="${RETAIN_SPLIT}" \
RETAIN_LOGS="${RETAIN_LOGS}" \
TRAINERS=NPO \
STEP_BUDGETS="50 100" \
SEEDS="0 1 2" \
INIT_MODES=kforge \
RUN_TAG="3b_minimal_v2" \
GPU_ID="${GPU_ID}" \
  bash scripts/kforge_week2_init_experiment.sh

echo "3B K-FORGE minimal queue complete: ${TS}"
