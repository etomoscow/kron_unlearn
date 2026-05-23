#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-2}"
FORGET_SPLIT="${FORGET_SPLIT:-forget10}"
RETAIN_SPLIT="${RETAIN_SPLIT:-retain90}"
MODEL_ID="${MODEL_ID:-Llama-3.2-1B-Instruct}"
BASE_MODEL="${BASE_MODEL:-open-unlearning/tofu_${MODEL_ID}_full}"
RETAIN_LOGS="${RETAIN_LOGS:-saves/eval/tofu_${MODEL_ID}_${RETAIN_SPLIT}/TOFU_EVAL.json}"
MODEL_TORCH_DTYPE="${MODEL_TORCH_DTYPE:-float32}"
RANK="${RANK:-4}"
MAX_TARGET_MODULES="${MAX_TARGET_MODULES:-1}"
MAX_CALIBRATION_BATCHES="${MAX_CALIBRATION_BATCHES:-2}"
FACTOR_MODE="${FACTOR_MODE:-kron}"
USE_RETAIN_FISHER="${USE_RETAIN_FISHER:-true}"
STRENGTHS="${STRENGTHS:-0.001 0.003 0.01 0.03}"
TARGET_MODULES_REGEX="${TARGET_MODULES_REGEX:-.*mlp\\.down_proj$}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

mkdir -p .cache/triton .cache/torch_extensions .cache/hf .cache/xdg logs

for STRENGTH in ${STRENGTHS}; do
  STRENGTH_TAG="${STRENGTH//./p}"
  RETAIN_TAG="retain"
  if [[ "${USE_RETAIN_FISHER}" != "true" ]]; then
    RETAIN_TAG="forgetonly"
  fi
  RUN_NAME="KFORGE_TOFU_${FORGET_SPLIT}_R${RANK}_M${MAX_TARGET_MODULES}_B${MAX_CALIBRATION_BATCHES}_S${STRENGTH_TAG}_${FACTOR_MODE}_${RETAIN_TAG}${RUN_SUFFIX}"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  TRITON_CACHE_DIR="${PWD}/.cache/triton" \
  TORCH_EXTENSIONS_DIR="${PWD}/.cache/torch_extensions" \
    HF_HOME="${PWD}/.cache/hf" \
    HF_HUB_DISABLE_XET=1 \
    XDG_CACHE_HOME="${PWD}/.cache/xdg" \
  WANDB_DISABLED=true \
  python src/train.py --config-name=unlearn experiment=unlearn/tofu/kforge \
    model="${MODEL_ID}" \
    forget_split="${FORGET_SPLIT}" retain_split="${RETAIN_SPLIT}" \
    model.model_args.pretrained_model_name_or_path="${BASE_MODEL}" \
    model.model_args.attn_implementation=sdpa \
    model.model_args.torch_dtype="${MODEL_TORCH_DTYPE}" \
    model.tokenizer_args.pretrained_model_name_or_path="${BASE_MODEL}" \
    retain_logs_path="${RETAIN_LOGS}" \
    trainer.args.report_to=none trainer.args.do_eval=false \
    trainer.method_args.max_target_modules="${MAX_TARGET_MODULES}" \
    trainer.method_args.max_calibration_batches="${MAX_CALIBRATION_BATCHES}" \
    trainer.method_args.target_modules_regex="${TARGET_MODULES_REGEX}" \
    trainer.method_args.rank="${RANK}" \
    trainer.method_args.strength="${STRENGTH}" \
    trainer.method_args.factor_mode="${FACTOR_MODE}" \
    trainer.method_args.use_retain_fisher="${USE_RETAIN_FISHER}" \
    ${EXTRA_TRAIN_ARGS} \
    task_name="${RUN_NAME}" hydra.run.dir="outputs/${RUN_NAME}" \
    2>&1 | tee "logs/${RUN_NAME}.log"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  TRITON_CACHE_DIR="${PWD}/.cache/triton" \
  TORCH_EXTENSIONS_DIR="${PWD}/.cache/torch_extensions" \
    HF_HOME="${PWD}/.cache/hf" \
    HF_HUB_DISABLE_XET=1 \
    XDG_CACHE_HOME="${PWD}/.cache/xdg" \
  WANDB_DISABLED=true \
  python src/eval.py --config-name=eval experiment=eval/tofu/default \
    model="${MODEL_ID}" \
    model.model_args.pretrained_model_name_or_path="saves/unlearn/${RUN_NAME}" \
    model.model_args.attn_implementation=sdpa \
    model.model_args.torch_dtype=float32 \
    model.tokenizer_args.pretrained_model_name_or_path="saves/unlearn/${RUN_NAME}" \
    retain_logs_path="${RETAIN_LOGS}" \
    task_name="${RUN_NAME}_EVAL_FP32" hydra.run.dir="outputs/${RUN_NAME}_EVAL_FP32" \
    2>&1 | tee "logs/${RUN_NAME}_EVAL_FP32.log"
done
