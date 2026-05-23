#!/usr/bin/env bash
set -euo pipefail

TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
GPU_IDS="${GPU_IDS:-2}"
MODEL_ID="${MODEL_ID:-Llama-3.2-1B-Instruct}"
BASE_MODEL="${BASE_MODEL:-open-unlearning/tofu_${MODEL_ID}_full}"
FORGET_SPLIT="${FORGET_SPLIT:-forget10}"
HOLDOUT_SPLIT="${HOLDOUT_SPLIT:-holdout10}"
RETAIN_SPLIT="${RETAIN_SPLIT:-retain90}"
RETAIN_LOGS="${RETAIN_LOGS:-saves/eval/tofu_${MODEL_ID}_${RETAIN_SPLIT}/TOFU_EVAL.json}"
TRAINERS="${TRAINERS:-NPO SimNPO RMU GradDiff}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-10}"
WAIT_FOR_BCAL="${WAIT_FOR_BCAL:-true}"
OPTIM="${OPTIM:-adamw_torch}"

mkdir -p .cache/triton .cache/torch_extensions .cache/hf .cache/xdg logs
MANIFEST="logs/tofu_week1_baselines_${TS}.tsv"
printf "timestamp\ttrainer\tstatus\n" > "${MANIFEST}"

if [[ "${WAIT_FOR_BCAL}" == "true" ]]; then
  while tmux ls 2>/dev/null | grep -q 'kforge_bcal_gpu'; do
    sleep 300
  done
fi

export MASTER_PORT
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

for TRAINER in ${TRAINERS}; do
  TASK_NAME="tofu_${MODEL_ID}_${FORGET_SPLIT}_${TRAINER}_week1"
  printf "%s\t%s\t%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${TRAINER}" started >> "${MANIFEST}"

  TRAIN_CMD=(python src/train.py --config-name=unlearn.yaml)
  if [[ "${GPU_IDS}" == *,* ]]; then
    TRAIN_CMD=(accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port "${MASTER_PORT}" src/train.py --config-name=unlearn.yaml)
  fi

  CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  TRITON_CACHE_DIR="${PWD}/.cache/triton" \
  TORCH_EXTENSIONS_DIR="${PWD}/.cache/torch_extensions" \
  HF_HOME="${PWD}/.cache/hf" \
  XDG_CACHE_HOME="${PWD}/.cache/xdg" \
  WANDB_DISABLED=true \
  "${TRAIN_CMD[@]}" \
    experiment=unlearn/tofu/default.yaml \
    trainer="${TRAINER}" \
    task_name="${TASK_NAME}" \
    model="${MODEL_ID}" \
    forget_split="${FORGET_SPLIT}" \
    holdout_split="${HOLDOUT_SPLIT}" \
    retain_split="${RETAIN_SPLIT}" \
    model.model_args.pretrained_model_name_or_path="${BASE_MODEL}" \
    model.model_args.attn_implementation=sdpa \
    model.tokenizer_args.pretrained_model_name_or_path="${BASE_MODEL}" \
    retain_logs_path="${RETAIN_LOGS}" \
    trainer.args.report_to=none \
    trainer.args.per_device_train_batch_size="${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    trainer.args.gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
    trainer.args.num_train_epochs="${NUM_TRAIN_EPOCHS}" \
    trainer.args.optim="${OPTIM}" \
    trainer.args.ddp_find_unused_parameters=true \
    trainer.args.gradient_checkpointing=true \
    2>&1 | tee "logs/${TASK_NAME}.log"

  CUDA_VISIBLE_DEVICES="${GPU_IDS%%,*}" \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  TRITON_CACHE_DIR="${PWD}/.cache/triton" \
  TORCH_EXTENSIONS_DIR="${PWD}/.cache/torch_extensions" \
  HF_HOME="${PWD}/.cache/hf" \
  XDG_CACHE_HOME="${PWD}/.cache/xdg" \
  WANDB_DISABLED=true \
  python src/eval.py --config-name=eval experiment=eval/tofu/default \
    forget_split="${FORGET_SPLIT}" \
    holdout_split="${HOLDOUT_SPLIT}" \
    model="${MODEL_ID}" \
    model.model_args.pretrained_model_name_or_path="saves/unlearn/${TASK_NAME}" \
    model.model_args.attn_implementation=sdpa \
    model.model_args.torch_dtype=float32 \
    model.tokenizer_args.pretrained_model_name_or_path="saves/unlearn/${TASK_NAME}" \
    retain_logs_path="${RETAIN_LOGS}" \
    task_name="${TASK_NAME}_EVAL_FP32" \
    hydra.run.dir="outputs/${TASK_NAME}_EVAL_FP32" \
    2>&1 | tee "logs/${TASK_NAME}_EVAL_FP32.log"

  printf "%s\t%s\t%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${TRAINER}" ok >> "${MANIFEST}"
done

echo "Baseline manifest: ${MANIFEST}"
