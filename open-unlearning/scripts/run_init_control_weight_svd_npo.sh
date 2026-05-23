#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-1}"
TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
BASE_MODEL="${BASE_MODEL:-open-unlearning/tofu_Llama-3.2-1B-Instruct_full}"
TEMPLATE="${TEMPLATE:-saves/unlearn/KFORGE_TOFU_forget10_R2_M1_B32_S0p45_kron_retain_cfix_retune_v2_lam0p01}"
FORGET_SPLIT="${FORGET_SPLIT:-forget10}"
RETAIN_SPLIT="${RETAIN_SPLIT:-retain90}"
HOLDOUT_SPLIT="${HOLDOUT_SPLIT:-holdout10}"
SEEDS="${SEEDS:-0 1 2}"
STEP_BUDGETS="${STEP_BUDGETS:-50 100}"
MODEL_ID="${MODEL_ID:-Llama-3.2-1B-Instruct}"
RUN_TAG="${RUN_TAG:-initctrl}"

mkdir -p .cache/triton .cache/torch_extensions .cache/hf .cache/xdg logs
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TRITON_CACHE_DIR="${PWD}/.cache/triton"
export TORCH_EXTENSIONS_DIR="${PWD}/.cache/torch_extensions"
export HF_HOME="${PWD}/.cache/hf"
export HF_HUB_DISABLE_XET=1
export XDG_CACHE_HOME="${PWD}/.cache/xdg"
export WANDB_DISABLED=true

MANIFEST="logs/init_control_weight_svd_npo_${FORGET_SPLIT}_${TS}.tsv"
printf "timestamp\tseed\tsteps\tstatus\n" > "${MANIFEST}"

for SEED in ${SEEDS}; do
  OUT="saves/unlearn/INITCTRL_TOFU_${FORGET_SPLIT}_weight_svd_matched_kforge_s045_seed${SEED}"
  if [[ ! -s "${OUT}/config.json" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python scripts/make_matched_init_checkpoint.py \
      --base-model "${BASE_MODEL}" \
      --template-model "${TEMPLATE}" \
      --output-dir "${OUT}" \
      --mode weight_svd \
      --rank 2 \
      --seed "${SEED}" \
      --device cuda \
      --dtype float32 \
      2>&1 | tee "logs/initctrl_make_weight_svd_seed${SEED}_${TS}.log"
  fi

  for STEPS in ${STEP_BUDGETS}; do
    TASK="tofu_${MODEL_ID}_${FORGET_SPLIT}_NPO_weight_svd_S${STEPS}_seed${SEED}_${RUN_TAG}"
    SUMMARY="saves/eval/${TASK}_EVAL_FP32/TOFU_SUMMARY.json"
    if [[ -s "${SUMMARY}" ]]; then
      printf "%s\t%s\t%s\tskipped_existing\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${SEED}" "${STEPS}" >> "${MANIFEST}"
      continue
    fi
    printf "%s\t%s\t%s\tstarted\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${SEED}" "${STEPS}" >> "${MANIFEST}"

    CUDA_VISIBLE_DEVICES="${GPU_ID}" python src/train.py --config-name=unlearn.yaml \
      experiment=unlearn/tofu/default.yaml \
      trainer=NPO \
      task_name="${TASK}" \
      model="${MODEL_ID}" \
      forget_split="${FORGET_SPLIT}" \
      holdout_split="${HOLDOUT_SPLIT}" \
      retain_split="${RETAIN_SPLIT}" \
      model.model_args.pretrained_model_name_or_path="${OUT}" \
      model.model_args.attn_implementation=sdpa \
      model.model_args.torch_dtype=float32 \
      model.tokenizer_args.pretrained_model_name_or_path="${OUT}" \
      retain_logs_path="saves/eval/tofu_${MODEL_ID}_${RETAIN_SPLIT}/TOFU_EVAL.json" \
      trainer.method_args.ref_model_path="${BASE_MODEL}" \
      trainer.args.report_to=none \
      trainer.args.per_device_train_batch_size=4 \
      trainer.args.gradient_accumulation_steps=8 \
      +trainer.args.max_steps="${STEPS}" \
      trainer.args.num_train_epochs=1 \
      trainer.args.optim=adamw_torch \
      trainer.args.seed="${SEED}" \
      trainer.args.ddp_find_unused_parameters=true \
      trainer.args.gradient_checkpointing=true \
      2>&1 | tee "logs/${TASK}.log"

    CUDA_VISIBLE_DEVICES="${GPU_ID}" python src/eval.py --config-name=eval experiment=eval/tofu/default \
      forget_split="${FORGET_SPLIT}" \
      holdout_split="${HOLDOUT_SPLIT}" \
      model="${MODEL_ID}" \
      model.model_args.pretrained_model_name_or_path="saves/unlearn/${TASK}" \
      model.model_args.attn_implementation=sdpa \
      model.model_args.torch_dtype=float32 \
      model.tokenizer_args.pretrained_model_name_or_path="saves/unlearn/${TASK}" \
      retain_logs_path="saves/eval/tofu_${MODEL_ID}_${RETAIN_SPLIT}/TOFU_EVAL.json" \
      task_name="${TASK}_EVAL_FP32" \
      hydra.run.dir="outputs/${TASK}_EVAL_FP32" \
      2>&1 | tee "logs/${TASK}_EVAL_FP32.log"

    printf "%s\t%s\t%s\tok\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${SEED}" "${STEPS}" >> "${MANIFEST}"
  done
done
