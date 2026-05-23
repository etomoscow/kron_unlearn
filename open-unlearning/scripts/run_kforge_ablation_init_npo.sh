#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-1}"
ABLATION="${ABLATION:-diagonal}"
TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
MODEL_ID="${MODEL_ID:-Llama-3.2-1B-Instruct}"
BASE_MODEL="${BASE_MODEL:-open-unlearning/tofu_${MODEL_ID}_full}"
DOWNSTREAM_TRAINER="${DOWNSTREAM_TRAINER:-NPO}"
FORGET_SPLIT="${FORGET_SPLIT:-forget10}"
RETAIN_SPLIT="${RETAIN_SPLIT:-retain90}"
HOLDOUT_SPLIT="${HOLDOUT_SPLIT:-holdout10}"
RETAIN_LOGS="${RETAIN_LOGS:-saves/eval/tofu_${MODEL_ID}_${RETAIN_SPLIT}/TOFU_EVAL.json}"
SEEDS="${SEEDS:-0 1 2}"
STEP_BUDGETS="${STEP_BUDGETS:-50 100}"
RANK="${RANK:-2}"
MAX_TARGET_MODULES="${MAX_TARGET_MODULES:-1}"
MAX_CALIBRATION_BATCHES="${MAX_CALIBRATION_BATCHES:-32}"
STRENGTH="${STRENGTH:-0.45}"
LAMBDA_TRADEOFF="${LAMBDA_TRADEOFF:-0.01}"
RUN_TAG="${RUN_TAG:-initctrl}"

case "${ABLATION}" in
  diagonal)
    FACTOR_MODE="diagonal"
    USE_RETAIN_FISHER="true"
    INIT_TAG="diagonal"
    ;;
  forget_only)
    FACTOR_MODE="kron"
    USE_RETAIN_FISHER="false"
    INIT_TAG="forget_only"
    ;;
  *)
    echo "Unknown ABLATION=${ABLATION}; expected diagonal or forget_only" >&2
    exit 2
    ;;
esac

mkdir -p .cache/triton .cache/torch_extensions .cache/hf .cache/xdg logs
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TRITON_CACHE_DIR="${PWD}/.cache/triton"
export TORCH_EXTENSIONS_DIR="${PWD}/.cache/torch_extensions"
export HF_HOME="${PWD}/.cache/hf"
export HF_HUB_DISABLE_XET=1
export XDG_CACHE_HOME="${PWD}/.cache/xdg"
export WANDB_DISABLED=true

STRENGTH_TAG="${STRENGTH//./p}"
LAMBDA_TAG="${LAMBDA_TRADEOFF//./p}"
RETAIN_TAG="retain"
if [[ "${USE_RETAIN_FISHER}" != "true" ]]; then
  RETAIN_TAG="forgetonly"
fi
INIT_RUN="KFORGE_TOFU_${FORGET_SPLIT}_R${RANK}_M${MAX_TARGET_MODULES}_B${MAX_CALIBRATION_BATCHES}_S${STRENGTH_TAG}_${FACTOR_MODE}_${RETAIN_TAG}_v2_lam${LAMBDA_TAG}_${RUN_TAG}"
INIT_DIR="saves/unlearn/${INIT_RUN}"
MANIFEST="logs/kforge_ablation_init_${ABLATION}_${DOWNSTREAM_TRAINER}_${TS}.tsv"
printf "timestamp\tablation\tseed\tsteps\tstatus\n" > "${MANIFEST}"

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
    trainer.method_args.lambda_tradeoff="${LAMBDA_TRADEOFF}" \
    trainer.method_args.max_target_modules="${MAX_TARGET_MODULES}" \
    trainer.method_args.max_calibration_batches="${MAX_CALIBRATION_BATCHES}" \
    trainer.method_args.target_modules_regex='.*mlp\.down_proj$' \
    trainer.method_args.rank="${RANK}" \
    trainer.method_args.strength="${STRENGTH}" \
    trainer.method_args.factor_mode="${FACTOR_MODE}" \
    trainer.method_args.use_retain_fisher="${USE_RETAIN_FISHER}" \
    task_name="${INIT_RUN}" hydra.run.dir="outputs/${INIT_RUN}" \
    2>&1 | tee "logs/${INIT_RUN}.log"
fi

for SEED in ${SEEDS}; do
  for STEPS in ${STEP_BUDGETS}; do
    TASK="tofu_${MODEL_ID}_${FORGET_SPLIT}_${DOWNSTREAM_TRAINER}_${INIT_TAG}_S${STEPS}_seed${SEED}_${RUN_TAG}"
    SUMMARY="saves/eval/${TASK}_EVAL_FP32/TOFU_SUMMARY.json"
    if [[ -s "${SUMMARY}" ]]; then
      printf "%s\t%s\t%s\t%s\tskipped_existing\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${ABLATION}" "${SEED}" "${STEPS}" >> "${MANIFEST}"
      continue
    fi
    printf "%s\t%s\t%s\t%s\tstarted\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${ABLATION}" "${SEED}" "${STEPS}" >> "${MANIFEST}"

    CUDA_VISIBLE_DEVICES="${GPU_ID}" python src/train.py --config-name=unlearn.yaml \
      experiment=unlearn/tofu/default.yaml \
      trainer="${DOWNSTREAM_TRAINER}" \
      task_name="${TASK}" \
      model="${MODEL_ID}" \
      forget_split="${FORGET_SPLIT}" \
      holdout_split="${HOLDOUT_SPLIT}" \
      retain_split="${RETAIN_SPLIT}" \
      model.model_args.pretrained_model_name_or_path="${INIT_DIR}" \
      model.model_args.attn_implementation=sdpa \
      model.model_args.torch_dtype=float32 \
      model.tokenizer_args.pretrained_model_name_or_path="${INIT_DIR}" \
      retain_logs_path="${RETAIN_LOGS}" \
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
      retain_logs_path="${RETAIN_LOGS}" \
      task_name="${TASK}_EVAL_FP32" \
      hydra.run.dir="outputs/${TASK}_EVAL_FP32" \
      2>&1 | tee "logs/${TASK}_EVAL_FP32.log"

    printf "%s\t%s\t%s\t%s\tok\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${ABLATION}" "${SEED}" "${STEPS}" >> "${MANIFEST}"
  done
done
