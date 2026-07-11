#!/usr/bin/env bash
set -euo pipefail

TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
GPU_ID="${GPU_ID:-2}"
MODEL_ID="${MODEL_ID:-Llama-3.2-1B-Instruct}"
BASE_MODEL="${BASE_MODEL:-open-unlearning/tofu_${MODEL_ID}_full}"
KFORGE_INIT_MODEL="${KFORGE_INIT_MODEL:-saves/unlearn/KFORGE_TOFU_forget10_R2_M1_B2_S0p00305_kron_retain_stage3down}"
KFORGE_INIT_TAG="${KFORGE_INIT_TAG:-kforge}"
MODEL_TORCH_DTYPE="${MODEL_TORCH_DTYPE:-float32}"
FORGET_SPLIT="${FORGET_SPLIT:-forget10}"
HOLDOUT_SPLIT="${HOLDOUT_SPLIT:-holdout10}"
RETAIN_SPLIT="${RETAIN_SPLIT:-retain90}"
RETAIN_LOGS="${RETAIN_LOGS:-saves/eval/tofu_${MODEL_ID}_${RETAIN_SPLIT}/TOFU_EVAL.json}"
TRAINERS="${TRAINERS:-NPO SimNPO}"
STEP_BUDGETS="${STEP_BUDGETS:-50 100 250 500 1000}"
SEEDS="${SEEDS:-0 1 2}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
OPTIM="${OPTIM:-adamw_torch}"
SKIP_COMPLETED="${SKIP_COMPLETED:-true}"
RUN_TAG="${RUN_TAG:-week2}"
INIT_MODES="${INIT_MODES:-scratch kforge}"

mkdir -p .cache/triton .cache/torch_extensions .cache/hf .cache/xdg logs
MANIFEST="logs/kforge_week2_init_experiment_${TS}.tsv"
printf "timestamp\ttrainer\tinit\tsteps\tseed\tstatus\n" > "${MANIFEST}"

valid_summary() {
  python - "$1" <<'PY'
import json, math, numbers, sys
with open(sys.argv[1]) as handle:
    data = json.load(handle)
for key in ("forget_Q_A_Prob", "model_utility", "extraction_strength", "forget_Q_A_ROUGE"):
    value = data[key]
    if not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise ValueError(f"invalid {key}: {value!r}")
PY
}

run_one() {
  local TRAINER="$1"
  local INIT_TAG="$2"
  local MODEL_PATH="$3"
  local STEPS="$4"
  local SEED="$5"
  local TASK_NAME="tofu_${MODEL_ID}_${FORGET_SPLIT}_${TRAINER}_${INIT_TAG}_S${STEPS}_seed${SEED}_${RUN_TAG}"
  local SUMMARY_PATH="saves/eval/${TASK_NAME}_EVAL_FP32/TOFU_SUMMARY.json"

  if [[ "${SKIP_COMPLETED}" == "true" && -s "${SUMMARY_PATH}" ]] && valid_summary "${SUMMARY_PATH}"; then
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${TRAINER}" "${INIT_TAG}" "${STEPS}" "${SEED}" skipped_existing >> "${MANIFEST}"
    return
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${TRAINER}" "${INIT_TAG}" "${STEPS}" "${SEED}" started >> "${MANIFEST}"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  TRITON_CACHE_DIR="${PWD}/.cache/triton" \
  TORCH_EXTENSIONS_DIR="${PWD}/.cache/torch_extensions" \
  HF_HOME="${PWD}/.cache/hf" \
  HF_HUB_DISABLE_XET=1 \
  XDG_CACHE_HOME="${PWD}/.cache/xdg" \
  WANDB_DISABLED=true \
  python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default.yaml \
    trainer="${TRAINER}" \
    task_name="${TASK_NAME}" \
    model="${MODEL_ID}" \
    forget_split="${FORGET_SPLIT}" \
    holdout_split="${HOLDOUT_SPLIT}" \
    retain_split="${RETAIN_SPLIT}" \
    model.model_args.pretrained_model_name_or_path="${MODEL_PATH}" \
    model.model_args.attn_implementation=sdpa \
    model.model_args.torch_dtype="${MODEL_TORCH_DTYPE}" \
    model.tokenizer_args.pretrained_model_name_or_path="${MODEL_PATH}" \
    retain_logs_path="${RETAIN_LOGS}" \
    trainer.method_args.ref_model_path="${BASE_MODEL}" \
    trainer.args.report_to=none \
    trainer.args.do_eval=false \
    trainer.args.eval_on_start=false \
    trainer.args.eval_strategy=no \
    trainer.args.per_device_train_batch_size="${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    trainer.args.gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
    +trainer.args.max_steps="${STEPS}" \
    trainer.args.num_train_epochs=1 \
    trainer.args.optim="${OPTIM}" \
    trainer.args.seed="${SEED}" \
    trainer.args.ddp_find_unused_parameters=true \
    trainer.args.gradient_checkpointing=true \
    2>&1 | tee "logs/${TASK_NAME}.log"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  TRITON_CACHE_DIR="${PWD}/.cache/triton" \
  TORCH_EXTENSIONS_DIR="${PWD}/.cache/torch_extensions" \
  HF_HOME="${PWD}/.cache/hf" \
  HF_HUB_DISABLE_XET=1 \
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

  valid_summary "${SUMMARY_PATH}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${TRAINER}" "${INIT_TAG}" "${STEPS}" "${SEED}" ok >> "${MANIFEST}"
}

for TRAINER in ${TRAINERS}; do
  for STEPS in ${STEP_BUDGETS}; do
    for SEED in ${SEEDS}; do
      for INIT_MODE in ${INIT_MODES}; do
        case "${INIT_MODE}" in
          scratch)
            run_one "${TRAINER}" scratch "${BASE_MODEL}" "${STEPS}" "${SEED}"
            ;;
          kforge)
            run_one "${TRAINER}" "${KFORGE_INIT_TAG}" "${KFORGE_INIT_MODEL}" "${STEPS}" "${SEED}"
            ;;
          *)
            echo "Unknown INIT_MODE: ${INIT_MODE}" >&2
            exit 1
            ;;
        esac
      done
    done
  done
done

echo "Init experiment manifest: ${MANIFEST}"
