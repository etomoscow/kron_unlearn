#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
mkdir -p logs/review_queue logs/review_layer7_frontier logs/review_budget_sweep logs/review_robustness
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="${ROOT}/.cache/triton"
mkdir -p "${TRITON_CACHE_DIR}"

MODEL_ID="Llama-3.2-1B-Instruct"
FORGET_SPLIT="forget10"
HOLDOUT_SPLIT="holdout10"
RETAIN_SPLIT="retain90"
RANK="2"
LAMBDA_TRADEOFF="0.01"
MAX_CALIBRATION_BATCHES="32"

slug_to_strength() {
  case "$1" in
    045) echo "0.45" ;;
    0p*) echo "${1/0p/0.}" ;;
    *) echo "$1" ;;
  esac
}

wait_for_gpu() {
  local gpu="$1"
  local min_free_mb="${2:-45000}"
  local max_util="${3:-35}"
  local stable_needed="${4:-3}"
  local stable=0
  echo "[$(date -Is)] Waiting for GPU ${gpu}: free>=${min_free_mb} MiB and util<=${max_util}% for ${stable_needed} consecutive checks"
  while true; do
    local row free util used total
    row="$(nvidia-smi --id="${gpu}" --query-gpu=memory.used,memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    IFS=',' read -r used total free util <<< "${row}"
    echo "[$(date -Is)] GPU ${gpu}: used=${used} total=${total} free=${free} util=${util} stable=${stable}/${stable_needed}"
    if [[ "${free}" -ge "${min_free_mb}" && "${util}" -le "${max_util}" ]]; then
      stable=$((stable+1))
      if [[ "${stable}" -ge "${stable_needed}" ]]; then
        echo "[$(date -Is)] GPU ${gpu} accepted"
        return 0
      fi
    else
      stable=0
    fi
    sleep "${GPU_WAIT_SECONDS:-300}"
  done
}

maybe_wait_for_gpu() {
  local gpu="$1"
  local min_free="${2:-45000}"
  local max_util="${3:-35}"
  if [[ "${WAIT_FOR_IDLE:-0}" == "1" ]]; then
    wait_for_gpu "${gpu}" "${min_free}" "${max_util}" "${GPU_STABLE_CHECKS:-3}"
  else
    echo "[$(date -Is)] WAIT_FOR_IDLE=0; starting on GPU ${gpu} without idle wait"
  fi
  export CUDA_VISIBLE_DEVICES="${gpu}"
}

ensure_kforge_init() {
  local layer="$1" s_slug="$2" out_tag="$3" log_dir="$4"
  local strength; strength="$(slug_to_strength "${s_slug}")"
  local module_regex="model\\.layers\\.${layer}\\.mlp\\.down_proj"
  local run="KFORGE_TOFU_forget10_R2_layer${layer}_down_S${s_slug}_v2lam0p01_${out_tag}"
  local log="${log_dir}/${run}.log"
  echo "[$(date -Is)] ensure_kforge_init layer=${layer} strength=${strength} run=${run}"
  if [[ -f "saves/unlearn/${run}/model.safetensors" ]]; then
    echo "[$(date -Is)] K-FORGE exists: saves/unlearn/${run}"
    return 0
  fi
  python src/train.py --config-name=unlearn \
    experiment=unlearn/tofu/kforge \
    trainer=KFORGE \
    model="${MODEL_ID}" \
    forget_split="${FORGET_SPLIT}" \
    holdout_split="${HOLDOUT_SPLIT}" \
    retain_split="${RETAIN_SPLIT}" \
    model.model_args.pretrained_model_name_or_path="open-unlearning/tofu_${MODEL_ID}_full" \
    model.tokenizer_args.pretrained_model_name_or_path="open-unlearning/tofu_${MODEL_ID}_full" \
    model.model_args.attn_implementation=sdpa \
    model.model_args.torch_dtype=float32 \
    paths.output_dir="saves/unlearn/${run}" \
    trainer.args.do_eval=false \
    trainer.args.eval_on_start=false \
    trainer.args.per_device_train_batch_size=1 \
    trainer.method_args.edit_variant=wiener_v2 \
    trainer.method_args.rank="${RANK}" \
    trainer.method_args.strength="${strength}" \
    trainer.method_args.lambda_tradeoff="${LAMBDA_TRADEOFF}" \
    trainer.method_args.max_calibration_batches="${MAX_CALIBRATION_BATCHES}" \
    trainer.method_args.max_target_modules=1 \
    trainer.method_args.target_modules_regex="${module_regex}" \
    retain_logs_path="saves/eval/tofu_${MODEL_ID}_${RETAIN_SPLIT}/TOFU_EVAL.json" \
    2>&1 | tee "${log}"
}

run_kforge_npo() {
  local layer="$1" s_slug="$2" steps="$3" seed="$4" out_tag="$5" log_dir="$6"
  local strength; strength="$(slug_to_strength "${s_slug}")"
  local kforge_run="KFORGE_TOFU_forget10_R2_layer${layer}_down_S${s_slug}_v2lam0p01_${out_tag}"
  local npo_run="tofu_${MODEL_ID}_${FORGET_SPLIT}_NPO_kforge_layer${layer}_down_s${s_slug}_S${steps}_seed${seed}_${out_tag}"
  local log="${log_dir}/${npo_run}.log"
  local summary="saves/unlearn/${npo_run}/checkpoint-${steps}/evals/TOFU_SUMMARY.json"
  echo "[$(date -Is)] run_kforge_npo layer=${layer} strength=${strength} steps=${steps} seed=${seed} run=${npo_run}"
  if [[ -f "${summary}" ]]; then
    echo "[$(date -Is)] summary exists; skipping: ${summary}"
    return 0
  fi
  python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default.yaml \
    trainer=NPO \
    task_name="${npo_run}" \
    model="${MODEL_ID}" \
    forget_split="${FORGET_SPLIT}" \
    holdout_split="${HOLDOUT_SPLIT}" \
    retain_split="${RETAIN_SPLIT}" \
    model.model_args.pretrained_model_name_or_path="saves/unlearn/${kforge_run}" \
    model.model_args.attn_implementation=sdpa \
    model.model_args.torch_dtype=float32 \
    model.tokenizer_args.pretrained_model_name_or_path="saves/unlearn/${kforge_run}" \
    retain_logs_path="saves/eval/tofu_${MODEL_ID}_${RETAIN_SPLIT}/TOFU_EVAL.json" \
    trainer.method_args.ref_model_path="open-unlearning/tofu_${MODEL_ID}_full" \
    trainer.args.report_to=none \
    trainer.args.per_device_train_batch_size=4 \
    trainer.args.gradient_accumulation_steps=8 \
    +trainer.args.max_steps="${steps}" \
    trainer.args.num_train_epochs=1 \
    trainer.args.optim=adamw_torch \
    trainer.args.seed="${seed}" \
    trainer.args.ddp_find_unused_parameters=true \
    trainer.args.gradient_checkpointing=true \
    2>&1 | tee "${log}"
  test -f "${summary}"
}

run_relearn() {
  local label="$1" ckpt="$2" seed="$3" out_tag="$4"
  local task="tofu_${MODEL_ID}_${FORGET_SPLIT}_${label}_seed${seed}_relearn_forget10_e1_${out_tag}"
  local log="logs/review_robustness/${task}.log"
  local summary="saves/finetune/${task}/checkpoint-12/evals/TOFU_SUMMARY.json"
  echo "[$(date -Is)] run_relearn label=${label} seed=${seed} ckpt=${ckpt} task=${task}"
  if [[ ! -f "${ckpt}/model.safetensors" ]]; then
    echo "[$(date -Is)] Missing checkpoint; skip: ${ckpt}/model.safetensors" >&2
    return 2
  fi
  if [[ -f "${summary}" ]]; then
    echo "[$(date -Is)] relearn summary exists; skipping: ${summary}"
    return 0
  fi
  python src/train.py \
    experiment=finetune/tofu/default.yaml \
    task_name="${task}" \
    model="${MODEL_ID}" \
    model.model_args.pretrained_model_name_or_path="${ckpt}" \
    model.tokenizer_args.pretrained_model_name_or_path="${ckpt}" \
    model.model_args.attn_implementation=sdpa \
    model.model_args.torch_dtype=float32 \
    data/datasets@data.train=TOFU_QA_forget \
    data.train.TOFU_QA_forget.args.hf_args.name="${FORGET_SPLIT}" \
    forget_split="${FORGET_SPLIT}" \
    holdout_split="${HOLDOUT_SPLIT}" \
    retain_logs_path="saves/eval/tofu_${MODEL_ID}_${RETAIN_SPLIT}/TOFU_EVAL.json" \
    trainer.args.report_to=none \
    trainer.args.per_device_train_batch_size=4 \
    trainer.args.gradient_accumulation_steps=8 \
    trainer.args.num_train_epochs=1 \
    trainer.args.optim=adamw_torch \
    trainer.args.seed="${seed}" \
    trainer.args.ddp_find_unused_parameters=true \
    trainer.args.gradient_checkpointing=true \
    2>&1 | tee "${log}"
  test -f "${summary}"
}

resolve_existing_npo_ckpt() {
  local layer="$1" s_slug="$2" steps="$3" seed="$4"
  local patterns=(
    "saves/unlearn/tofu_${MODEL_ID}_${FORGET_SPLIT}_NPO_kforge_layer${layer}_down_s${s_slug}_S${steps}_seed${seed}_review_interp"
    "saves/unlearn/tofu_${MODEL_ID}_${FORGET_SPLIT}_NPO_kforge_layer${layer}_down_s${s_slug}_S${steps}_seed${seed}_review_reps"
    "saves/unlearn/tofu_${MODEL_ID}_${FORGET_SPLIT}_NPO_kforge_layer${layer}_down_s${s_slug}_S${steps}_seed${seed}_review_ablation"
    "saves/unlearn/tofu_${MODEL_ID}_${FORGET_SPLIT}_NPO_kforge_layer${layer}_down_s${s_slug}_S${steps}_seed${seed}_review_layer7_frontier"
    "saves/unlearn/tofu_${MODEL_ID}_${FORGET_SPLIT}_NPO_kforge_layer${layer}_down_s${s_slug}_S${steps}_seed${seed}_review_budget_sweep"
  )
  local p
  for p in "${patterns[@]}"; do
    if [[ -f "${p}/model.safetensors" ]]; then
      echo "${p}"
      return 0
    fi
  done
  return 1
}
