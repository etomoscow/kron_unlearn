#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
mkdir -p logs/review_queue logs/review_muse_pilot .cache/triton .cache/torch_extensions .cache/hf .cache/xdg

GPU_CANDIDATES="${GPU_CANDIDATES:-0 3 2 1 4}"
MIN_FREE_MB="${MIN_FREE_MB:-60000}"
MAX_UTIL="${MAX_UTIL:-10}"
GPU_STABLE_CHECKS="${GPU_STABLE_CHECKS:-3}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-300}"
DEPENDENCY_WAIT_SECONDS="${DEPENDENCY_WAIT_SECONDS:-600}"
STEPS="${STEPS:-50}"
MODEL="${MODEL:-Llama-3.2-3B-Instruct}"
DATA_SPLIT="${DATA_SPLIT:-News}"
TRAINERS="${TRAINERS:-NPO SimNPO}"
RETAIN_LOGS_PATH="${RETAIN_LOGS_PATH:-saves/eval/muse_Llama-2-7b-hf_News_retrain/MUSE_EVAL.json}"
LOG="logs/review_queue/queue_muse_news_pilot_after_3b_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1

echo "[$(date -Is)] Queue J MUSE News pilot waiter start"
echo "[$(date -Is)] depends on Queue I 3B pilot summaries; candidates=${GPU_CANDIDATES}; need one GPU free>=${MIN_FREE_MB} util<=${MAX_UTIL} stable=${GPU_STABLE_CHECKS}"
echo "[$(date -Is)] wrapper log=${LOG}"

export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TRITON_CACHE_DIR="${ROOT}/.cache/triton"
export TORCH_EXTENSIONS_DIR="${ROOT}/.cache/torch_extensions"
export HF_HOME="${ROOT}/.cache/hf"
export HF_HUB_DISABLE_XET=1
export XDG_CACHE_HOME="${ROOT}/.cache/xdg"
export WANDB_DISABLED=true

scratch_3b="saves/unlearn/tofu_Llama-3.2-3B-Instruct_forget10_NPO_scratch_S50_seed0_review_3b_pilot/checkpoint-50/evals/TOFU_SUMMARY.json"
kforge_3b="saves/unlearn/tofu_Llama-3.2-3B-Instruct_forget10_NPO_kforge_layer15_s0p62_S50_seed0_review_3b_pilot/checkpoint-50/evals/TOFU_SUMMARY.json"
while [[ ! -s "${scratch_3b}" || ! -s "${kforge_3b}" ]]; do
  echo "[$(date -Is)] Waiting for 3B pilot dependency: scratch=$([[ -s "${scratch_3b}" ]] && echo done || echo missing) kforge=$([[ -s "${kforge_3b}" ]] && echo done || echo missing)"
  sleep "${DEPENDENCY_WAIT_SECONDS}"
done

echo "[$(date -Is)] 3B pilot dependency satisfied"

choose_gpu_pair() {
  local stable_pair="" stable_count=0
  while true; do
    local available=()
    for gpu in ${GPU_CANDIDATES}; do
      local row used total free util
      row="$(nvidia-smi --id="${gpu}" --query-gpu=memory.used,memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')" || continue
      IFS=',' read -r used total free util <<< "${row}"
      echo "[$(date -Is)] GPU ${gpu}: used=${used} total=${total} free=${free} util=${util} stable_pair=${stable_pair:-none} stable_count=${stable_count}/${GPU_STABLE_CHECKS}" >&2
      if [[ "${free}" -ge "${MIN_FREE_MB}" && "${util}" -le "${MAX_UTIL}" ]]; then
        available+=("${gpu}")
      fi
    done
    if [[ "${#available[@]}" -ge 2 ]]; then
      local pair="${available[0]},${available[1]}"
      if [[ "${stable_pair}" == "${pair}" ]]; then
        stable_count=$((stable_count+1))
      else
        stable_pair="${pair}"
        stable_count=1
      fi
      if [[ "${stable_count}" -ge "${GPU_STABLE_CHECKS}" ]]; then
        echo "${pair}"
        return 0
      fi
    else
      stable_pair=""
      stable_count=0
    fi
    sleep "${GPU_WAIT_SECONDS}"
  done
}

choose_gpu() {
  local stable_gpu="" stable_count=0
  while true; do
    local gpu
    for gpu in ${GPU_CANDIDATES}; do
      local row used total free util
      row="$(nvidia-smi --id="${gpu}" --query-gpu=memory.used,memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')" || continue
      IFS=',' read -r used total free util <<< "${row}"
      echo "[$(date -Is)] GPU ${gpu}: used=${used} total=${total} free=${free} util=${util} stable=${stable_gpu:-none} ${stable_count}/${GPU_STABLE_CHECKS}"
      if [[ "${free}" -ge "${MIN_FREE_MB}" && "${util}" -le "${MAX_UTIL}" ]]; then
        if [[ "${stable_gpu}" == "${gpu}" ]]; then
          stable_count=$((stable_count+1))
        else
          stable_gpu="${gpu}"
          stable_count=1
        fi
        if [[ "${stable_count}" -ge "${GPU_STABLE_CHECKS}" ]]; then
          echo "${gpu}"
          return 0
        fi
      fi
    done
    stable_gpu=""
    stable_count=0
    sleep "${GPU_WAIT_SECONDS}"
  done
}

GPU_ID="$(choose_gpu | tail -1)"
echo "[$(date -Is)] Selected GPU ${GPU_ID} for MUSE News pilot"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

wait_for_train_summary() {
  local summary="$1"
  local task="$2"
  while [[ ! -s "${summary}" ]]; do
    echo "[$(date -Is)] Waiting for summary: ${task} -> ${summary}"
    sleep 60
  done
}

eval_task() {
  local task="$1"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python src/eval.py --config-name=eval \
    experiment=eval/muse/default \
    data_split="${DATA_SPLIT}" \
    task_name="${task}" \
    model="${MODEL}" \
    model.model_args.attn_implementation=eager \
    model.model_args.torch_dtype=float32 \
    +model.model_args.use_cache=false \
    model.template_args.apply_chat_template=false \
    model.template_args.user_start_tag="Question: " \
    model.template_args.user_end_tag="\n\n" \
    model.template_args.asst_start_tag="Answer: " \
    model.template_args.asst_end_tag="\n\n" \
    +model.tokenizer_args.padding_side=left \
    model.model_args.pretrained_model_name_or_path="/home/d.moskovskiy/.cache/huggingface/models--open-unlearning--tofu_Llama-3.2-3B-Instruct_full/snapshots/24f31ca19f6966dcb6f6b29abc511cce71222d4a" \
    model.tokenizer_args.pretrained_model_name_or_path="/home/d.moskovskiy/.cache/huggingface/models--open-unlearning--tofu_Llama-3.2-3B-Instruct_full/snapshots/24f31ca19f6966dcb6f6b29abc511cce71222d4a" \
    retain_logs_path="${RETAIN_LOGS_PATH}" \
    2>&1 | tee "logs/review_muse_pilot/${task}_eval.log"
}

for trainer in ${TRAINERS}; do
  task="muse_${MODEL}_${DATA_SPLIT}_${trainer}_S${STEPS}_review_muse_pilot"
  summary="saves/eval/${task}/MUSE_SUMMARY.json"
  echo "[$(date -Is)] MUSE pilot trainer=${trainer} task=${task}"
  if [[ -s "${summary}" ]]; then
    echo "[$(date -Is)] summary exists; skipping ${summary}"
    continue
  fi
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/muse/default.yaml \
    model="${MODEL}" \
    model.model_args.attn_implementation=eager \
    model.model_args.torch_dtype=float32 \
    +model.model_args.use_cache=false \
    model.template_args.apply_chat_template=false \
    model.template_args.user_start_tag="Question: " \
    model.template_args.user_end_tag="\n\n" \
    model.template_args.asst_start_tag="Answer: " \
    model.template_args.asst_end_tag="\n\n" \
    +model.tokenizer_args.padding_side=left \
    model.model_args.pretrained_model_name_or_path="/home/d.moskovskiy/.cache/huggingface/models--open-unlearning--tofu_Llama-3.2-3B-Instruct_full/snapshots/24f31ca19f6966dcb6f6b29abc511cce71222d4a" \
    model.tokenizer_args.pretrained_model_name_or_path="/home/d.moskovskiy/.cache/huggingface/models--open-unlearning--tofu_Llama-3.2-3B-Instruct_full/snapshots/24f31ca19f6966dcb6f6b29abc511cce71222d4a" \
    data_split="${DATA_SPLIT}" \
    trainer="${trainer}" \
    task_name="${task}" \
    retain_logs_path="${RETAIN_LOGS_PATH}" \
    trainer.args.report_to=none \
    trainer.args.per_device_train_batch_size=1 \
    trainer.args.gradient_accumulation_steps=16 \
    trainer.args.optim=adamw_torch \
    +trainer.args.max_steps="${STEPS}" \
    trainer.args.num_train_epochs=1 \
    trainer.args.ddp_find_unused_parameters=true \
    trainer.args.gradient_checkpointing=true \
    2>&1 | tee "logs/review_muse_pilot/${task}.log"
  eval_task "${task}"
  test -s "${summary}"
done

echo "[$(date -Is)] Queue J MUSE News pilot complete"
