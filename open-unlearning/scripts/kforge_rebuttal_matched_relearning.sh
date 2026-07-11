#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

GPU_ID="${GPU_ID:?set GPU_ID}"
EPOCHS="${EPOCHS:?set EPOCHS}"
RUN_TAG="${RUN_TAG:-rebuttal_matched_relearn_v1}"
LOG_DIR="logs/review_matched_relearning"
MANIFEST="${LOG_DIR}/${RUN_TAG}_e${EPOCHS}.tsv"

mkdir -p "${LOG_DIR}" .cache/triton
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="${ROOT}/.cache/triton"
export WANDB_DISABLED=true

printf 'arm\tseed\tepochs\tstatus\tsummary\n' > "${MANIFEST}"

valid_summary() {
  python - "$1" <<'PY'
import json, sys
with open(sys.argv[1]) as handle:
    data = json.load(handle)
for key in ("forget_Q_A_Prob", "model_utility", "extraction_strength", "forget_Q_A_ROUGE"):
    data[key]
PY
}

source_checkpoint() {
  local arm="$1" seed="$2"
  if [[ "${arm}" == scratch ]]; then
    echo "saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget10_NPO_scratch_S100_seed${seed}_v2lam0p01_corr"
  else
    echo "saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget10_NPO_kforge_s06_S50_seed${seed}_v2lam0p01_corr"
  fi
}

latest_summary() {
  [[ -d "$1" ]] || return 0
  find "$1" -path '*/evals/TOFU_SUMMARY.json' -print 2>/dev/null | sort -V | tail -1
}

for seed in 0 1 2; do
  for arm in scratch kforge; do
    ckpt="$(source_checkpoint "${arm}" "${seed}")"
    task="tofu_Llama-3.2-1B-Instruct_forget10_NPO_${arm}_matched_seed${seed}_relearn_e${EPOCHS}_${RUN_TAG}"
    out="saves/finetune/${task}"
    summary="$(latest_summary "${out}")"
    if [[ -n "${summary}" ]] && valid_summary "${summary}"; then
      printf '%s\t%s\t%s\tskipped\t%s\n' "${arm}" "${seed}" "${EPOCHS}" "${summary}" >> "${MANIFEST}"
      continue
    fi

    test -s "${ckpt}/model.safetensors"
    printf '%s\t%s\t%s\tstarted\t-\n' "${arm}" "${seed}" "${EPOCHS}" >> "${MANIFEST}"
    python src/train.py \
      experiment=finetune/tofu/default.yaml \
      task_name="${task}" \
      model=Llama-3.2-1B-Instruct \
      model.model_args.pretrained_model_name_or_path="${ckpt}" \
      model.tokenizer_args.pretrained_model_name_or_path="${ckpt}" \
      model.model_args.attn_implementation=sdpa \
      model.model_args.torch_dtype=float32 \
      data/datasets@data.train=TOFU_QA_forget \
      data.train.TOFU_QA_forget.args.hf_args.name=forget10 \
      forget_split=forget10 \
      holdout_split=holdout10 \
      retain_logs_path=saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json \
      trainer.args.report_to=none \
      trainer.args.per_device_train_batch_size=4 \
      trainer.args.gradient_accumulation_steps=8 \
      trainer.args.num_train_epochs="${EPOCHS}" \
      trainer.args.optim=adamw_torch \
      trainer.args.seed="${seed}" \
      trainer.args.ddp_find_unused_parameters=true \
      trainer.args.gradient_checkpointing=true \
      > "${LOG_DIR}/${task}.log" 2>&1

    summary="$(latest_summary "${out}")"
    test -n "${summary}"
    valid_summary "${summary}"
    printf '%s\t%s\t%s\tok\t%s\n' "${arm}" "${seed}" "${EPOCHS}" "${summary}" >> "${MANIFEST}"
  done
done

echo "[$(date -Is)] matched relearning e${EPOCHS} complete"
