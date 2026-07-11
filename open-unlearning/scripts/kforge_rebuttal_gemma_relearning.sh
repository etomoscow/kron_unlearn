#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

GPU_ID="${GPU_ID:?set GPU_ID}"
EPOCHS="${EPOCHS:?set EPOCHS}"
RUN_TAG="${RUN_TAG:-rebuttal_gemma_relearn_v1}"
LOG_DIR="logs/review_gemma_relearning"
MANIFEST="${LOG_DIR}/${RUN_TAG}_e${EPOCHS}.tsv"

mkdir -p "${LOG_DIR}" .cache/triton
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="${ROOT}/.cache/triton"
export WANDB_DISABLED=true

printf 'arm\tseed\tepochs\tstatus\tsummary\n' > "${MANIFEST}"

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

for seed in 0 1 2; do
  for arm in scratch kforge; do
    ckpt="saves/unlearn/tofu_gemma-3-1b-it_forget10_NPO_${arm}_S50_seed${seed}_rebuttal_gemma3_tuned080_v1"
    task="tofu_gemma-3-1b-it_forget10_NPO_${arm}_matched_S50_seed${seed}_relearn_e${EPOCHS}_${RUN_TAG}"
    out="saves/finetune/${task}"
    summary="${out}/checkpoint-$((13 * EPOCHS))/evals/TOFU_SUMMARY.json"
    if [[ -s "${summary}" ]] && valid_summary "${summary}"; then
      printf '%s\t%s\t%s\tskipped\t%s\n' "${arm}" "${seed}" "${EPOCHS}" "${summary}" >> "${MANIFEST}"
      continue
    fi

    test -s "${ckpt}/model.safetensors"
    printf '%s\t%s\t%s\tstarted\t-\n' "${arm}" "${seed}" "${EPOCHS}" >> "${MANIFEST}"
    python src/train.py \
      experiment=finetune/tofu/default.yaml \
      task_name="${task}" \
      model=gemma-3-1b-it \
      model.model_args.pretrained_model_name_or_path="${ckpt}" \
      model.tokenizer_args.pretrained_model_name_or_path="${ckpt}" \
      model.model_args.attn_implementation=sdpa \
      model.model_args.torch_dtype=float32 \
      data/datasets@data.train=TOFU_QA_forget \
      data.train.TOFU_QA_forget.args.hf_args.name=forget10 \
      forget_split=forget10 \
      holdout_split=holdout10 \
      retain_logs_path=saves/eval/tofu_gemma-3-1b-it_retain90_rebuttal_gemma3_1b_v1/TOFU_EVAL.json \
      trainer.args.report_to=none \
      trainer.args.per_device_train_batch_size=4 \
      trainer.args.gradient_accumulation_steps=8 \
      trainer.args.num_train_epochs="${EPOCHS}" \
      trainer.args.optim=adamw_torch \
      trainer.args.seed="${seed}" \
      trainer.args.ddp_find_unused_parameters=true \
      trainer.args.gradient_checkpointing=true \
      > "${LOG_DIR}/${task}.log" 2>&1

    test -s "${summary}"
    valid_summary "${summary}"
    printf '%s\t%s\t%s\tok\t%s\n' "${arm}" "${seed}" "${EPOCHS}" "${summary}" >> "${MANIFEST}"
  done
done

echo "[$(date -Is)] Gemma matched NPO relearning e${EPOCHS} complete"
