#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

GPU_ID="${GPU_ID:?set GPU_ID}"
RUN_TAG="${RUN_TAG:-rebuttal_gemma_quant_v1}"
LOG_DIR="logs/review_gemma_quantization"
MANIFEST="${LOG_DIR}/${RUN_TAG}.tsv"

mkdir -p "${LOG_DIR}" .cache/triton
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="${ROOT}/.cache/quant_bnb_049:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="${ROOT}/.cache/triton"

printf 'bits\tmethod\tarm\tseed\tstatus\tsummary\n' > "${MANIFEST}"

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

for bits in 8 4; do
  for method in NPO SimNPO; do
    for seed in 0 1 2 3; do
      source_tag=rebuttal_gemma3_tuned080_v1
      [[ "${seed}" == 3 ]] && source_tag=rebuttal_gemma3_confirm_seed3_v1
      for arm in scratch kforge; do
        ckpt="saves/unlearn/tofu_gemma-3-1b-it_forget10_${method}_${arm}_S100_seed${seed}_${source_tag}"
        task="tofu_gemma-3-1b-it_forget10_${method}_${arm}_S100_seed${seed}_quant${bits}_${RUN_TAG}"
        summary="saves/eval/${task}/TOFU_SUMMARY.json"
        if [[ -s "${summary}" ]] && valid_summary "${summary}"; then
          printf '%s\t%s\t%s\t%s\tskipped\t%s\n' "${bits}" "${method}" "${arm}" "${seed}" "${summary}" >> "${MANIFEST}"
          continue
        fi

        test -s "${ckpt}/model.safetensors"
        if [[ "${bits}" == 8 ]]; then
          quant_args=(+model.model_args.quantization_config.load_in_8bit=true)
        else
          quant_args=(
            +model.model_args.quantization_config.load_in_4bit=true
            +model.model_args.quantization_config.bnb_4bit_quant_type=nf4
            +model.model_args.quantization_config.bnb_4bit_compute_dtype=bfloat16
            +model.model_args.quantization_config.bnb_4bit_use_double_quant=false
          )
        fi

        printf '%s\t%s\t%s\t%s\tstarted\t%s\n' "${bits}" "${method}" "${arm}" "${seed}" "${summary}" >> "${MANIFEST}"
        python src/eval.py \
          experiment=eval/tofu/default.yaml \
          task_name="${task}" \
          seed="${seed}" \
          model=gemma-3-1b-it \
          model.model_args.pretrained_model_name_or_path="${ckpt}" \
          model.tokenizer_args.pretrained_model_name_or_path="${ckpt}" \
          model.model_args.attn_implementation=sdpa \
          model.model_args.torch_dtype=bfloat16 \
          "${quant_args[@]}" \
          forget_split=forget10 \
          holdout_split=holdout10 \
          retain_logs_path=saves/eval/tofu_gemma-3-1b-it_retain90_rebuttal_gemma3_1b_v1/TOFU_EVAL.json \
          > "${LOG_DIR}/${task}.log" 2>&1
        valid_summary "${summary}"
        printf '%s\t%s\t%s\t%s\tok\t%s\n' "${bits}" "${method}" "${arm}" "${seed}" "${summary}" >> "${MANIFEST}"
      done
    done
  done
done

echo "[$(date -Is)] Gemma quantization complete"
