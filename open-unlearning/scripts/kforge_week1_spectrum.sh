#!/usr/bin/env bash
set -euo pipefail

TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
GPU_ID="${GPU_ID:-2}"
MODEL_ID="${MODEL_ID:-Llama-3.2-1B-Instruct}"
BASE_MODEL="${BASE_MODEL:-open-unlearning/tofu_${MODEL_ID}_full}"
MODEL_TORCH_DTYPE="${MODEL_TORCH_DTYPE:-float32}"
FORGET_SPLIT="${FORGET_SPLIT:-forget10}"
RETAIN_SPLIT="${RETAIN_SPLIT:-retain90}"
RETAIN_LOGS="${RETAIN_LOGS:-saves/eval/tofu_${MODEL_ID}_${RETAIN_SPLIT}/TOFU_EVAL.json}"
MAX_CALIBRATION_BATCHES="${MAX_CALIBRATION_BATCHES:-32}"
SPECTRUM_TOP_K="${SPECTRUM_TOP_K:-64}"
WAIT_FOR_JOBS="${WAIT_FOR_JOBS:-true}"
LAYERS="${LAYERS:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15}"
MODULES="${MODULES:-self_attn.q_proj self_attn.k_proj self_attn.v_proj self_attn.o_proj mlp.gate_proj mlp.up_proj mlp.down_proj}"

mkdir -p .cache/triton .cache/torch_extensions .cache/hf .cache/xdg logs saves/spectrum
MANIFEST="logs/kforge_week1_spectrum_${TS}.tsv"
printf "timestamp\tmodule_regex\tstatus\tpath\n" > "${MANIFEST}"

if [[ "${WAIT_FOR_JOBS}" == "true" ]]; then
  while tmux ls 2>/dev/null | grep -Eq 'kforge_bcal_gpu|tofu_week1_baselines'; do
    sleep 300
  done
fi

for LAYER in ${LAYERS}; do
for MODULE in ${MODULES}; do
  MODULE_REGEX="${MODULE//./\\.}"
  REGEX="model\\.layers\\.${LAYER}\\.${MODULE_REGEX}$"
  TAG="layer${LAYER}_${MODULE//./_}"
  OUT="saves/spectrum/kforge_${MODEL_ID}_${FORGET_SPLIT}_B${MAX_CALIBRATION_BATCHES}_${TAG}_${TS}.json"
  TASK_NAME="KFORGE_SPECTRUM_${FORGET_SPLIT}_B${MAX_CALIBRATION_BATCHES}_${TAG}_${TS}"
  printf "%s\t%s\t%s\t%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${REGEX}" started "${OUT}" >> "${MANIFEST}"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  TRITON_CACHE_DIR="${PWD}/.cache/triton" \
  TORCH_EXTENSIONS_DIR="${PWD}/.cache/torch_extensions" \
  HF_HOME="${PWD}/.cache/hf" \
  HF_HUB_DISABLE_XET=1 \
  XDG_CACHE_HOME="${PWD}/.cache/xdg" \
  WANDB_DISABLED=true \
  python src/train.py --config-name=unlearn experiment=unlearn/tofu/kforge \
    forget_split="${FORGET_SPLIT}" retain_split="${RETAIN_SPLIT}" \
    model.model_args.attn_implementation=sdpa \
    model.model_args.torch_dtype="${MODEL_TORCH_DTYPE}" \
    model.tokenizer_args.pretrained_model_name_or_path="${BASE_MODEL}" \
    retain_logs_path="${RETAIN_LOGS}" \
    trainer.args.report_to=none trainer.args.do_eval=false \
    trainer.method_args.max_target_modules=1 \
    trainer.method_args.max_calibration_batches="${MAX_CALIBRATION_BATCHES}" \
    trainer.method_args.target_modules_regex="${REGEX}" \
    trainer.method_args.rank=2 \
    trainer.method_args.strength=0.0 \
    trainer.method_args.factor_mode=kron \
    trainer.method_args.use_retain_fisher=true \
    trainer.method_args.spectrum_output_path="${OUT}" \
    trainer.method_args.spectrum_top_k="${SPECTRUM_TOP_K}" \
    trainer.method_args.skip_edit=true \
    task_name="${TASK_NAME}" hydra.run.dir="outputs/${TASK_NAME}" \
    2>&1 | tee "logs/${TASK_NAME}.log"

  printf "%s\t%s\t%s\t%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${REGEX}" ok "${OUT}" >> "${MANIFEST}"
done
done

echo "Spectrum manifest: ${MANIFEST}"
