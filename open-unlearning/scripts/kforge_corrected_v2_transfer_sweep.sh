#!/usr/bin/env bash
set -euo pipefail

TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
GPU_ID="${GPU_ID:-0}"
FORGET_SPLIT="${FORGET_SPLIT:-forget05}"
RETAIN_SPLIT="${RETAIN_SPLIT:-retain95}"
MODEL_TORCH_DTYPE="${MODEL_TORCH_DTYPE:-float32}"
BATCHES="${BATCHES:-32}"
STRENGTHS="${STRENGTHS:-0.35 0.45 0.6 0.8}"
LAMBDA="${LAMBDA:-0.01}"
LAMBDA_TAG="${LAMBDA//./p}"
MANIFEST="${MANIFEST:-logs/kforge_corrected_v2_transfer_${FORGET_SPLIT}_${TS}.tsv}"

mkdir -p logs
printf "timestamp\tforget_split\tlambda\tstrengths\tstatus\n" > "${MANIFEST}"
printf "%s\t%s\t%s\t%s\tstarted\n" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${FORGET_SPLIT}" "${LAMBDA}" "${STRENGTHS}" \
  >> "${MANIFEST}"

GPU_ID="${GPU_ID}" \
MODEL_TORCH_DTYPE="${MODEL_TORCH_DTYPE}" \
FORGET_SPLIT="${FORGET_SPLIT}" \
RETAIN_SPLIT="${RETAIN_SPLIT}" \
RETAIN_LOGS="saves/eval/tofu_Llama-3.2-1B-Instruct_${RETAIN_SPLIT}/TOFU_EVAL.json" \
RANK=2 \
MAX_TARGET_MODULES=1 \
MAX_CALIBRATION_BATCHES="${BATCHES}" \
FACTOR_MODE=kron \
USE_RETAIN_FISHER=true \
STRENGTHS="${STRENGTHS}" \
RUN_SUFFIX="_cfix_v2_transfer_lam${LAMBDA_TAG}" \
EXTRA_TRAIN_ARGS="trainer.method_args.edit_variant=wiener_v2 trainer.method_args.lambda_tradeoff=${LAMBDA}" \
  bash scripts/kforge_tofu_sweep.sh

printf "%s\t%s\t%s\t%s\tok\n" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${FORGET_SPLIT}" "${LAMBDA}" "${STRENGTHS}" \
  >> "${MANIFEST}"
