#!/usr/bin/env bash
set -euo pipefail

TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
GPU_ID="${GPU_ID:-1}"
MODEL_TORCH_DTYPE="${MODEL_TORCH_DTYPE:-float32}"
LEGACY_BCAL_LIST="${LEGACY_BCAL_LIST:-2 4 8 16 32 64}"
LEGACY_STRENGTHS="${LEGACY_STRENGTHS:-0.002 0.0025 0.003 0.0033 0.004}"
V2_BATCHES="${V2_BATCHES:-32}"
V2_STRENGTHS="${V2_STRENGTHS:-0.0005 0.001 0.002 0.003 0.004}"
V2_LAMBDAS="${V2_LAMBDAS:-0.0 0.0001 0.001 0.01}"
MANIFEST="${MANIFEST:-logs/kforge_corrected_validation_${TS}.tsv}"

mkdir -p logs
printf "timestamp\tphase\tbatches\tlambda\tstrengths\tstatus\n" > "${MANIFEST}"

for BATCHES in ${LEGACY_BCAL_LIST}; do
  printf "%s\tlegacy_bcal\t%s\t-\t%s\tstarted\n" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${BATCHES}" "${LEGACY_STRENGTHS}" \
    >> "${MANIFEST}"
  GPU_ID="${GPU_ID}" \
  MODEL_TORCH_DTYPE="${MODEL_TORCH_DTYPE}" \
  FORGET_SPLIT=forget10 \
  RETAIN_SPLIT=retain90 \
  RANK=2 \
  MAX_TARGET_MODULES=1 \
  MAX_CALIBRATION_BATCHES="${BATCHES}" \
  FACTOR_MODE=kron \
  USE_RETAIN_FISHER=true \
  STRENGTHS="${LEGACY_STRENGTHS}" \
  RUN_SUFFIX="_cfix_legacy_bcal" \
  EXTRA_TRAIN_ARGS="trainer.method_args.edit_variant=legacy_v1" \
    bash scripts/kforge_tofu_sweep.sh
  printf "%s\tlegacy_bcal\t%s\t-\t%s\tok\n" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${BATCHES}" "${LEGACY_STRENGTHS}" \
    >> "${MANIFEST}"
done

for LAMBDA in ${V2_LAMBDAS}; do
  LAMBDA_TAG="${LAMBDA//./p}"
  printf "%s\twiener_v2\t%s\t%s\t%s\tstarted\n" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${V2_BATCHES}" "${LAMBDA}" "${V2_STRENGTHS}" \
    >> "${MANIFEST}"
  GPU_ID="${GPU_ID}" \
  MODEL_TORCH_DTYPE="${MODEL_TORCH_DTYPE}" \
  FORGET_SPLIT=forget10 \
  RETAIN_SPLIT=retain90 \
  RANK=2 \
  MAX_TARGET_MODULES=1 \
  MAX_CALIBRATION_BATCHES="${V2_BATCHES}" \
  FACTOR_MODE=kron \
  USE_RETAIN_FISHER=true \
  STRENGTHS="${V2_STRENGTHS}" \
  RUN_SUFFIX="_cfix_v2_lam${LAMBDA_TAG}" \
  EXTRA_TRAIN_ARGS="trainer.method_args.edit_variant=wiener_v2 trainer.method_args.lambda_tradeoff=${LAMBDA}" \
    bash scripts/kforge_tofu_sweep.sh
  printf "%s\twiener_v2\t%s\t%s\t%s\tok\n" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${V2_BATCHES}" "${LAMBDA}" "${V2_STRENGTHS}" \
    >> "${MANIFEST}"
done

echo "Corrected validation manifest: ${MANIFEST}"
