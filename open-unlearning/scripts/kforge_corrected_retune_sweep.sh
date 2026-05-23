#!/usr/bin/env bash
set -euo pipefail

TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
GPU_ID="${GPU_ID:-2}"
MODE="${MODE:-legacy}"
MODEL_TORCH_DTYPE="${MODEL_TORCH_DTYPE:-float32}"
BATCHES="${BATCHES:-32}"
MANIFEST="${MANIFEST:-logs/kforge_corrected_retune_${MODE}_${TS}.tsv}"

mkdir -p logs
printf "timestamp\tmode\tlambda\tstrengths\tstatus\n" > "${MANIFEST}"

case "${MODE}" in
  legacy)
    STRENGTHS="${STRENGTHS:-0.00001 0.00002 0.00004 0.00008 0.00016 0.00032}"
    printf "%s\tlegacy\t-\t%s\tstarted\n" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${STRENGTHS}" >> "${MANIFEST}"
    GPU_ID="${GPU_ID}" \
    MODEL_TORCH_DTYPE="${MODEL_TORCH_DTYPE}" \
    FORGET_SPLIT=forget10 \
    RETAIN_SPLIT=retain90 \
    RANK=2 \
    MAX_TARGET_MODULES=1 \
    MAX_CALIBRATION_BATCHES="${BATCHES}" \
    FACTOR_MODE=kron \
    USE_RETAIN_FISHER=true \
    STRENGTHS="${STRENGTHS}" \
    RUN_SUFFIX="_cfix_retune_legacy" \
    EXTRA_TRAIN_ARGS="trainer.method_args.edit_variant=legacy_v1" \
      bash scripts/kforge_tofu_sweep.sh
    printf "%s\tlegacy\t-\t%s\tok\n" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${STRENGTHS}" >> "${MANIFEST}"
    ;;
  v2)
    STRENGTHS="${STRENGTHS:-0.1 0.3 1.0 3.0 10.0}"
    LAMBDAS="${LAMBDAS:-0.0 0.0001 0.001 0.01}"
    for LAMBDA in ${LAMBDAS}; do
      TAG="${LAMBDA//./p}"
      printf "%s\tv2\t%s\t%s\tstarted\n" \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${LAMBDA}" "${STRENGTHS}" >> "${MANIFEST}"
      GPU_ID="${GPU_ID}" \
      MODEL_TORCH_DTYPE="${MODEL_TORCH_DTYPE}" \
      FORGET_SPLIT=forget10 \
      RETAIN_SPLIT=retain90 \
      RANK=2 \
      MAX_TARGET_MODULES=1 \
      MAX_CALIBRATION_BATCHES="${BATCHES}" \
      FACTOR_MODE=kron \
      USE_RETAIN_FISHER=true \
      STRENGTHS="${STRENGTHS}" \
      RUN_SUFFIX="_cfix_retune_v2_lam${TAG}" \
      EXTRA_TRAIN_ARGS="trainer.method_args.edit_variant=wiener_v2 trainer.method_args.lambda_tradeoff=${LAMBDA}" \
        bash scripts/kforge_tofu_sweep.sh
      printf "%s\tv2\t%s\t%s\tok\n" \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${LAMBDA}" "${STRENGTHS}" >> "${MANIFEST}"
    done
    ;;
  *)
    echo "Unknown MODE=${MODE}" >&2
    exit 2
    ;;
esac

echo "Corrected retune manifest: ${MANIFEST}"
