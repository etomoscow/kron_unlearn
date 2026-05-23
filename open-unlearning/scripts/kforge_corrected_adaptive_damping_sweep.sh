#!/usr/bin/env bash
set -euo pipefail

TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
GPU_ID="${GPU_ID:-0}"
MODEL_TORCH_DTYPE="${MODEL_TORCH_DTYPE:-float32}"
BATCHES="${BATCHES:-32}"
STRENGTHS="${STRENGTHS:-0.002 0.0025 0.003 0.0033 0.004}"
COEFFS="${COEFFS:-0.01 0.1 1.0}"
MANIFEST="${MANIFEST:-logs/kforge_corrected_adaptive_damping_${TS}.tsv}"

mkdir -p logs
printf "timestamp\tcoeff\tstatus\n" > "${MANIFEST}"

for COEFF in ${COEFFS}; do
  TAG="${COEFF//./p}"
  printf "%s\t%s\tstarted\n" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${COEFF}" >> "${MANIFEST}"
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
  RUN_SUFFIX="_cfix_adamp${TAG}" \
  EXTRA_TRAIN_ARGS="trainer.method_args.edit_variant=legacy_v1 trainer.method_args.damping_mode=adaptive trainer.method_args.adaptive_damping_coeff=${COEFF}" \
    bash scripts/kforge_tofu_sweep.sh
  printf "%s\t%s\tok\n" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${COEFF}" >> "${MANIFEST}"
done

echo "Corrected adaptive damping manifest: ${MANIFEST}"
