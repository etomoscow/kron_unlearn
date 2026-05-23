#!/usr/bin/env bash
set -euo pipefail

TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
GPU_ID="${GPU_ID:-2}"
FORGET_SPLIT="${FORGET_SPLIT:-forget10}"
RETAIN_SPLIT="${RETAIN_SPLIT:-retain90}"
MODEL_ID="${MODEL_ID:-Llama-3.2-1B-Instruct}"
BASE_MODEL="${BASE_MODEL:-open-unlearning/tofu_${MODEL_ID}_full}"
RETAIN_LOGS="${RETAIN_LOGS:-saves/eval/tofu_${MODEL_ID}_${RETAIN_SPLIT}/TOFU_EVAL.json}"
MAX_CALIBRATION_BATCHES="${MAX_CALIBRATION_BATCHES:-32}"
STRENGTHS="${STRENGTHS:-0.002 0.0025 0.003 0.0033 0.004}"
COEFFS="${COEFFS:-0.01 0.1 1.0}"

mkdir -p logs
MANIFEST="logs/kforge_week2_adaptive_damping_${TS}.tsv"
printf "timestamp\tcoeff\tstatus\n" > "${MANIFEST}"

for COEFF in ${COEFFS}; do
  TAG="${COEFF//./p}"
  printf "%s\t%s\t%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${COEFF}" started >> "${MANIFEST}"
  GPU_ID="${GPU_ID}" \
  FORGET_SPLIT="${FORGET_SPLIT}" \
  RETAIN_SPLIT="${RETAIN_SPLIT}" \
  MODEL_ID="${MODEL_ID}" \
  BASE_MODEL="${BASE_MODEL}" \
  RETAIN_LOGS="${RETAIN_LOGS}" \
  RANK=2 \
  MAX_TARGET_MODULES=1 \
  MAX_CALIBRATION_BATCHES="${MAX_CALIBRATION_BATCHES}" \
  FACTOR_MODE=kron \
  USE_RETAIN_FISHER=true \
  STRENGTHS="${STRENGTHS}" \
  RUN_SUFFIX="_adamp${TAG}" \
  EXTRA_TRAIN_ARGS="trainer.method_args.damping_mode=adaptive trainer.method_args.adaptive_damping_coeff=${COEFF}" \
  bash scripts/kforge_tofu_sweep.sh
  printf "%s\t%s\t%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${COEFF}" ok >> "${MANIFEST}"
done

echo "Adaptive damping manifest: ${MANIFEST}"
