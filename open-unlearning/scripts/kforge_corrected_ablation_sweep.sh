#!/usr/bin/env bash
set -euo pipefail

TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
GPU_ID="${GPU_ID:-2}"
MODEL_TORCH_DTYPE="${MODEL_TORCH_DTYPE:-float32}"
BATCHES="${BATCHES:-32}"
STRENGTHS="${STRENGTHS:-0.002 0.0025 0.003 0.0033 0.004}"
MANIFEST="${MANIFEST:-logs/kforge_corrected_ablation_${TS}.tsv}"

mkdir -p logs
printf "timestamp\tablation\tfactor_mode\tuse_retain_fisher\tstatus\n" > "${MANIFEST}"

run_block() {
  local ABLATION="$1"
  local FACTOR_MODE="$2"
  local USE_RETAIN_FISHER="$3"
  local RUN_SUFFIX="$4"

  printf "%s\t%s\t%s\t%s\tstarted\n" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${ABLATION}" "${FACTOR_MODE}" "${USE_RETAIN_FISHER}" \
    >> "${MANIFEST}"
  GPU_ID="${GPU_ID}" \
  MODEL_TORCH_DTYPE="${MODEL_TORCH_DTYPE}" \
  FORGET_SPLIT=forget10 \
  RETAIN_SPLIT=retain90 \
  RANK=2 \
  MAX_TARGET_MODULES=1 \
  MAX_CALIBRATION_BATCHES="${BATCHES}" \
  FACTOR_MODE="${FACTOR_MODE}" \
  USE_RETAIN_FISHER="${USE_RETAIN_FISHER}" \
  STRENGTHS="${STRENGTHS}" \
  RUN_SUFFIX="${RUN_SUFFIX}" \
  EXTRA_TRAIN_ARGS="trainer.method_args.edit_variant=legacy_v1" \
    bash scripts/kforge_tofu_sweep.sh
  printf "%s\t%s\t%s\t%s\tok\n" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${ABLATION}" "${FACTOR_MODE}" "${USE_RETAIN_FISHER}" \
    >> "${MANIFEST}"
}

run_block "a1_diagonal" diagonal true "_cfix_a1_diag"
run_block "a2_forgetonly" kron false "_cfix_a2_forgetonly"

echo "Corrected ablation manifest: ${MANIFEST}"
