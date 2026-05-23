#!/usr/bin/env bash
set -uo pipefail

GPU_ID="${GPU_ID:-2}"
MODEL_ID="${MODEL_ID:-Llama-3.2-1B-Instruct}"
BASE_MODEL="${BASE_MODEL:-open-unlearning/tofu_${MODEL_ID}_full}"
RANK="${RANK:-4}"
MAX_CALIBRATION_BATCHES="${MAX_CALIBRATION_BATCHES:-2}"
STARTED_AT="$(date -u +%Y%m%dT%H%M%SZ)"
MANIFEST="logs/kforge_overnight_${STARTED_AT}.tsv"

mkdir -p logs
printf "timestamp_utc\tstatus\tforget_split\tretain_split\tmodules\tfactor_mode\tuse_retain_fisher\tstrengths\n" > "${MANIFEST}"

run_block() {
  local forget_split="$1"
  local retain_split="$2"
  local modules="$3"
  local factor_mode="$4"
  local use_retain_fisher="$5"
  local strengths="$6"

  local status="ok"
  echo "[$(date -u --iso-8601=seconds)] START forget=${forget_split} retain=${retain_split} modules=${modules} factor=${factor_mode} retain_fisher=${use_retain_fisher} strengths=${strengths}"

  GPU_ID="${GPU_ID}" \
  MODEL_ID="${MODEL_ID}" \
  BASE_MODEL="${BASE_MODEL}" \
  FORGET_SPLIT="${forget_split}" \
  RETAIN_SPLIT="${retain_split}" \
  RETAIN_LOGS="saves/eval/tofu_${MODEL_ID}_${retain_split}/TOFU_EVAL.json" \
  RANK="${RANK}" \
  MAX_TARGET_MODULES="${modules}" \
  MAX_CALIBRATION_BATCHES="${MAX_CALIBRATION_BATCHES}" \
  FACTOR_MODE="${factor_mode}" \
  USE_RETAIN_FISHER="${use_retain_fisher}" \
  STRENGTHS="${strengths}" \
  ./scripts/kforge_tofu_sweep.sh
  local code=$?

  if [[ "${code}" -ne 0 ]]; then
    status="failed:${code}"
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date -u --iso-8601=seconds)" "${status}" "${forget_split}" "${retain_split}" \
    "${modules}" "${factor_mode}" "${use_retain_fisher}" "${strengths}" >> "${MANIFEST}"
  echo "[$(date -u --iso-8601=seconds)] END status=${status}"
}

# Strength midpoint search around the first viable point from the initial runs.
run_block forget10 retain90 1 kron true "0.0015 0.002 0.003 0.005"

# A1 diagonal Fisher ablation at matched strengths.
run_block forget10 retain90 1 diagonal true "0.0015 0.002 0.003 0.005"

# A2 forget-only Fisher ablation at matched strengths.
run_block forget10 retain90 1 kron false "0.001 0.002 0.003"

# Test whether a second edited module improves forgetting at still-small strength.
run_block forget10 retain90 2 kron true "0.0005 0.001"
run_block forget10 retain90 2 diagonal true "0.0005 0.001"

# Extend to TOFU forget05 once forget10 calibration points are queued.
run_block forget05 retain95 1 kron true "0.001 0.002 0.003"
run_block forget05 retain95 1 diagonal true "0.001 0.002 0.003"

# Rank and calibration-size probes around the utility-preserving regime.
RANK=2
run_block forget10 retain90 1 kron true "0.001 0.002 0.003"
run_block forget10 retain90 1 diagonal true "0.001 0.002 0.003"

RANK=8
run_block forget10 retain90 1 kron true "0.0005 0.001 0.002"
run_block forget10 retain90 1 diagonal true "0.0005 0.001 0.002"

RANK=4
MAX_CALIBRATION_BATCHES=8
run_block forget10 retain90 1 kron true "0.001 0.002"
run_block forget10 retain90 1 diagonal true "0.001 0.002"

# Tiny forget-set regime from the PLAN's TOFU forget01 target.
MAX_CALIBRATION_BATCHES=2
run_block forget01 retain99 1 kron true "0.001 0.002"
run_block forget01 retain99 1 diagonal true "0.001 0.002"

echo "[$(date -u --iso-8601=seconds)] Overnight K-FORGE sweep complete. Manifest: ${MANIFEST}"
