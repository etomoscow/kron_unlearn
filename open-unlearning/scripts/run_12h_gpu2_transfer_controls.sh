#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-2}"
HOURS="${HOURS:-12}"
DEADLINE=$(( $(date +%s) + HOURS * 3600 ))

run_if_time() {
  local label="$1"
  shift
  if (( $(date +%s) >= DEADLINE )); then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] deadline reached before ${label}"
    exit 0
  fi
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting ${label}"
  "$@"
}

run_if_time "forget05 SimNPO weight_svd" env GPU_ID="${GPU_ID}" \
  FORGET_SPLIT=forget05 RETAIN_SPLIT=retain95 HOLDOUT_SPLIT=holdout05 \
  TEMPLATE=saves/unlearn/KFORGE_TOFU_forget05_R2_M1_B32_S0p45_kron_retain_cfix_v2_transfer_lam0p01 \
  ./scripts/run_init_control_weight_svd_simnpo.sh

run_if_time "forget01 SimNPO random" env GPU_ID="${GPU_ID}" \
  FORGET_SPLIT=forget01 RETAIN_SPLIT=retain99 HOLDOUT_SPLIT=holdout01 \
  TEMPLATE=saves/unlearn/KFORGE_TOFU_forget01_R2_M1_B32_S0p45_kron_retain_cfix_v2_transfer_lam0p01 \
  ./scripts/run_init_control_random_simnpo.sh

run_if_time "forget01 SimNPO weight_svd" env GPU_ID="${GPU_ID}" \
  FORGET_SPLIT=forget01 RETAIN_SPLIT=retain99 HOLDOUT_SPLIT=holdout01 \
  TEMPLATE=saves/unlearn/KFORGE_TOFU_forget01_R2_M1_B32_S0p45_kron_retain_cfix_v2_transfer_lam0p01 \
  ./scripts/run_init_control_weight_svd_simnpo.sh

run_if_time "forget05 NPO random" env GPU_ID="${GPU_ID}" \
  FORGET_SPLIT=forget05 RETAIN_SPLIT=retain95 HOLDOUT_SPLIT=holdout05 \
  TEMPLATE=saves/unlearn/KFORGE_TOFU_forget05_R2_M1_B32_S0p45_kron_retain_cfix_v2_transfer_lam0p01 \
  ./scripts/run_init_control_random_npo.sh
