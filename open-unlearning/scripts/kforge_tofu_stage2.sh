#!/usr/bin/env bash
set -uo pipefail

GPU_ID="${GPU_ID:-2}"
MODEL_ID="${MODEL_ID:-Llama-3.2-1B-Instruct}"
BASE_MODEL="${BASE_MODEL:-open-unlearning/tofu_${MODEL_ID}_full}"
STARTED_AT="$(date -u +%Y%m%dT%H%M%SZ)"
MANIFEST="logs/kforge_stage2_${STARTED_AT}.tsv"

mkdir -p logs
printf "timestamp_utc\tstatus\tlabel\tforget_split\tretain_split\trank\tmodules\tbatches\tfactor_mode\tuse_retain_fisher\tstrengths\n" > "${MANIFEST}"

run_block() {
  local label="$1"
  local forget_split="$2"
  local retain_split="$3"
  local rank="$4"
  local modules="$5"
  local batches="$6"
  local factor_mode="$7"
  local use_retain_fisher="$8"
  local strengths="$9"
  local target_regex="${10}"
  local run_suffix="${11}"

  local status="ok"
  echo "[$(date -u --iso-8601=seconds)] START ${label} forget=${forget_split} retain=${retain_split} rank=${rank} modules=${modules} batches=${batches} factor=${factor_mode} retain_fisher=${use_retain_fisher} strengths=${strengths}"

  GPU_ID="${GPU_ID}" \
  MODEL_ID="${MODEL_ID}" \
  BASE_MODEL="${BASE_MODEL}" \
  FORGET_SPLIT="${forget_split}" \
  RETAIN_SPLIT="${retain_split}" \
  RETAIN_LOGS="saves/eval/tofu_${MODEL_ID}_${retain_split}/TOFU_EVAL.json" \
  RANK="${rank}" \
  MAX_TARGET_MODULES="${modules}" \
  MAX_CALIBRATION_BATCHES="${batches}" \
  FACTOR_MODE="${factor_mode}" \
  USE_RETAIN_FISHER="${use_retain_fisher}" \
  STRENGTHS="${strengths}" \
  TARGET_MODULES_REGEX="${target_regex}" \
  RUN_SUFFIX="${run_suffix}" \
  ./scripts/kforge_tofu_sweep.sh
  local code=$?

  if [[ "${code}" -ne 0 ]]; then
    status="failed:${code}"
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date -u --iso-8601=seconds)" "${status}" "${label}" "${forget_split}" "${retain_split}" \
    "${rank}" "${modules}" "${batches}" "${factor_mode}" "${use_retain_fisher}" "${strengths}" >> "${MANIFEST}"
  echo "[$(date -u --iso-8601=seconds)] END ${label} status=${status}"
}

# Refine the best high-utility rank-2 frontier from the overnight run.
run_block frontier_r2 forget10 retain90 2 1 2 kron true "0.00325 0.0035 0.00375 0.004" '.*mlp\.down_proj$' "_down"

# Check whether a slightly wider edit improves rank-2 without crossing the utility cliff.
run_block frontier_r2_m2 forget10 retain90 2 2 2 kron true "0.0015 0.002 0.0025" '.*mlp\.down_proj$' "_down"

# A4 layer-selection probes at conservative strengths.
run_block layer_gate forget10 retain90 2 1 2 kron true "0.001 0.002 0.003" '.*mlp\.gate_proj$' "_gate"
run_block layer_up forget10 retain90 2 1 2 kron true "0.001 0.002 0.003" '.*mlp\.up_proj$' "_up"
run_block layer_o forget10 retain90 2 1 2 kron true "0.001 0.002 0.003" '.*self_attn\.o_proj$' "_o"

echo "[$(date -u --iso-8601=seconds)] Stage-2 K-FORGE sweep complete. Manifest: ${MANIFEST}"
