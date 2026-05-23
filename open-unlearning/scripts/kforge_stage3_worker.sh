#!/usr/bin/env bash
set -uo pipefail

GPU_ID="${GPU_ID:?GPU_ID is required}"
WORKER_LABEL="${WORKER_LABEL:?WORKER_LABEL is required}"
MODEL_ID="${MODEL_ID:-Llama-3.2-1B-Instruct}"
BASE_MODEL="${BASE_MODEL:-open-unlearning/tofu_${MODEL_ID}_full}"
STARTED_AT="$(date -u +%Y%m%dT%H%M%SZ)"
MANIFEST="logs/kforge_stage3_${WORKER_LABEL}_${STARTED_AT}.tsv"

mkdir -p logs
printf "timestamp_utc\tstatus\tlabel\tforget_split\tretain_split\trank\tmodules\tbatches\tfactor_mode\tuse_retain_fisher\tstrengths\tsuffix\n" > "${MANIFEST}"

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
  echo "[$(date -u --iso-8601=seconds)] START ${WORKER_LABEL}/${label} gpu=${GPU_ID} forget=${forget_split} retain=${retain_split} rank=${rank} modules=${modules} batches=${batches} factor=${factor_mode} retain_fisher=${use_retain_fisher} strengths=${strengths}"

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
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date -u --iso-8601=seconds)" "${status}" "${label}" "${forget_split}" "${retain_split}" \
    "${rank}" "${modules}" "${batches}" "${factor_mode}" "${use_retain_fisher}" "${strengths}" "${run_suffix}" >> "${MANIFEST}"
  echo "[$(date -u --iso-8601=seconds)] END ${WORKER_LABEL}/${label} status=${status}"
}

case "${WORKER_LABEL}" in
  frontier)
    run_block frontier_tight forget10 retain90 2 1 2 kron true "0.00305 0.0031 0.00315 0.0032" '.*mlp\.down_proj$' "_stage3down"
    run_block frontier_diag forget10 retain90 2 1 2 diagonal true "0.003 0.0031 0.0032" '.*mlp\.down_proj$' "_stage3down"
    run_block frontier_forgetonly forget10 retain90 2 1 2 kron false "0.003 0.0031 0.0032" '.*mlp\.down_proj$' "_stage3down"
    ;;
  transfer)
    run_block forget05_r2 forget05 retain95 2 1 2 kron true "0.002 0.003 0.004" '.*mlp\.down_proj$' "_stage3down"
    run_block forget01_r2 forget01 retain99 2 1 2 kron true "0.002 0.003 0.004" '.*mlp\.down_proj$' "_stage3down"
    run_block forget05_gate forget05 retain95 2 1 2 kron true "0.002 0.003" '.*mlp\.gate_proj$' "_stage3gate"
    ;;
  *)
    echo "Unknown WORKER_LABEL=${WORKER_LABEL}" >&2
    exit 2
    ;;
esac

echo "[$(date -u --iso-8601=seconds)] Stage-3 worker ${WORKER_LABEL} complete. Manifest: ${MANIFEST}"
