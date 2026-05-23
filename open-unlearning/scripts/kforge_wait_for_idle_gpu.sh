#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:?usage: kforge_wait_for_idle_gpu.sh GPU_ID [deadline_epoch]}"
DEADLINE_EPOCH="${2:-0}"
MAX_MEMORY_USED_MB="${MAX_MEMORY_USED_MB:-20000}"
MAX_UTILIZATION="${MAX_UTILIZATION:-15}"
CONSECUTIVE_CHECKS="${CONSECUTIVE_CHECKS:-5}"
SLEEP_SECONDS="${SLEEP_SECONDS:-60}"

ok_count=0
while true; do
  now="$(date +%s)"
  if [[ "${DEADLINE_EPOCH}" != "0" && "${now}" -ge "${DEADLINE_EPOCH}" ]]; then
    echo "deadline reached while waiting for GPU ${GPU_ID}" >&2
    exit 124
  fi

  line="$(nvidia-smi --id="${GPU_ID}" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)"
  memory_used="$(printf '%s' "${line}" | awk -F, '{gsub(/ /, "", $1); print $1}')"
  utilization="$(printf '%s' "${line}" | awk -F, '{gsub(/ /, "", $2); print $2}')"

  if [[ "${memory_used}" -lt "${MAX_MEMORY_USED_MB}" && "${utilization}" -lt "${MAX_UTILIZATION}" ]]; then
    ok_count=$((ok_count + 1))
  else
    ok_count=0
  fi

  if [[ "${ok_count}" -ge "${CONSECUTIVE_CHECKS}" ]]; then
    echo "GPU ${GPU_ID} idle: memory_used=${memory_used}MB utilization=${utilization}%"
    exit 0
  fi

  sleep "${SLEEP_SECONDS}"
done
