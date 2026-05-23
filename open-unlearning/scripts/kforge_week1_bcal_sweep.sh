#!/usr/bin/env bash
set -euo pipefail

TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="${LOG_DIR:-logs}"
MANIFEST="${MANIFEST:-${LOG_DIR}/kforge_week1_bcal_${TS}.tsv}"
GPU0="${GPU0:-0}"
GPU2="${GPU2:-2}"
BATCH_LIST_GPU0="${BATCH_LIST_GPU0:-2 8 32}"
BATCH_LIST_GPU2="${BATCH_LIST_GPU2:-4 16 64}"
STRENGTH_LIST="${STRENGTH_LIST:-0.002 0.0025 0.003 0.0033 0.004}"
RUN_SUFFIX_BASE="${RUN_SUFFIX_BASE:-_bcal}"
MODEL_TORCH_DTYPE="${MODEL_TORCH_DTYPE:-float32}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

mkdir -p "${LOG_DIR}"
printf "timestamp\tgpu\tbatches\tstrengths\tstatus\n" > "${MANIFEST}"

start_worker() {
  local GPU="$1"
  local BATCH_LIST="$2"
  local SESSION="kforge_bcal_gpu${GPU}_${TS}"
  LOG_FILE="${LOG_DIR}/${SESSION}.log"

  tmux new-session -d -s "${SESSION}" \
    "cd '${PWD}' && for BATCHES in ${BATCH_LIST}; do printf '%s\t%s\t%s\t%s\t%s\n' \"\$(date -u +%Y-%m-%dT%H:%M:%SZ)\" '${GPU}' \"\${BATCHES}\" '${STRENGTH_LIST}' started >> '${MANIFEST}'; GPU_ID='${GPU}' FORGET_SPLIT=forget10 RETAIN_SPLIT=retain90 RANK=2 MAX_TARGET_MODULES=1 MAX_CALIBRATION_BATCHES=\"\${BATCHES}\" FACTOR_MODE=kron USE_RETAIN_FISHER=true STRENGTHS='${STRENGTH_LIST}' RUN_SUFFIX='${RUN_SUFFIX_BASE}' MODEL_TORCH_DTYPE='${MODEL_TORCH_DTYPE}' EXTRA_TRAIN_ARGS='${EXTRA_TRAIN_ARGS}' bash scripts/kforge_tofu_sweep.sh; status=\$?; printf '%s\t%s\t%s\t%s\t%s\n' \"\$(date -u +%Y-%m-%dT%H:%M:%SZ)\" '${GPU}' \"\${BATCHES}\" '${STRENGTH_LIST}' \"exit:\${status}\" >> '${MANIFEST}'; if [[ \${status} -ne 0 ]]; then exit \${status}; fi; done" \
    > "${LOG_FILE}" 2>&1

  printf "started\t%s\t%s\t%s\t%s\n" "${GPU}" "${BATCH_LIST}" "${STRENGTH_LIST}" "${SESSION}" >> "${MANIFEST}"
}

start_worker "${GPU0}" "${BATCH_LIST_GPU0}"
start_worker "${GPU2}" "${BATCH_LIST_GPU2}"

echo "Started B_cal sweep sessions. Manifest: ${MANIFEST}"
tmux ls | grep "kforge_bcal_.*_${TS}" || true
