#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

GPU_ID="${GPU_ID:?set GPU_ID}"
BASE_MODEL="saves/finetune/tofu_gemma-3-1b-it_full_rebuttal_gemma3_1b_v1"
RETAIN_LOGS="saves/eval/tofu_gemma-3-1b-it_retain90_rebuttal_gemma3_1b_v1/TOFU_EVAL.json"

for spec in NPO:103 SimNPO:105; do
  IFS=: read -r trainer steps <<< "${spec}"
  env \
    GPU_ID="${GPU_ID}" \
    TS="$(date -u +%Y%m%dT%H%M%SZ)_compute_seed3_${trainer}" \
    MODEL_ID=gemma-3-1b-it \
    BASE_MODEL="${BASE_MODEL}" \
    MODEL_TORCH_DTYPE=bfloat16 \
    FORGET_SPLIT=forget10 \
    HOLDOUT_SPLIT=holdout10 \
    RETAIN_SPLIT=retain90 \
    RETAIN_LOGS="${RETAIN_LOGS}" \
    TRAINERS="${trainer}" \
    STEP_BUDGETS="${steps}" \
    SEEDS=3 \
    INIT_MODES=scratch \
    RUN_TAG=rebuttal_gemma3_compute_matched_v2 \
    bash scripts/kforge_week2_init_experiment.sh
done
