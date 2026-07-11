#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

GPU_ID="${GPU_ID:?set GPU_ID}"
SEED="${SEED:?set SEED}"
BASE_MODEL="${BASE_MODEL:?set BASE_MODEL}"
KFORGE_INIT_MODEL="${KFORGE_INIT_MODEL:?set KFORGE_INIT_MODEL}"
RUN_TAG="${RUN_TAG:-rebuttal_wall_exact_v1}"
RETAIN_LOGS="saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json"

python - <<'PY'
import accelerate
import torch
import transformers

expected = {"transformers": "4.51.3", "accelerate": "0.34.2"}
actual = {"transformers": transformers.__version__, "accelerate": accelerate.__version__}
if actual != expected:
    raise RuntimeError(f"incompatible trainer stack: expected {expected}, found {actual}")
print(f"trainer stack: {actual}; torch={torch.__version__}")
PY

# The longer scratch budgets cover the complete 80.2 s K-FORGE setup for every
# paired runtime on the measured hardware, as well as the analytical FLOP charge.
for spec in NPO:kforge:50 NPO:scratch:73 SimNPO:kforge:50 SimNPO:scratch:86; do
  IFS=: read -r trainer init steps <<< "${spec}"
  if [[ "${init}" == "kforge" ]]; then
    init_modes=kforge
    init_tag=kforge_s06_pinned
  else
    init_modes=scratch
    init_tag=scratch
  fi
  env \
    GPU_ID="${GPU_ID}" \
    TS="$(date -u +%Y%m%dT%H%M%SZ)_wall_seed${SEED}_${trainer}_${init_tag}_S${steps}" \
    MODEL_ID=Llama-3.2-1B-Instruct \
    BASE_MODEL="${BASE_MODEL}" \
    KFORGE_INIT_MODEL="${KFORGE_INIT_MODEL}" \
    KFORGE_INIT_TAG=kforge_s06_pinned \
    MODEL_TORCH_DTYPE=float32 \
    FORGET_SPLIT=forget10 \
    HOLDOUT_SPLIT=holdout10 \
    RETAIN_SPLIT=retain90 \
    RETAIN_LOGS="${RETAIN_LOGS}" \
    TRAINERS="${trainer}" \
    STEP_BUDGETS="${steps}" \
    SEEDS="${SEED}" \
    INIT_MODES="${init_modes}" \
    RUN_TAG="${RUN_TAG}" \
    bash scripts/kforge_week2_init_experiment.sh
done
