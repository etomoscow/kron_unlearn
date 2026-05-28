#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-1}"
TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"

mkdir -p logs

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] missing paper runs queue started on GPU ${GPU_ID}"

# 1B transfer grid gap: forget05 SimNPO with the alpha=.60 K-FORGE initializer.
GPU_ID="${GPU_ID}" \
MODEL_ID="Llama-3.2-1B-Instruct" \
BASE_MODEL="open-unlearning/tofu_Llama-3.2-1B-Instruct_full" \
KFORGE_INIT_MODEL="saves/unlearn/KFORGE_TOFU_forget05_R2_M1_B32_S0p6_kron_retain_cfix_v2_transfer_lam0p01" \
KFORGE_INIT_TAG="kforge_s06" \
MODEL_TORCH_DTYPE="float32" \
FORGET_SPLIT="forget05" \
HOLDOUT_SPLIT="holdout05" \
RETAIN_SPLIT="retain95" \
RETAIN_LOGS="saves/eval/tofu_Llama-3.2-1B-Instruct_retain95/TOFU_EVAL.json" \
TRAINERS="SimNPO" \
STEP_BUDGETS="50 100 250" \
SEEDS="0 1 2" \
INIT_MODES="kforge" \
RUN_TAG="v2lam0p01_corr" \
SKIP_COMPLETED="true" \
  bash scripts/kforge_week2_init_experiment.sh

# 3B seed-0 gap for the symmetric 50/100 sanity-check table.
for trainer in NPO SimNPO; do
  GPU_ID="${GPU_ID}" \
  MODEL_ID="Llama-3.2-3B-Instruct" \
  BASE_MODEL="open-unlearning/tofu_Llama-3.2-3B-Instruct_full" \
  KFORGE_INIT_MODEL="saves/unlearn/KFORGE_TOFU_forget10_R2_M1_B32_S0p45_kron_retain_cfix_v2_lam0p01_3b_minimal" \
  KFORGE_INIT_TAG="kforge_s045" \
  MODEL_TORCH_DTYPE="float32" \
  FORGET_SPLIT="forget10" \
  HOLDOUT_SPLIT="holdout10" \
  RETAIN_SPLIT="retain90" \
  RETAIN_LOGS="saves/eval/tofu_Llama-3.2-3B-Instruct_retain90/TOFU_EVAL.json" \
  TRAINERS="${trainer}" \
  STEP_BUDGETS="50 100" \
  SEEDS="0" \
  INIT_MODES="scratch kforge" \
  RUN_TAG="3b_minimal_v2" \
  SKIP_COMPLETED="true" \
    bash scripts/kforge_week2_init_experiment.sh
done

# Complete the 3B 250-step grid in case we decide to report the longer budget.
for trainer in NPO SimNPO; do
  GPU_ID="${GPU_ID}" \
  MODEL_ID="Llama-3.2-3B-Instruct" \
  BASE_MODEL="open-unlearning/tofu_Llama-3.2-3B-Instruct_full" \
  KFORGE_INIT_MODEL="saves/unlearn/KFORGE_TOFU_forget10_R2_M1_B32_S0p45_kron_retain_cfix_v2_lam0p01_3b_minimal" \
  KFORGE_INIT_TAG="kforge_s045" \
  MODEL_TORCH_DTYPE="float32" \
  FORGET_SPLIT="forget10" \
  HOLDOUT_SPLIT="holdout10" \
  RETAIN_SPLIT="retain90" \
  RETAIN_LOGS="saves/eval/tofu_Llama-3.2-3B-Instruct_retain90/TOFU_EVAL.json" \
  TRAINERS="${trainer}" \
  STEP_BUDGETS="250" \
  SEEDS="0 1 2" \
  INIT_MODES="scratch kforge" \
  RUN_TAG="3b_minimal_v2" \
  SKIP_COMPLETED="true" \
    bash scripts/kforge_week2_init_experiment.sh
done

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] missing paper runs queue finished"
