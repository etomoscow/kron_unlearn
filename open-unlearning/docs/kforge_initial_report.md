# K-FORGE Initial Report

Date: 2026-05-09

Update: the Week 2 init-experiment analysis is now available in
[`docs/kforge_week2_report.md`](kforge_week2_report.md). The original one-shot
findings below remain useful as implementation and ablation context, but the
current paper framing is K-FORGE as an initializer for NPO/SimNPO rather than as
a standalone SOTA unlearner.

## Scope

This report summarizes the first implementation and experiment pass for
K-FORGE, following the TOFU portion of `PLAN.md`.

Completed scope:

- Implemented K-FORGE as a training-free `UnlearnTrainer`.
- Ran TOFU experiments on `Llama-3.2-1B-Instruct`.
- Swept `forget10`, with transfer probes on `forget05` and `forget01`.
- Ran ablations for Kronecker vs. diagonal factors, retain-whitened vs.
  forget-only factors, rank, strength, calibration batches, and layer target.

Not yet completed:

- MUSE and WMDP evaluation.
- 3B/7B model scaling.
- Full comparison against all OpenUnlearning baselines.
- Relearning and quantization robustness audits.

## Implementation Summary

K-FORGE estimates empirical Kronecker factors for selected linear modules:

- `A = E[x x^T]` from layer inputs.
- `B = E[g g^T]` from output gradients.

It collects both forget-set and retain-set factors, applies damped Cholesky
factorization, forms a retain-whitened forget matrix, runs a truncated SVD, and
applies a negative rank-r edit to the selected weights.

The implementation includes:

- `src/trainer/unlearn/kforge.py`
- `configs/trainer/KFORGE.yaml`
- `configs/experiment/unlearn/tofu/kforge.yaml`
- `scripts/kforge_tofu_sweep.sh`
- `scripts/kforge_tofu_overnight.sh`
- `scripts/kforge_tofu_stage2.sh`
- `scripts/kforge_stage3_worker.sh`
- `community/methods/KFORGE/`
- `docs/kforge.md`

The trainer supports the main ablation knobs needed by the plan:

- `factor_mode=kron|diagonal`
- `use_retain_fisher=true|false`
- `rank`
- `strength`
- `target_modules_regex`
- `max_target_modules`
- `max_calibration_batches`

## Data And Evaluation

Primary benchmark: TOFU.

Primary model:

- `open-unlearning/tofu_Llama-3.2-1B-Instruct_full`

Primary aggregate output:

- `saves/eval/kforge_all_summary.csv`

The aggregate contains 75 evaluated K-FORGE runs.

Baseline full-model metrics on `forget10`:

| Run | Model Utility | Forget Q/A Prob | Forget Q/A ROUGE | Extraction Strength |
|---|---:|---:|---:|---:|
| Base model | 0.5992 | 0.8805 | 0.8201 | 0.7063 |

Lower forget probability, ROUGE, and extraction strength indicate stronger
forgetting. Higher model utility indicates better retention.

## Main Result

The best current utility-preserving `forget10` point is:

`KFORGE_TOFU_forget10_R2_M1_B2_S0p00305_kron_retain_stage3down`

| Run | Model Utility | Forget Q/A Prob | Forget Q/A ROUGE | Extraction Strength |
|---|---:|---:|---:|---:|
| Base model | 0.5992 | 0.8805 | 0.8201 | 0.7063 |
| K-FORGE best, utility >= 0.55 | 0.5508 | 0.5292 | 0.4662 | 0.2134 |

Relative to the base model, this point reduces:

- Forget Q/A probability by 39.9%.
- Forget Q/A ROUGE by 43.2%.
- Extraction strength by 69.8%.

The previous overnight best remains essentially tied:

`KFORGE_TOFU_forget10_R2_M1_B2_S0p003_kron_retain`

| Run | Model Utility | Forget Q/A Prob | Forget Q/A ROUGE | Extraction Strength |
|---|---:|---:|---:|---:|
| K-FORGE overnight best | 0.5535 | 0.5442 | 0.4755 | 0.2269 |

## Frontier Around The Best Point

The strongest tradeoff occurs around strength `0.0030-0.0031` for rank 2,
one `down_proj` target module, two calibration batches, and retain whitening.

| Strength | Model Utility | Forget Q/A Prob | Forget Q/A ROUGE | Extraction Strength |
|---:|---:|---:|---:|---:|
| 0.0020 | 0.5836 | 0.7736 | 0.6412 | 0.4768 |
| 0.0030 | 0.5535 | 0.5442 | 0.4755 | 0.2269 |
| 0.00305 | 0.5508 | 0.5292 | 0.4662 | 0.2134 |
| 0.00310 | 0.5486 | 0.5130 | 0.4584 | 0.2070 |
| 0.00320 | 0.5460 | 0.4792 | 0.4512 | 0.1954 |
| 0.0040 | 0.5044 | 0.2779 | 0.3586 | 0.1037 |

Interpretation: increasing strength produces monotonic forgetting gains, but the
utility drop becomes material past roughly `0.00305`.

## Ablation Findings

### Kronecker vs. diagonal

Matched diagonal controls preserve utility but forget much less.

| Run | Model Utility | Forget Q/A Prob | Forget Q/A ROUGE | Extraction Strength |
|---|---:|---:|---:|---:|
| Kron, strength 0.0032 | 0.5460 | 0.4792 | 0.4512 | 0.1954 |
| Diagonal, strength 0.0032 | 0.5728 | 0.7886 | 0.6519 | 0.4480 |

This supports the plan's A1 hypothesis: the Kronecker structure is load-bearing
in the current setup.

### Retain whitening vs. forget-only

The forget-only control is effectively inactive at matched strength.

| Run | Model Utility | Forget Q/A Prob | Forget Q/A ROUGE | Extraction Strength |
|---|---:|---:|---:|---:|
| Kron + retain whitening, strength 0.0032 | 0.5460 | 0.4792 | 0.4512 | 0.1954 |
| Kron forget-only, strength 0.0032 | 0.5981 | 0.8808 | 0.8195 | 0.7099 |

This supports the plan's A2 hypothesis: the contrastive forget-vs-retain
geometry is necessary for the observed behavior.

### Layer target

The strongest target so far is `mlp.down_proj`.

At matched rank/strength, `gate_proj`, `up_proj`, and `self_attn.o_proj` were
weaker or damaged utility more sharply. This makes `down_proj` the default
target for the next pass.

## Transfer Probes

Transfer to other TOFU forget splits works directionally but is weaker at
matched settings.

| Split | Config | Model Utility | Forget Q/A Prob | Forget Q/A ROUGE | Extraction Strength |
|---|---|---:|---:|---:|---:|
| forget05 | R2 M1 B2 S0.003 kron retain | 0.5644 | 0.6778 | 0.5846 | 0.3727 |
| forget01 | R2 M1 B2 S0.003 kron retain | 0.5695 | 0.6746 | 0.5959 | 0.4355 |

These results are usable as early evidence, but the failed `forget05_r2` Stage 3
block should be rerun before treating transfer as complete.

## Operational Notes

The runs required several environment fixes:

- Used local writable cache directories for Triton, Torch extensions,
  HuggingFace, and XDG cache.
- Overrode attention implementation to `sdpa`; default `flash_attention_2`
  was not viable in this environment.
- Evaluated in `float32`; `bf16` evaluation hit an unsupported scalar type.
- Added `CUDA_DEVICE_ORDER=PCI_BUS_ID` after a CUDA ordinal mismatch caused the
  first Stage 3 GPU 0 attempt to OOM.
- Used GPU 0 and GPU 2 for parallel Stage 3 work. GPUs 1, 3, and 4 were not
  used because they showed high memory use and/or high utilization.

## Current Conclusions

1. K-FORGE is implemented end to end in OpenUnlearning and can run full TOFU
   train/eval cycles.
2. The first TOFU/Llama-3.2-1B sweep finds a useful utility-preserving
   frontier around rank 2, one `down_proj` target, two calibration batches, and
   strength `0.0030-0.0031`.
3. Kronecker factors materially outperform the diagonal ablation at matched
   strength.
4. Retain whitening is essential; the forget-only variant does not produce
   meaningful forgetting in the current setup.
5. The current evidence is strong enough for an initial report, but not yet
   enough for the full PLAN.md claim across TOFU/MUSE/WMDP and larger models.

## Recommended Next Steps

1. Rerun the failed `forget05_r2` Stage 3 transfer block.
2. Repeat the top `forget10` configs with multiple seeds or repeated
   calibrations to estimate variance.
3. Add a compact baseline table against NPO, SimNPO, RMU, GradDiff, and SSD
   within OpenUnlearning.
4. Run MUSE or WMDP next to test whether the TOFU frontier generalizes.
5. Add a small robustness pass for relearning or extraction after unlearning.
