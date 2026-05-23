# K-FORGE Current Experimental Findings

Date: 2026-05-14 08:21 UTC

Status note added 2026-05-15: these findings were produced with the historical
`legacy_v1` implementation before the fp32-edit, token-masking, Fisher-scaling,
and damping-floor corrections. Treat every quantitative claim below as
provisional until the corrected `legacy_v1` and `wiener_v2` reruns finish.

## Executive Summary

The current evidence supports the revised paper framing:

**K-FORGE is a closed-form Kronecker-Fisher initializer that improves early and
mid-budget NPO/SimNPO unlearning, especially on TOFU forget01/05, but it is not
a standalone SOTA unlearner and it does not improve RMU's final utility tradeoff.**

The result is now stronger than the Week 1 story:

- `forget10`: K-FORGE-init improves NPO and SimNPO forgetting at matched step
  budgets across 3 seeds.
- `forget05`: K-FORGE-init transfers cleanly; NPO gains are strong and SimNPO
  gains are consistent.
- `forget01`: early completed rows are very strong; K-FORGE-init makes
  SimNPO/NPO forget much faster at `50` and `100` steps.
- RMU: K-FORGE lowers forget metrics but costs too much utility, so RMU should
  be a baseline/negative result rather than the headline.

Current publishability assessment: promising for a Findings-style paper after
finishing `forget01 S250`, the 3B smoke, and at least one MUSE split or a
robustness audit. Not yet a locked EMNLP main-track submission.

## Completed Artifacts

Main analysis artifacts:

- `saves/analysis/week2/week2_runs.csv`
- `saves/analysis/week2/week2_aggregate.csv`
- `saves/analysis/week2/week2_summary.md`
- `saves/analysis/week2/forget10_pareto_scatter.png`
- `saves/analysis/week2/forget10_npo_forget_prob.png`
- `saves/analysis/week2/forget10_simnpo_forget_prob.png`
- `saves/analysis/week2/forget10_rmu_forget_prob.png`
- `docs/kforge_week2_report.md`

As of this report:

- Total TOFU summary files: 313.
- Week-2/Week-3 init summaries: 146.
- Completed `forget01` init summaries: 26.

## One-Shot K-FORGE Findings

The best one-shot `forget10` utility-preserving point remains:

`KFORGE_TOFU_forget10_R2_M1_B2_S0p00305_kron_retain_stage3down`

| Run | Utility | Forget Prob | Forget ROUGE | ES |
|---|---:|---:|---:|---:|
| Base model | 0.5992 | 0.8805 | 0.8201 | 0.7063 |
| K-FORGE one-shot best | 0.5508 | 0.5292 | 0.4662 | 0.2134 |

Main interpretation:

- The one-shot edit finds a meaningful forget direction.
- It is not competitive with strong iterative baselines such as RMU.
- The strength cliff is real signal: K-FORGE is a good initialization direction,
  not the final optimization destination.

Important ablations:

- Kronecker factors are load-bearing: diagonal Fisher controls preserve utility
  but forget much less.
- Retain whitening is load-bearing: forget-only controls are nearly inactive.
- Larger calibration batches smooth the one-shot utility cliff but weaken
  forgetting at fixed strength.
- Adaptive damping as currently parameterized is not useful: large coefficients
  are near no-op, small coefficients collapse utility.

## Forget10: Init Experiment

### NPO

| Steps | Scratch Utility | K-FORGE Utility | Scratch Forget Prob | K-FORGE Forget Prob | Scratch ES | K-FORGE ES |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.3959 | 0.5201 | 0.2790 | 0.1453 | 0.0943 | 0.0883 |
| 100 | 0.3261 | 0.5490 | 0.2283 | 0.1217 | 0.0932 | 0.0926 |
| 250 | 0.5753 | 0.5739 | 0.0980 | 0.0711 | 0.1162 | 0.0979 |
| 500 | 0.5932 | 0.5749 | 0.0528 | 0.0452 | 0.1148 | 0.0925 |
| 1000 | 0.5897 | 0.5769 | 0.0307 | 0.0270 | 0.1037 | 0.0871 |

Conclusion: K-FORGE clearly accelerates early NPO. At `50/100` steps it improves
both utility and forget probability. At longer budgets scratch recovers utility,
but K-FORGE retains lower forget probability and extraction strength.

### SimNPO

| Steps | Scratch Utility | K-FORGE Utility | Scratch Forget Prob | K-FORGE Forget Prob | Scratch ES | K-FORGE ES |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.5974 | 0.5804 | 0.8696 | 0.6762 | 0.6492 | 0.3374 |
| 100 | 0.5974 | 0.5857 | 0.8519 | 0.6582 | 0.5896 | 0.3135 |
| 250 | 0.5963 | 0.5905 | 0.7526 | 0.5486 | 0.4004 | 0.2307 |
| 500 | 0.5939 | 0.5894 | 0.5209 | 0.4103 | 0.2212 | 0.1759 |
| 1000 | 0.5881 | 0.5877 | 0.2963 | 0.2653 | 0.1579 | 0.1346 |

Conclusion: K-FORGE consistently improves SimNPO forgetting. By `1000` steps,
the utility gap is nearly gone while forget probability and ES remain better.

### RMU

| Steps | Scratch Utility | K-FORGE Utility | Scratch Forget Prob | K-FORGE Forget Prob | Scratch ES | K-FORGE ES |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.5360 | 0.5086 | 0.4866 | 0.2675 | 0.1616 | 0.0828 |
| 100 | 0.5628 | 0.5361 | 0.1937 | 0.1024 | 0.0734 | 0.0523 |
| 250 | 0.5835 | 0.5671 | 0.0076 | 0.0041 | 0.0351 | 0.0326 |
| 500 | 0.5890 | 0.5740 | 0.0048 | 0.0021 | 0.0357 | 0.0325 |

Conclusion: K-FORGE makes RMU forget more aggressively, but with a consistent
utility penalty. RMU-from-scratch should remain the dominant baseline.

## Forget05 Transfer

### NPO

| Steps | Scratch Utility | K-FORGE Utility | Scratch Forget Prob | K-FORGE Forget Prob | Scratch ES | K-FORGE ES |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.4621 | 0.5450 | 0.2941 | 0.0940 | 0.1114 | 0.0822 |
| 100 | 0.5187 | 0.5717 | 0.1834 | 0.0670 | 0.0909 | 0.0816 |
| 250 | 0.5887 | 0.5821 | 0.0792 | 0.0417 | 0.0821 | 0.0802 |

Conclusion: this is one of the cleanest findings. K-FORGE gives NPO a much
better start on `forget05`, with clear gains at 50 and 100 steps.

### SimNPO

| Steps | Scratch Utility | K-FORGE Utility | Scratch Forget Prob | K-FORGE Forget Prob | Scratch ES | K-FORGE ES |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.5976 | 0.5823 | 0.8580 | 0.5679 | 0.6064 | 0.2924 |
| 100 | 0.5967 | 0.5836 | 0.8180 | 0.5103 | 0.5067 | 0.2527 |
| 250 | 0.5962 | 0.5864 | 0.6186 | 0.3929 | 0.2662 | 0.1850 |

Conclusion: K-FORGE transfers strongly to SimNPO on `forget05`, with a modest
utility cost and large reductions in forget probability and ES.

## Forget01 Transfer, Partial

`forget01 S50/S100` is complete for NPO and SimNPO. `S250` is currently running.

### NPO

| Steps | Init | n | Utility | Forget Prob | Forget ROUGE | ES |
|---:|---|---:|---:|---:|---:|---:|
| 50 | scratch | 3 | 0.5898 | 0.0843 | 0.3364 | 0.0859 |
| 50 | K-FORGE | 3 | 0.5790 | 0.0118 | 0.1767 | 0.0304 |
| 100 | scratch | 3 | 0.5961 | 0.0574 | 0.3313 | 0.0807 |
| 100 | K-FORGE | 3 | 0.5846 | 0.0129 | 0.1783 | 0.0314 |
| 250 | scratch | 1 | 0.5983 | 0.0413 | 0.3208 | 0.0715 |

### SimNPO

| Steps | Init | n | Utility | Forget Prob | Forget ROUGE | ES |
|---:|---|---:|---:|---:|---:|---:|
| 50 | scratch | 3 | 0.5953 | 0.7199 | 0.6158 | 0.3122 |
| 50 | K-FORGE | 3 | 0.5791 | 0.0154 | 0.1746 | 0.0335 |
| 100 | scratch | 3 | 0.5967 | 0.5606 | 0.4977 | 0.2095 |
| 100 | K-FORGE | 3 | 0.5844 | 0.0247 | 0.2164 | 0.0335 |
| 250 | scratch | 1 | 0.5997 | 0.3590 | 0.4410 | 0.1423 |

Conclusion: `forget01` is currently the strongest transfer evidence. K-FORGE
massively reduces forget probability and ES at `50/100` steps. The utility cost
is real but not catastrophic. We need the remaining `S250` K-FORGE rows before
finalizing the claim.

## Spectrum Diagnostics

The per-layer spectrum dump produced 112 module summaries. The strongest
forget/retain spectrum leaders include:

- layer 0 `self_attn_q_proj`
- layer 15 `mlp_down_proj`
- layer 0 `self_attn_k_proj`
- several early-layer `self_attn_q_proj` modules

Interpretation: the spectrum is a plausible audit signal, but it is not yet a
standalone publishable result. It can be used as a diagnostic appendix unless we
later show it predicts robustness or layer choice.

## Current Running Experiments

As of 2026-05-14 08:21 UTC:

- Active:
  - GPU 1: `forget01 NPO S250`.
  - GPU 2: `forget01 SimNPO S250`.
- Completed:
  - `forget01 S50/S100` for NPO and SimNPO.
  - `forget05` NPO/SimNPO transfer grid.
  - `forget10` NPO/SimNPO/RMU grid.
- Queued:
  - Llama-3.2-3B `forget10` smoke, behind GPU 1's `forget01 NPO` queue.
- Not started:
  - MUSE.
  - Relearning attack.
  - Quantization-revert attack.
  - Paired projection variant.

## Publishability Assessment

### What is already strong

The core empirical pattern is now robust on TOFU:

- K-FORGE + NPO works on `forget10`, `forget05`, and early `forget01`.
- K-FORGE + SimNPO works on `forget10`, `forget05`, and early `forget01`.
- The effect appears across 3 seeds for completed rows.
- The ablations support that the Kronecker/retain structure is load-bearing.

### What is still missing

For EMNLP main-track strength, we still need:

1. Finish `forget01 S250`.
2. Complete the Llama-3.2-3B smoke.
3. Add at least one non-TOFU benchmark, ideally MUSE News or MUSE Books.
4. Add robustness evidence: relearning or quantization-revert.
5. Convert “better at matched steps” into a clean “steps-to-target” plot.

### Current submission positioning

As of now:

- Workshop/arXiv report: yes.
- Findings-style paper: plausible after `forget01` and 3B smoke complete.
- EMNLP main-track: still needs MUSE or robustness evidence.

The recommended headline remains:

**K-FORGE is a closed-form Kronecker-Fisher initializer that accelerates
preference-based LLM unlearning, rather than a replacement for iterative
unlearning.**

## Recommended Next Actions

1. Let `forget01 S250` finish.
2. Let the queued 3B smoke run.
3. Re-run the analysis script after those finish and update aggregate plots.
4. Start MUSE News with NPO/SimNPO scratch vs. K-FORGE init.
5. Pick 2-3 representative checkpoints for relearning and quantization-revert
   audits.
