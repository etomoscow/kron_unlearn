# K-FORGE Week 2 Report

Date: 2026-05-13

Status note added 2026-05-15: the results in this report predate the
fp32-edit, calibration, damping-floor, and theorem-aligned `wiener_v2`
corrections. They are historical diagnostics, not paper-ready headline
numbers, until the corrected reruns replace them.

Update: a consolidated current findings report, including partial `forget01`
results from 2026-05-14, is available in
[`docs/kforge_current_findings_2026-05-14.md`](kforge_current_findings_2026-05-14.md).

## Scope

This report summarizes the Week 2 init experiments for the current paper
framing:

**K-FORGE: A Closed-Form Kronecker-Fisher Initialization that Accelerates
Second-Order LLM Unlearning.**

The completed batch contains 120 TOFU eval summaries:

- `forget10`: NPO, SimNPO, and RMU, scratch vs. K-FORGE init, 3 seeds.
- `forget05`: NPO and SimNPO, scratch vs. K-FORGE init, 3 seeds.
- Step budgets: `50 100 250 500 1000` where scheduled.

Generated artifacts:

- `saves/analysis/week2/week2_runs.csv`
- `saves/analysis/week2/week2_aggregate.csv`
- `saves/analysis/week2/week2_summary.md`
- `saves/analysis/week2/forget10_pareto_scatter.png`
- `saves/analysis/week2/forget10_npo_forget_prob.png`
- `saves/analysis/week2/forget10_simnpo_forget_prob.png`
- `saves/analysis/week2/forget10_rmu_forget_prob.png`

## Headline Finding

K-FORGE is not a one-shot SOTA unlearner, but it is a useful initializer for
preference-style unlearning.

The strongest evidence is:

- NPO, short budgets: K-FORGE improves both utility and forgetting.
- SimNPO, all tested budgets: K-FORGE consistently lowers forget probability
  and extraction strength, usually with a small utility cost.
- RMU: K-FORGE does not improve the final tradeoff; RMU from scratch keeps
  higher utility once enough steps are allowed.

This supports the paper story: the closed-form edit gives a good direction, but
the magnitude should be refined by an iterative optimizer.

## Forget10 Results

### NPO

At low budgets, K-FORGE is a clear accelerator:

| Steps | Scratch Utility | K-FORGE Utility | Scratch Forget Prob | K-FORGE Forget Prob |
|---:|---:|---:|---:|---:|
| 50 | 0.3959 | 0.5201 | 0.2790 | 0.1453 |
| 100 | 0.3261 | 0.5490 | 0.2283 | 0.1217 |
| 250 | 0.5753 | 0.5739 | 0.0980 | 0.0711 |
| 500 | 0.5932 | 0.5749 | 0.0528 | 0.0452 |
| 1000 | 0.5897 | 0.5769 | 0.0307 | 0.0270 |

Interpretation: K-FORGE dominates at 50/100 steps, remains better on forgetting
at 250 steps with matched utility, and becomes a forget-strength/utility
tradeoff at 500/1000 steps.

### SimNPO

K-FORGE consistently improves forgetting:

| Steps | Scratch Utility | K-FORGE Utility | Scratch Forget Prob | K-FORGE Forget Prob |
|---:|---:|---:|---:|---:|
| 50 | 0.5974 | 0.5804 | 0.8696 | 0.6762 |
| 100 | 0.5974 | 0.5857 | 0.8519 | 0.6582 |
| 250 | 0.5963 | 0.5905 | 0.7526 | 0.5486 |
| 500 | 0.5939 | 0.5894 | 0.5209 | 0.4103 |
| 1000 | 0.5881 | 0.5877 | 0.2963 | 0.2653 |

Interpretation: by 1000 steps, the utility gap nearly disappears while
K-FORGE still reduces forget probability and extraction strength.

### RMU

K-FORGE makes RMU forget faster, but not better at the final tradeoff:

| Steps | Scratch Utility | K-FORGE Utility | Scratch Forget Prob | K-FORGE Forget Prob |
|---:|---:|---:|---:|---:|
| 50 | 0.5360 | 0.5086 | 0.4866 | 0.2675 |
| 100 | 0.5628 | 0.5361 | 0.1937 | 0.1024 |
| 250 | 0.5835 | 0.5671 | 0.0076 | 0.0041 |
| 500 | 0.5890 | 0.5740 | 0.0048 | 0.0021 |

Interpretation: RMU already finds very strong forgetting; K-FORGE lowers forget
probability further but costs too much utility for the headline claim.

## Forget05 Transfer

The transfer result is strong for NPO:

| Trainer | Steps | Scratch Utility | K-FORGE Utility | Scratch Forget Prob | K-FORGE Forget Prob |
|---|---:|---:|---:|---:|---:|
| NPO | 50 | 0.4621 | 0.5450 | 0.2941 | 0.0940 |
| NPO | 100 | 0.5187 | 0.5717 | 0.1834 | 0.0670 |
| NPO | 250 | 0.5887 | 0.5821 | 0.0792 | 0.0417 |
| SimNPO | 50 | 0.5976 | 0.5823 | 0.8580 | 0.5679 |
| SimNPO | 100 | 0.5967 | 0.5836 | 0.8180 | 0.5103 |
| SimNPO | 250 | 0.5962 | 0.5864 | 0.6186 | 0.3929 |

Interpretation: K-FORGE transfers beyond `forget10`. The NPO acceleration is
especially clean at 50 and 100 steps.

## Paper Conclusions

1. The one-shot K-FORGE edit should be framed as an initializer, not the final
   unlearning destination.
2. The strongest main-paper result is K-FORGE + NPO/SimNPO versus the same
   optimizer from scratch at matched steps and seeds.
3. The RMU result should be reported honestly as mixed: better forgetting, worse
   utility.
4. The `forget05` transfer result strengthens the claim that K-FORGE is not only
   tuned to one split.
5. The next paper-critical experiments are `forget01`, Llama-3.2-3B, MUSE, and
   robustness audits.

## Recommended Next Experiments

Run these in order:

1. `forget01` transfer for NPO and SimNPO at `50 100 250` steps.
2. Llama-3.2-3B smoke: K-FORGE init plus NPO/SimNPO at `50 100 250`.
3. MUSE News/Books with K-FORGE init plus NPO/SimNPO/RMU.
4. Paired projection variant, only after the current plots are finalized.
5. Relearning and quantization-revert audits for the final selected points.
