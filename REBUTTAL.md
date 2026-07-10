# K-FORGE Rebuttal Draft

This draft is organized by reviewer. The main strategy is to sharpen the claim:

> K-FORGE is not a new unlearning loss and not a standalone robustness guarantee. It is a one-time Fisher-guided initialization that improves the early forget-probability trajectory of fixed downstream preference-based optimizers.

We should avoid saying "better unlearning" without qualification. The strongest supported claim is "lower Forget Q/A Probability at matched optimization budgets, usually with comparable or better utility."

## Common Notation for Added Results

All TOFU numbers below are means over three seeds unless stated otherwise. We report standard deviations in the revised paper tables, but omit them here for readability.

For a metric value $x_i$ over $n$ seeds,

$$
\bar{x}=\frac{1}{n}\sum_{i=1}^n x_i,\qquad
s=\sqrt{\frac{1}{n-1}\sum_{i=1}^n (x_i-\bar{x})^2}.
$$

For the compute-matched comparison requested by R1, we charge K-FORGE for its measured one-time setup time. Let $t_{\rm KF}$ be the wall-clock time to estimate Fisher factors, compute the edit, and write the initialized checkpoint. Let $\bar{t}_{\rm step}$ be the measured scratch training time per optimizer step. We compare a K-FORGE-initialized run of $T$ steps against a scratch run with

$$
T_{\rm cm}=T+\left\lceil \frac{t_{\rm KF}}{\bar{t}_{\rm step}}\right\rceil
$$

steps. In our Llama-3.2-1B setup this rounds to scratch budgets $55,105,255$ for K-FORGE budgets $50,100,250$.

We report relative forget-probability reduction and utility change as

$$
\operatorname{RelRed}_{\rm FP}
=
\frac{\operatorname{FP}_{\rm scratch}(T_{\rm cm})-\operatorname{FP}_{\rm KF}(T)}
{\operatorname{FP}_{\rm scratch}(T_{\rm cm})}\times 100\%,
\qquad
\Delta U
=
U_{\rm KF}(T)-U_{\rm scratch}(T_{\rm cm}).
$$

For relearning robustness, we use

$$
\Delta_{\rm relearn}(m)=m_{\rm post}-m_{\rm pre},\qquad
\operatorname{PostGap}(m)=
\frac{m_{\rm scratch,post}-m_{\rm KF,post}}{m_{\rm scratch,post}}.
$$

For forgetting metrics such as Forget Q/A Probability, extraction, and forget ROUGE, lower is better. For model utility and retain ROUGE, higher is better.

## Reviewer 1

### Concern: K-FORGE may be too expensive; compare by FLOPs and wall-clock, not only by training steps.

We agree that step-matched comparisons alone are incomplete. K-FORGE is a one-time curvature computation, so the relevant question is whether the initialization still helps after charging the method for Fisher estimation, factorizations, and checkpoint writing. We have added both an overhead table and a compute-matched comparison.

For one edited `mlp.down_proj` layer with $B_{\rm cal}=32$ forget and retain batches, the measured setup cost is modest relative to even a 50-step run:

| Model | Edited W | Calibration | Time, calib./total | FLOPs | vs. 50-step NPO/SimNPO |
|---|---:|---:|---:|---:|---:|
| Llama-3.2-1B | 2048 x 8192 | 512 ex. / 18.6k tok. | 18 / 61 s | $1.44\times 10^{14}$ | 8.9% / 9.7% |
| Llama-3.2-3B | 3072 x 8192 | 512 ex. / 18.6k tok. | 51 / 104 s | $3.65\times 10^{14}$ | 10.0% / 10.9% |

The cost scales with the edited matrices and calibration size, but it is not paid at every downstream step. It is also reusable across downstream step budgets and optimizer seeds for the same edit configuration.

After charging this setup cost, K-FORGE still improves the 1B TOFU `forget10` runs. We compare K-FORGE at $T\in\{50,100,250\}$ to scratch at $T_{\rm cm}\in\{55,105,255\}$:

| Method | Budget | Scratch FP | K-FORGE FP | Rel. FP red. | Scratch U | K-FORGE U | Delta U |
|---|---:|---:|---:|---:|---:|---:|---:|
| NPO | 50 vs. 55 | 0.0783 | **0.0454** | 42.0% | 0.5224 | **0.5738** | +0.0514 |
| NPO | 100 vs. 105 | 0.0411 | **0.0308** | 25.1% | 0.5712 | **0.5932** | +0.0220 |
| NPO | 250 vs. 255 | 0.0276 | **0.0223** | 19.2% | 0.5925 | **0.6034** | +0.0109 |
| SimNPO | 50 vs. 55 | 0.6462 | **0.5341** | 17.3% | **0.5795** | 0.5712 | -0.0083 |
| SimNPO | 100 vs. 105 | 0.5030 | **0.4066** | 19.2% | **0.5872** | 0.5817 | -0.0055 |
| SimNPO | 250 vs. 255 | 0.3290 | **0.2719** | 17.4% | **0.5957** | 0.5917 | -0.0040 |

Thus, the conclusion does not rely on giving K-FORGE a free setup. For NPO, the compute-matched comparison improves both forget probability and utility. For SimNPO, it consistently lowers forget probability with a small utility cost.

### Concern: model-family scope is narrow.

We agree this was a real limitation in the submitted version. We expanded the Llama-3.2-3B sanity check from two seeds to three seeds and include the full 50/100/250-step comparison:

| Method | Budget | Scratch FP | K-FORGE FP | Rel. FP red. | Scratch U | K-FORGE U | Delta U |
|---|---:|---:|---:|---:|---:|---:|---:|
| NPO | 50 | 0.0863 | **0.0649** | 24.8% | 0.5895 | **0.6393** | +0.0498 |
| NPO | 100 | **0.0342** | 0.0388 | -13.5% | 0.6480 | **0.6700** | +0.0220 |
| NPO | 250 | **0.0267** | 0.0291 | -9.0% | **0.6649** | 0.6624 | -0.0025 |
| SimNPO | 50 | 0.6911 | **0.5829** | 15.7% | 0.6365 | **0.6558** | +0.0193 |
| SimNPO | 100 | 0.5168 | **0.4550** | 12.0% | 0.6466 | **0.6546** | +0.0080 |
| SimNPO | 250 | 0.3342 | **0.3145** | 5.9% | 0.6617 | **0.6656** | +0.0039 |

This result is more nuanced than the 1B result: SimNPO transfers cleanly at 3B, while NPO mainly benefits at early steps. We will reflect that in the claim rather than presenting this as broad scaling evidence.

More importantly, we completed a non-Llama evaluation on Gemma-3-1B and then added a held-out fourth seed at 100 steps. We selected the K-FORGE strength before downstream training using a one-shot grid: among the points whose utility drop from the base model was at most 0.01, we chose the point with the lowest Forget Q/A Probability. This selected $\alpha=0.8$; $\alpha=1.0$ was excluded because its utility drop was 0.0111. The matched downstream results are:

| Method | Budget | $n$ | Scratch FP | K-FORGE FP | Rel. FP red. | Scratch U | K-FORGE U | Delta U |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NPO | 50 | 3 | 0.07436 | **0.07434** | 0.02% | 0.34903 | **0.35804** | +0.00900 |
| NPO | 100 | 4 | 0.06557 | **0.05328** | **18.74%** | 0.40062 | **0.40122** | +0.00060 |
| SimNPO | 50 | 3 | 0.27334 | **0.27005** | 1.20% | **0.40401** | 0.40230 | -0.00171 |
| SimNPO | 100 | 4 | 0.27240 | **0.26930** | 1.14% | **0.41050** | 0.40896 | -0.00154 |

The held-out seed independently confirms the probability direction for both optimizers. For NPO, its Forget Q/A Probability changes from 0.06386 to 0.05762 with a utility change of -0.00645; for SimNPO, it changes from 0.27212 to 0.26907 with a utility change of -0.00312. Both satisfy the prespecified utility margin of -0.01.

The strongest aggregate Gemma result is NPO at 100 steps: over four seeds, K-FORGE lowers Forget Q/A Probability by 18.7% and extraction by 40.0% (0.0550 to 0.0330) while leaving mean utility effectively unchanged. Forget ROUGE moves in the opposite direction (0.2980 to 0.3647), so we do not present this as uniform improvement across all forgetting metrics. SimNPO gives a smaller but highly consistent 1.14% probability reduction, together with lower extraction (0.1265 to 0.1219), lower Forget ROUGE (0.3993 to 0.3948), and a utility change of only -0.00154. Two-sided paired tests on Forget Q/A Probability give $p=0.0115$ for NPO and $p=8.8\times10^{-7}$ for SimNPO at 100 steps. Given the small sample, we treat the NPO test as descriptive; the SimNPO result remains below the $p<0.001$ exploratory threshold after adding the held-out seed.

The completed Qwen2.5-1.5B pilot was near-neutral: SimNPO Forget Q/A Probability changed by less than 0.06% relatively at both budgets. We therefore use Gemma as positive non-Llama evidence but do not claim universal transfer across architectures.

### Concern: Algorithm 1 readability.

We will split the combined SVD and update lines in Algorithm 1, specifically the lines corresponding to the cross-Cholesky maps, the two SVDs, and the target/rank truncation. This is a presentation-only change; it does not alter the method.

## Reviewer 2

### Concern: scope is narrow and some claims sound broader than the evidence.

We agree and will narrow the claim. The paper should not claim that K-FORGE is uniformly "better unlearning" across all forgetting metrics. The supported claim is:

> K-FORGE improves the early forget-probability trajectory of fixed downstream NPO/SimNPO optimizers, with comparable utility in the tested regimes.

This change also addresses the extraction metric concern. In several settings, K-FORGE improves Forget Q/A Probability while another forgetting metric is flat or worse. The new Gemma result makes this boundary explicit: NPO at 100 steps improves probability and extraction but worsens Forget ROUGE, whereas SimNPO improves probability, extraction, and Forget ROUGE with a utility change below 0.002 in magnitude. We will report these metrics separately instead of treating them as interchangeable.

The compute-matched table above is the main quantitative support for this narrower claim. It shows that K-FORGE improves all compute-matched 1B NPO forget-probability comparisons and all compute-matched SimNPO forget-probability comparisons, while the utility effect differs by optimizer.

### Concern: robustness audits were written in future tense and looked incomplete.

This criticism is fair. We completed a relearning audit and will remove future-tense language from the main text. The audit takes matched post-unlearning models and then performs one epoch of supervised fine-tuning on held-out forget examples. The result is not that K-FORGE is recovery-proof; relearning increases forget-set behavior for both methods. The result is that the recovered forget-set behavior remains substantially lower for the K-FORGE-initialized run:

| Init | FP pre | FP post | Delta FP | Extraction pre | Extraction post |
|---|---:|---:|---:|---:|---:|
| Scratch | 0.2795 | 0.7135 | +0.4340 | 0.0938 | 0.2972 |
| K-FORGE | **0.0980** | **0.4154** | **+0.3174** | **0.0608** | **0.1275** |

After relearning, K-FORGE has 41.8% lower Forget Q/A Probability and 57.1% lower extraction than scratch:

$$
\operatorname{PostGap}_{\rm FP}
=
\frac{0.7135-0.4154}{0.7135}
=41.8\%,
\qquad
\operatorname{PostGap}_{\rm Ext}
=
\frac{0.2972-0.1275}{0.2972}
=57.1\%.
$$

The limitation is that post-relearning utility is lower for K-FORGE than for scratch in this audit. We will report this as a robustness result with a clear trade-off, not as a complete solution to recovery attacks. We will also remove or clearly label quantization-revert as out of scope unless the corresponding audit is completed before the final response.

### Concern: no usable software.

We will make the code release easier to reproduce by adding the exact run scripts used for the main tables, the compute-overhead script, and the plotting/aggregation scripts. The revised reproducibility material will include fixed random seeds, calibration size, edited layer, K-FORGE rank/strength, and evaluation dtype settings.

## Reviewer 3

### Concern: robustness against relearning or recovery attacks is important.

We agree. We completed the relearning audit described above. The short version is:

| Init | FP pre | FP post | Extraction pre | Extraction post |
|---|---:|---:|---:|---:|
| Scratch | 0.2795 | 0.7135 | 0.0938 | 0.2972 |
| K-FORGE | **0.0980** | **0.4154** | **0.0608** | **0.1275** |

This supports a limited robustness statement: K-FORGE is not immune to relearning, but the forget-set behavior after relearning remains lower than for the scratch-initialized baseline under the same audit. We will present it exactly this way.

### Concern: evaluation is mainly TOFU; add another benchmark, preferably MUSE or WMDP.

We added a MUSE-News pilot with Llama-2-7B and SimNPO over three seeds. This gives evidence beyond TOFU, but the result is mixed, so we will use it to define the boundary of the claim rather than as a headline win.

| Budget | Init | Extraction ↓ | Forget KnowMem ROUGE ↓ | Forget VerbMem ROUGE ↓ | Retain KnowMem ROUGE ↑ |
|---:|---|---:|---:|---:|---:|
| 50 | Scratch | **0.3075** | 0.6301 | 0.5741 | **0.5338** |
| 50 | K-FORGE | 0.3149 | **0.6235** | **0.5639** | 0.5321 |
| 100 | Scratch | **0.3034** | 0.6286 | 0.5727 | **0.5318** |
| 100 | K-FORGE | 0.3097 | **0.6278** | **0.5623** | 0.5216 |

K-FORGE reduces MUSE forget ROUGE metrics, especially verbatim memorization, but extraction and retain quality are mixed. Therefore, the right claim is not "K-FORGE solves MUSE"; it is that the Fisher-guided initializer has measurable transfer beyond TOFU, while benchmark-robust unlearning remains an open limitation.

We additionally completed the same three-seed comparison on the MUSE-Books domain:

| Budget | Init | Extraction ↓ | Forget KnowMem ROUGE ↓ | Forget VerbMem ROUGE ↓ | Retain KnowMem ROUGE ↑ |
|---:|---|---:|---:|---:|---:|
| 50 | Scratch | 0.9110 | **0.4106** | **0.9941** | **0.6488** |
| 50 | K-FORGE | **0.9109** | 0.4265 | 0.9958 | 0.6451 |
| 100 | Scratch | **0.8519** | 0.3655 | **0.9554** | **0.6099** |
| 100 | K-FORGE | 0.8692 | **0.3567** | 0.9632 | 0.5860 |

The Books result is mixed rather than a headline win. At 100 steps K-FORGE improves Forget KnowMem ROUGE from 0.3655 to 0.3567, but extraction, VerbMem, and retain quality worsen; at 50 steps it provides no meaningful advantage. Together, the two MUSE domains show that the initializer can transfer individual forgetting gains beyond TOFU, but not yet a uniformly better benchmark-level trade-off.

### Concern: narrow empirical scope.

We will revise the limitations accordingly. The final paper should say:

1. The strongest evidence remains TOFU `forget10` on Llama-3.2-1B.
2. The 3B results support transfer most clearly for SimNPO; NPO transfer is strongest at early steps.
3. Gemma-3-1B provides positive non-Llama evidence, especially for NPO at 100 steps, while the completed Qwen pilot is near-neutral.
4. MUSE-News shows partial transfer on forget ROUGE, whereas MUSE-Books is mixed and does not improve the overall forget-retain trade-off.

## Proposed Claim Revision for the Paper

Old phrasing to avoid:

> K-FORGE improves unlearning.

Better phrasing:

> K-FORGE improves the forget-probability trajectory of fixed downstream preference-based unlearning methods. On TOFU `forget10`, the improvement remains after charging the one-time Fisher-estimation overhead in wall-clock terms. A held-out fourth seed confirms transfer to Gemma-3-1B: at 100 steps, K-FORGE reduces NPO Forget Q/A Probability by 18.7% and extraction by 40.0% at unchanged mean utility, and gives a smaller but highly consistent SimNPO gain. Results on Qwen and the two MUSE domains are more mixed, so we do not claim universal model- or benchmark-level transfer.

This is more defensible and directly answers the main reviewer concerns without overclaiming.
