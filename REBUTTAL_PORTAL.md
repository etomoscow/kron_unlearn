# Portal-Ready Author Responses

Each section below is a standalone response for the corresponding review thread. The responses intentionally contain no external links.

## Reviewer 1

Thank you for identifying compute alignment and model-family breadth as the two main gaps. We ran both additions with the originally released K-FORGE implementation and unchanged downstream losses; the estimator terminology correction is detailed below.

**FLOP- and wall-clock-matched comparison.** We now charge the complete one-time K-FORGE setup: forget/retain calibration, factor accumulation, Cholesky/SVD computations, edit application, and checkpoint writing. We select the smallest tested scratch budget $T'$ that directly satisfies

$$
t_{\mathrm{train}}^{\mathrm{scratch}}(T')\ge t_{\mathrm{setup}}+t_{\mathrm{train}}^{\mathrm{KF}}(T),
\qquad
F_{\mathrm{train}}^{\mathrm{scratch}}(T')\ge F_{\mathrm{KF}}+F_{\mathrm{train}}^{\mathrm{KF}}(T).
$$

For Llama-3.2-1B, calibration through edit application takes 61.8 s and an estimated $3.62\times10^{14}$ FLOPs for one edited `mlp.down_proj` layer using 512 calibration examples (an estimated 47.7k non-padding prompt-and-answer input tokens; exactly 18,614 loss-bearing factor rows). For a conservative complete wall-clock bound, we measure from creation of the command log to the final checkpoint file: 80.2 s, including model/data startup and serialization, which we round upward to 81 s. Our accounting is

$$
F_{\mathrm{KF}}\simeq6PT_{\mathrm{cal}}+2N_F(m^2+n^2)+10(m^3+n^3),
$$

Here $P$ is model parameter count, $T_{\mathrm{cal}}$ is the calibration input-token count, $N_F$ is the number of loss-bearing factor rows, and $n\times m$ is the edited matrix shape.

With $P=1.236$B, $\tau_f=3042$ and $\tau_r=2924$ tokens per optimizer step,

$$
F_{\mathrm{step}}^{\mathrm{SimNPO}}\simeq6P(\tau_f+\tau_r),\qquad
F_{\mathrm{step}}^{\mathrm{NPO}}\simeq F_{\mathrm{step}}^{\mathrm{SimNPO}}+2P\tau_f.
$$

The setup equals 7.00 NPO or 8.19 SimNPO steps by FLOPs; a dynamic-padding sensitivity calculation raises the largest ratio only to 8.65. The edit is folded into model weights, so later optimizer steps and inference have no added cost. A timing audit found that our earlier 50-step denominator included evaluation, so we discarded that comparison and reran both arms with evaluation disabled under PyTorch 2.9.1+cu130, Transformers 4.51.3, and Accelerate 0.34.2. The strict runs use layer-0 `mlp.down_proj`, rank 2, factor damping $10^{-3}$, retain penalty $10^{-2}$, FP32, and seeds 0/1/2. We reused the previously reported $\alpha=.60$ initialized checkpoint rather than selecting a new edit from these outcomes. Evaluation-free S50 training averages 197.0 s for NPO and 130.3 s for SimNPO; the full 80.2 s setup is therefore 40.7% and 61.5% of S50 training time. Before inspecting final metrics, direct runtime matching selected scratch S73 for NPO and S86 for SimNPO:

| Method | K-FORGE / scratch steps | Wall: scratch / K-FORGE+setup | Scratch FP | K-FORGE FP | Rel. reduction | Scratch / K-FORGE utility |
|---|---:|---:|---:|---:|---:|---:|
| NPO | 50 / 73 | 285.5 / **277.2 s** | 0.0590 | **0.0459** | **22.3%** | 0.5486 / **0.5766** |
| SimNPO | 50 / 86 | 221.6 / **210.5 s** | 0.5527 | **0.5371** | **2.8%** | **0.5832** / 0.5722 |

Every seed satisfies the wall inequality, with minimum margins of 3.56/3.97 s, and both scratch budgets exceed the FLOP charge. NPO improves FP and utility in every seed; FP is descriptive ($p=.00696$), while utility meets our exploratory threshold ($p=4.91\times10^{-4}$). NPO extraction and ROUGE worsen, so this is not metric-wide dominance. SimNPO improves FP and ROUGE in every seed and improves mean extraction, but FP is descriptive ($p=.126$) and utility changes by -0.0109, missing our prespecified -0.01 margin by 0.00093. This is the distinction compute alignment was intended to expose: NPO retains a substantial residual gain after paying for curvature, whereas the SimNPO advantage largely attenuates. Thus strict compute matching supports the NPO result and leaves SimNPO as a weaker trade-off.

At 3B, calibration through edit takes 104 s, complete setup is 145 s, and the analytical setup is $9.29\times10^{14}$ FLOPs; we do not infer an evaluation-free 3B step equivalent without a corresponding timing rerun. A code audit clarified that the executed estimator always streamed K-FAC covariance factors rather than running the Lanczos MFF routine described in the submitted text; we corrected the description, not the checkpoints. The derivation, which assumes SPD Kronecker factor pairs, is unchanged. Executed cost is $\mathcal{O}(N_F(m^2+n^2)+m^3+n^3)$ and remains a limitation for wide matrices or many edited layers. Calibration model passes account for 97.7%/99.1% of estimated setup FLOPs. We charge the full setup to every comparison although one initialized checkpoint can in principle be reused.

**Non-Llama family.** We added Gemma-3-1B; the utility-constrained one-shot rule selected $\alpha=.8$ before downstream training, and seed 3 remained held out until the configuration and rule were fixed. At 100 steps over four seeds, NPO FP changes from 0.06557 to **0.05328** (-18.7%) and extraction from 0.05496 to **0.03298** (-40.0%), while utility is 0.40062/0.40122. Forget ROUGE worsens from 0.29801 to 0.36471, which we report explicitly. SimNPO FP changes from 0.27240 to **0.26930** ($p=8.8\times10^{-7}$), extraction from 0.12651 to **0.12191**, and ROUGE from 0.39930 to **0.39480**, with utility delta -0.00154. All four paired seeds preserve the FP direction for both methods; the held-out seed also satisfies the utility margin.

The Gemma conclusion also survives the setup charge. Comparing K-FORGE S100 against scratch S103/S105 over four seeds gives NPO FP 0.06397 to **0.05328** ($p=0.0048$, treated descriptively) and SimNPO FP 0.27223 to **0.26930** ($p=9.4\times10^{-6}$). NPO extraction improves, but its ROUGE exception remains. We use $p<0.001$ for exploratory inferential claims.

The controls show why this is not merely “more computation.” For Gemma NPO, matched random, diagonal-Fisher, and forget-only initializers have FP 0.06579, 0.06374, and 0.05850 versus 0.05328 for K-FORGE. Weight-SVD reaches 0.05297 but lowers utility to 0.32702 and fails the utility margin. For SimNPO, diagonal Fisher has nearly identical FP and slightly better utility, while K-FORGE is better in extraction and ROUGE. We therefore conclude that retain-aware Fisher structure matters for the NPO result, not that full Kronecker structure wins every optimizer/metric comparison.

We also expanded Llama-3.2-3B to three seeds and all three budgets. SimNPO improves FP, utility, extraction, and ROUGE at 50/100/250 steps; NPO improves mainly at 50 steps. A three-seed Qwen2.5-1.5B pilot was near-neutral. These results support transfer to Gemma and larger Llama for SimNPO, but not universal architecture transfer.

For reproducibility, the revised artifact includes the explicit FLOP calculator, the fixed-seed compute-matched runner, all effective configuration values, CPU algebra tests, and a compact per-seed metric snapshot from which the new tables and paired tests are regenerated without checkpoints.

Finally, we implemented the requested presentation fix by splitting Algorithm 1's cross-map, two-SVD, and target/truncation lines. This changes only readability.

## Reviewer 2

Thank you. We agree with the key distinction in the review: our evidence supports an improved Forget Q/A Probability trajectory, not uniformly “better unlearning.” We directly addressed the scope, incomplete-audit, metric-alignment, and software concerns without changing K-FORGE or the downstream losses.

**Model and benchmark breadth.** We added a held-out-seed Gemma-3-1B evaluation. At 100 steps ($n=4$), NPO scratch/K-FORGE values are FP 0.06557/**0.05328**, utility 0.40062/0.40122, extraction 0.05496/**0.03298**, and ROUGE **0.29801**/0.36471. SimNPO values are FP 0.27240/**0.26930**, utility 0.41050/0.40896, extraction 0.12651/**0.12191**, and ROUGE 0.39930/**0.39480**. All four FP pairs move in the favorable direction. The NPO probability/extraction gain is substantial but its ROUGE result is adverse; SimNPO gives a smaller, highly consistent gain across all forgetting metrics. We retain both outcomes.

We also completed the Llama-3.2-3B matrix over three seeds and all three budgets: SimNPO improves FP, utility, extraction, and ROUGE throughout, whereas NPO transfers mainly at 50 steps. A three-seed Qwen2.5-1.5B pilot is near-neutral and is retained as a null result rather than omitted.

We also evaluated MUSE-News and MUSE-Books with Llama-2-7B and SimNPO over three paired seeds at 50 and 100 steps. At 100 steps:

| Domain | Init. | Extraction $\downarrow$ | Forget KnowMem $\downarrow$ | Forget VerbMem $\downarrow$ | Retain KnowMem $\uparrow$ |
|---|---|---:|---:|---:|---:|
| News | Scratch | **0.3034** | 0.6286 | 0.5727 | **0.5318** |
| News | K-FORGE | 0.3097 | **0.6278** | **0.5623** | 0.5216 |
| Books | Scratch | **0.8519** | 0.3655 | **0.9554** | **0.6099** |
| Books | K-FORGE | 0.8692 | **0.3567** | 0.9632 | 0.5860 |

MUSE-News improves both forgetting-ROUGE metrics but worsens extraction and retain quality; Books improves only KnowMem. A News follow-up fixed its selection rule before downstream training: minimize one-shot KnowMem subject to at most 0.01 one-shot retain drop. The rule selected $\alpha=1.0$; it improves extraction 0.3034 to 0.3012 and VerbMem 0.5727 to 0.5507, but KnowMem and downstream retain quality worsen slightly. We therefore present MUSE as partial metric-level transfer, not benchmark dominance.

**Completed robustness audits.** We replaced the submitted future-tense paragraph with matched-start experiments. For NPO, scratch S100 and K-FORGE S50 begin close in FP (0.0472/0.0519) and utility (0.5683/0.5754). Both receive the same supervised `forget10` attack (AdamW, $10^{-5}$, effective batch 32) for 13 and 39 steps. After 13 steps, scratch/K-FORGE FP is **0.3757**/0.5464, extraction **0.1122**/0.2002, ROUGE **0.3755**/0.4741, and utility 0.4553/**0.4926**. K-FORGE recovers more FP (+0.4945 versus +0.3285, $p=4.7\times10^{-4}$); 39 steps strengthens the same conclusion.

The one-epoch optimizer-transfer audit gives the same boundary for SimNPO. Its matched scratch/K-FORGE starting FP is 0.5452/0.5685. After 13 attack steps, FP is **0.8027**/0.8744 and extraction **0.5393**/0.6942; K-FORGE recovers +0.3059 FP versus +0.2575 for scratch. After 39 steps, both arms are almost fully recovered (FP 0.9778/0.9819), with no distinguishable recovery difference. Gemma repeats the boundary: after one epoch, NPO post-attack FP is 0.2611/0.2947 and SimNPO is 0.3527/0.3530. Thus K-FORGE is an optimization initializer, not a relearning defense.

Quantization is a different intervention. On the matched NPO pair, 8-bit loading changes FP by only +0.00181/+0.00164 (scratch/K-FORGE), and 4-bit by +0.00843/+0.01168; neither shows major recovery. At the ordinary matched SimNPO S50 budget, K-FORGE remains better after 8-bit loading (FP/extraction/ROUGE 0.5610/0.1924/0.4646 versus 0.6936/0.2844/0.5475) and 4-bit loading (0.4725/0.1373/0.4271 versus 0.5629/0.1777/0.4680), with utility differences below 0.005. We claim persistence of an existing optimizer advantage, not recovery immunity. Gemma is optimizer-dependent: NPO's FP gap is erased at 8-bit and reversed at 4-bit, whereas SimNPO retains a small FP advantage at both precisions. Hence quantization persistence is not universal.

In the strict S50 comparison, NPO extraction and ROUGE move against the FP/utility gain. SimNPO improves mean extraction and ROUGE, but narrowly misses the utility margin. This is why we keep the endpoints separate.

**Revised claim.** We replaced broad “better unlearning” wording with:

> K-FORGE improves the early Forget Q/A Probability trajectory of fixed NPO and SimNPO optimizers in the tested TOFU regimes. Direct wall- and FLOP-matching supports the NPO gain; SimNPO retains a smaller probability gain with a slight utility trade-off. K-FORGE does not uniformly improve extraction or Forget ROUGE, transfer to every model/benchmark, or resist active relearning.

**Artifact.** The revised artifact package contains the K-FORGE trainer/config, exact initialization harness, fixed-seed recovery and quantization runners, structured result aggregator, CPU regression tests, and a compact per-seed metric snapshot that reproduces the reviewer-requested aggregate tables and paired tests without checkpoints. The aggregator fails on missing, malformed, unequal, or non-finite paired results. The README states calibration size, edited layer, rank/strength, dtype, seeds, and executable commands. This directly addresses the software/reproducibility concern without adding an external link to this response.

## Reviewer 3

Thank you. We directly completed both requested additions: active recovery audits and an additional benchmark. We did not change K-FORGE or tune an attack after observing its outcome.

**Relearning and quantization.** To avoid confounding recovery with different initial forgetting strength, we compare scratch NPO S100 with K-FORGE-initialized NPO S50. Their pre-attack FP is 0.0472/0.0519 and utility is 0.5683/0.5754. Both receive identical supervised `forget10` fine-tuning for 13 or 39 optimizer steps.

| Attack | FP post S / KF $\downarrow$ | Extraction post S / KF $\downarrow$ | ROUGE post S / KF $\downarrow$ | Utility post S / KF $\uparrow$ |
|---|---:|---:|---:|---:|
| 13 steps | **0.3757** / 0.5464 | **0.1122** / 0.2002 | **0.3755** / 0.4741 | 0.4553 / **0.4926** |
| 39 steps | **0.7298** / 0.9110 | **0.4067** / 0.8123 | **0.6219** / 0.8842 | 0.4932 / **0.5097** |

This is a negative robustness result: K-FORGE recovers more forgotten behavior at the matched starting point. The one-epoch SimNPO transfer audit agrees: K-FORGE/scratch FP recovery is +0.3059/+0.2575, extraction recovery +0.4923/+0.3553, and ROUGE recovery +0.3490/+0.2278. After 39 SimNPO attack steps, both arms are nearly fully recovered (FP 0.9778/0.9819), so neither initialization is resistant. Gemma NPO and SimNPO show the same boundary after one epoch (post-attack FP 0.2611/0.2947 and 0.3527/0.3530, scratch/K-FORGE). We therefore remove any recovery-resistance implication and state that K-FORGE is an initializer, not a defense.

Under quantization, the existing optimizer gain can persist. On a matched NPO pair, 8-bit and 4-bit loading produce only small FP changes in both arms. At matched SimNPO S50 budget, K-FORGE remains lower in FP/extraction/ROUGE after 8-bit (0.5610/0.1924/0.4646 versus 0.6936/0.2844/0.5475) and 4-bit loading (0.4725/0.1373/0.4271 versus 0.5629/0.1777/0.4680), with utility differences below 0.005. This is stability of the optimizer advantage, not immunity to recovery. On Gemma, NPO loses the FP advantage after quantization while SimNPO retains a small one, so we do not generalize across optimizers or model families.

**MUSE benchmark.** We evaluated Llama-2-7B on both MUSE domains with three paired seeds:

| Domain | Steps | Init. | Extraction $\downarrow$ | Forget KnowMem $\downarrow$ | Forget VerbMem $\downarrow$ | Retain KnowMem $\uparrow$ |
|---|---:|---|---:|---:|---:|---:|
| News | 50 | Scratch | **0.3075** | 0.6301 | 0.5741 | **0.5338** |
| News | 50 | K-FORGE | 0.3149 | **0.6235** | **0.5639** | 0.5321 |
| News | 100 | Scratch | **0.3034** | 0.6286 | 0.5727 | **0.5318** |
| News | 100 | K-FORGE | 0.3097 | **0.6278** | **0.5623** | 0.5216 |
| Books | 50 | Scratch | 0.9110 | **0.4106** | **0.9941** | **0.6488** |
| Books | 50 | K-FORGE | **0.9109** | 0.4265 | 0.9958 | 0.6451 |
| Books | 100 | Scratch | **0.8519** | 0.3655 | **0.9554** | **0.6099** |
| Books | 100 | K-FORGE | 0.8692 | **0.3567** | 0.9632 | 0.5860 |

News provides partial transfer: both forgetting-ROUGE metrics improve at both budgets, while extraction is worse and retain quality is lower. In an S100 follow-up whose selection rule was fixed before downstream training, the selected point improves extraction and VerbMem but slightly worsens KnowMem and retain quality. Books improves only isolated metrics. We therefore do not claim benchmark dominance.

The evidence-aligned conclusion is that K-FORGE improves a fixed optimizer's TOFU forget-probability trajectory. Strict compute matching supports the NPO gain, while SimNPO pays a slight utility cost; MUSE transfer is metric-dependent, and direct relearning is a demonstrated weakness.
