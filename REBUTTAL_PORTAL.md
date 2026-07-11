# Portal-Ready Author Responses

Each section below is a standalone response for the corresponding review thread. The responses intentionally contain no external links.

## Reviewer 1

Thank you for identifying compute alignment and model-family breadth as the two main gaps. We ran both additions directly in response to these concerns; neither K-FORGE nor the downstream losses were changed.

**FLOP- and wall-clock-matched comparison.** We now charge the complete one-time K-FORGE setup: forget/retain calibration, factor accumulation, Cholesky/SVD computations, edit application, and checkpoint writing. If $t_{\mathrm{KF}}$ is this setup time and $\bar t_{\mathrm{step}}$ is measured scratch time per optimizer step, we compare K-FORGE at budget $T$ against scratch at

$$
T_{\mathrm{cm}}=T+\max\left(
\left\lceil\frac{t_{\mathrm{KF}}}{\bar t_{\mathrm{step}}}\right\rceil,
\left\lceil\frac{F_{\mathrm{KF}}}{F_{\mathrm{step}}}\right\rceil
\right).
$$

For Llama-3.2-1B, K-FORGE takes 61.2 s and $1.435\times10^{14}$ FLOPs for one edited `mlp.down_proj` layer using 512 calibration examples (18.6k valid tokens). With $P=1.236$B, $\tau_f=3042$ and $\tau_r=2924$ tokens per optimizer step, our accounting gives

$$
F_{\mathrm{step}}^{\mathrm{SimNPO}}\simeq6P(\tau_f+\tau_r),\qquad
F_{\mathrm{step}}^{\mathrm{NPO}}\simeq F_{\mathrm{step}}^{\mathrm{SimNPO}}+2P\tau_f.
$$

The setup equals 2.77 NPO or 3.24 SimNPO steps by FLOPs. The edit is folded into the model weights, so it adds no cost to later optimizer steps or inference. On the same single-GPU NVIDIA RTX PRO 6000 Blackwell hardware class, measured 50-step scratch times are 691.5 s for NPO and 629.1 s for SimNPO (13.83/12.58 s per step). Five extra steps cost 69.2/62.9 s, covering the full 61.2 s setup, and are stricter than FLOP matching. Charging those five steps gives the following means over three paired seeds:

| Method | K-FORGE / scratch budget | Scratch FP | K-FORGE FP | Relative reduction | Scratch / K-FORGE utility |
|---|---:|---:|---:|---:|---:|
| NPO | 50 / 55 | 0.0783 | **0.0454** | 42.0% | 0.5224 / **0.5738** |
| NPO | 100 / 105 | 0.0411 | **0.0308** | 25.2% | 0.5712 / **0.5932** |
| NPO | 250 / 255 | 0.0276 | **0.0223** | 19.1% | 0.5925 / **0.6034** |
| SimNPO | 50 / 55 | 0.6462 | **0.5341** | 17.3% | **0.5795** / 0.5712 |
| SimNPO | 100 / 105 | 0.5030 | **0.4066** | 19.2% | **0.5872** / 0.5817 |
| SimNPO | 250 / 255 | 0.3290 | **0.2719** | 17.4% | **0.5957** / 0.5917 |

The FP direction holds for each of the three paired seeds at every budget, and all mean utility differences satisfy the prespecified -0.01 margin. Thus the result does not come from giving K-FORGE a free second-order step. The contribution is not merely that second-order information improves progress per downstream step, but that a one-time structured edit remains useful after its full cost is paid and can be amortized across runs. The secondary metrics are not uniform: NPO extraction worsens at these compute-matched budgets, whereas SimNPO extraction and ROUGE generally improve. We therefore make the compute-matched claim specifically about FP and mean utility. The 3B setup takes 104 s and $3.65\times10^{14}$ FLOPs, or 10.0%/10.9% of a 50-step NPO/SimNPO run. We agree that dense MFF algebra scales as $\mathcal{O}(mn^2+m^2n)$ and therefore carries an additional width factor. This remains a limitation, especially if many layers are edited. In our measured 1B/3B settings, calibration model passes account for 96.2%/98.3% of estimated setup FLOPs; dense algebra accounts for 3.8%/1.7%. We conservatively charge the full setup to every comparison although one initialized checkpoint can in principle be reused across downstream budgets and optimizer seeds.

**Non-Llama family.** We added Gemma-3-1B and held out seed 3 until the configuration and one-shot selection rule were fixed. At 100 steps over four seeds, NPO FP changes from 0.06557 to **0.05328** (-18.7%) and extraction from 0.05496 to **0.03298** (-40.0%), while utility is 0.40062/0.40122. Forget ROUGE worsens from 0.29801 to 0.36471, which we report explicitly. SimNPO FP changes from 0.27240 to **0.26930** ($p=8.8\times10^{-7}$), extraction from 0.12651 to **0.12191**, and ROUGE from 0.39930 to **0.39480**, with utility delta -0.00154. All four paired seeds preserve the FP direction for both methods; the held-out seed also satisfies the utility margin.

The Gemma conclusion also survives the setup charge. Comparing K-FORGE S100 against scratch S103/S105 over four seeds gives NPO FP 0.06397 to **0.05328** ($p=0.0048$, treated descriptively) and SimNPO FP 0.27223 to **0.26930** ($p=9.4\times10^{-6}$). NPO extraction improves, but its ROUGE exception remains. We use $p<.001$ for exploratory inferential claims.

The controls show why this is not merely “more computation.” For Gemma NPO, matched random, diagonal-Fisher, and forget-only initializers have FP 0.06579, 0.06374, and 0.05850 versus 0.05328 for K-FORGE. Weight-SVD reaches 0.05297 but lowers utility to 0.32702 and fails the utility margin. For SimNPO, diagonal Fisher has nearly identical FP and slightly better utility, while K-FORGE is better in extraction and ROUGE. We therefore conclude that retain-aware Fisher structure matters for the NPO result, not that full Kronecker structure wins every optimizer/metric comparison.

We also expanded Llama-3.2-3B to three seeds and all three budgets. SimNPO improves FP, utility, extraction, and ROUGE at 50/100/250 steps; NPO improves mainly at 50 steps. A three-seed Qwen2.5-1.5B pilot was near-neutral. These results support transfer to Gemma and larger Llama for SimNPO, but not universal architecture transfer.

Finally, we implemented the requested presentation fix by splitting Algorithm 1's cross-map, two-SVD, and target/truncation lines. This changes only readability.

## Reviewer 2

Thank you. We agree with the key distinction in the review: our evidence supports an improved Forget Q/A Probability trajectory, not uniformly “better unlearning.” We directly addressed the scope, incomplete-audit, metric-alignment, and software concerns without changing K-FORGE or the downstream losses.

**Model and benchmark breadth.** We added a held-out-seed Gemma-3-1B evaluation. At 100 steps ($n=4$), NPO scratch/K-FORGE values are FP 0.06557/**0.05328**, utility 0.40062/0.40122, extraction 0.05496/**0.03298**, and ROUGE **0.29801**/0.36471. SimNPO values are FP 0.27240/**0.26930**, utility 0.41050/0.40896, extraction 0.12651/**0.12191**, and ROUGE 0.39930/**0.39480**. All four FP pairs move in the favorable direction. The NPO probability/extraction gain is substantial but its ROUGE result is adverse; SimNPO gives a smaller, highly consistent gain across all forgetting metrics. We retain both outcomes.

We also evaluated MUSE-News and MUSE-Books with Llama-2-7B and SimNPO over three paired seeds at 50 and 100 steps. At 100 steps:

| Domain | Init. | Extraction $\downarrow$ | Forget KnowMem $\downarrow$ | Forget VerbMem $\downarrow$ | Retain KnowMem $\uparrow$ |
|---|---|---:|---:|---:|---:|
| News | Scratch | **0.3034** | 0.6286 | 0.5727 | **0.5318** |
| News | K-FORGE | 0.3097 | **0.6278** | **0.5623** | 0.5216 |
| Books | Scratch | **0.8519** | 0.3655 | **0.9554** | **0.6099** |
| Books | K-FORGE | 0.8692 | **0.3567** | 0.9632 | 0.5860 |

MUSE-News improves both forgetting-ROUGE metrics but worsens extraction and retain quality; Books improves only KnowMem. A News follow-up fixed its selection rule before downstream training: minimize one-shot KnowMem subject to at most 0.01 one-shot retain drop. The selected point improves extraction 0.3034 to 0.3012 and VerbMem 0.5727 to 0.5507, but KnowMem and downstream retain quality worsen slightly. We therefore present MUSE as partial metric-level transfer, not benchmark dominance.

**Completed robustness audits.** We replaced the submitted future-tense paragraph with matched-start experiments. For NPO, scratch S100 and K-FORGE S50 begin close in FP (0.0472/0.0519) and utility (0.5683/0.5754). Both receive the same supervised `forget10` attack (AdamW, $10^{-5}$, effective batch 32) for 13 and 39 steps. After 13 steps, scratch/K-FORGE FP is **0.3757**/0.5464, extraction **0.1122**/0.2002, ROUGE **0.3755**/0.4741, and utility 0.4553/**0.4926**. K-FORGE recovers more FP (+0.4945 versus +0.3285, $p=4.7\times10^{-4}$); 39 steps strengthens the same conclusion.

The one-epoch optimizer-transfer audit gives the same boundary for SimNPO. Its matched scratch/K-FORGE starting FP is 0.5452/0.5685. After 13 attack steps, FP is **0.8027**/0.8744 and extraction **0.5393**/0.6942; K-FORGE recovers +0.3059 FP versus +0.2575 for scratch. After 39 steps, both arms are almost fully recovered (FP 0.9778/0.9819), with no distinguishable recovery difference. Gemma repeats the boundary: after one epoch, NPO post-attack FP is 0.2611/0.2947 and SimNPO is 0.3527/0.3530. Thus K-FORGE is an optimization initializer, not a relearning defense.

Quantization is a different intervention. On the matched NPO pair, 8-bit loading changes FP by only +0.00181/+0.00164 (scratch/K-FORGE), and 4-bit by +0.00843/+0.01168; neither shows major recovery. At the ordinary matched SimNPO S50 budget, K-FORGE remains better after 8-bit loading (FP/extraction/ROUGE 0.5610/0.1924/0.4646 versus 0.6936/0.2844/0.5475) and 4-bit loading (0.4725/0.1373/0.4271 versus 0.5629/0.1777/0.4680), with utility differences below 0.005. We claim persistence of an existing optimizer advantage, not recovery immunity. Gemma is optimizer-dependent: NPO's FP gap is erased at 8-bit and reversed at 4-bit, whereas SimNPO retains a small FP advantage at both precisions. Hence quantization persistence is not universal.

The original Llama NPO extraction metric moves against the FP gain at all three compute-matched budgets; SimNPO improves extraction and ROUGE more consistently. This is why we keep the endpoints separate.

**Revised claim.** We replaced broad “better unlearning” wording with:

> K-FORGE improves the early Forget Q/A Probability trajectory of fixed NPO and SimNPO optimizers in the tested TOFU regimes, generally with comparable utility. It does not uniformly improve extraction or Forget ROUGE, transfer to every model/benchmark, or resist active relearning.

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

The evidence-aligned conclusion is that K-FORGE improves a fixed optimizer's TOFU forget-probability trajectory with comparable utility; MUSE transfer is metric-dependent, and direct relearning remains an open weakness.
