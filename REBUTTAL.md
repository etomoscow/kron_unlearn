# K-FORGE Rebuttal

## Response Overview

We thank the reviewers for identifying three concrete gaps in the submitted version: compute-aligned evaluation, model/benchmark breadth, and completed recovery audits. We ran the requested additions. In a direct Llama-3.2-1B wall- and FLOP-matched S50 comparison, K-FORGE lowers NPO Forget Q/A Probability by 22.3% and raises utility by 0.0280. SimNPO preserves a smaller 2.8% probability gain in every seed, but its utility change is -0.0109 and narrowly misses our prespecified -0.01 margin. A held-out fourth Gemma-3-1B seed independently preserves the primary direction; at 100 steps, NPO probability falls by 18.7% and extraction by 40.0% at unchanged mean utility, while SimNPO gives a smaller but highly consistent probability gain.

The added controls, MUSE experiments, and recovery audits also define where the result stops. Random and forget-only edits are weaker in the Gemma NPO comparison, MUSE transfer is metric-dependent, and matched relearning shows that K-FORGE is not a recovery defense. We therefore revise the central claim to:

> K-FORGE is not a new unlearning loss and not a standalone robustness guarantee. It is a one-time Fisher-guided initialization that improves the early forget-probability trajectory of fixed downstream preference-based optimizers.

The strongest supported result is lower Forget Q/A Probability at matched compute budgets: NPO also improves utility, while strict SimNPO matching exposes a small utility trade-off. Extraction and Forget ROUGE are reported separately wherever they disagree.

## Common Notation for Added Results

All TOFU numbers below are means over three seeds unless stated otherwise; sample sizes and standard deviations are shown where space permits and are included in the revised paper tables. The primary endpoint, utility margin, and selection rules were fixed in the rebuttal experiment design before the new compute runs; audits labeled post-hoc remain labeled as such. We use two-sided paired tests and require $p<0.001$ for exploratory inferential claims; larger $p$-values are reported as descriptive evidence only.

For a metric value $x_i$ over $n$ seeds,

$$
\bar{x}=\frac{1}{n}\sum_{i=1}^n x_i,\qquad
s=\sqrt{\frac{1}{n-1}\sum_{i=1}^n (x_i-\bar{x})^2}.
$$

For the compute-matched comparison requested by R1, we charge K-FORGE for its measured one-time setup time. Let $t_{\rm setup}$ be the wall-clock time to estimate Fisher factors, compute the edit, and write the initialized checkpoint; let $t_{\rm train}^{a}(T)$ and $F_{\rm train}^{a}(T)$ be the measured training time and analytical training FLOPs for arm $a$ at $T$ steps. We choose the smallest tested scratch budget that satisfies both resources directly:

$$
T_{\rm cm}=\min\left\{T':
\begin{aligned}
t_{\rm train}^{\rm scratch}(T')&\ge t_{\rm setup}+t_{\rm train}^{\rm KF}(T),\\
F_{\rm train}^{\rm scratch}(T')&\ge F_{\rm KF}+F_{\rm train}^{\rm KF}(T)
\end{aligned}
\right\}.
$$

We also charge analytical FLOPs using the same parameter-token convention as the K-FORGE estimate. For an edited $n\times m$ matrix, our one-time setup convention is

$$
F_{\rm KF}\simeq 6PT_{\rm cal}+2N_F(m^2+n^2)+10(m^3+n^3),
$$

where $T_{\rm cal}$ denotes non-padding prompt and answer tokens processed by calibration and $N_F$ counts loss-bearing rows accumulated into the factors. Numerically, we estimate $T_{\rm cal}$ from the exact tokenizer/template length distribution and use the recorded $N_F=18{,}614$. If $\tau_f,\tau_r$ are forget/retain tokens per optimizer step, then

$$
F_{\rm step}^{\rm SimNPO}\simeq 6P(\tau_f+\tau_r),\qquad
F_{\rm step}^{\rm NPO}\simeq F_{\rm step}^{\rm SimNPO}+2P\tau_f,
$$

where the second term is NPO's reference-model forward pass. In our Llama-3.2-1B setup, the audited setup equals 7.00 NPO or 8.19 SimNPO steps by FLOPs; a dynamic-padding sensitivity calculation raises the largest ratio only to 8.65 steps. The stricter wall-clock comparison uses evaluation-free trainer runtimes from the same software stack and charges the complete 80.157 s setup without amortization. Before inspecting final metrics, this selected scratch S73 for NPO and scratch S86 for SimNPO against K-FORGE S50.

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

For each recovery audit, we compare the change from the arm's own pre-attack checkpoint:

$$
\Delta_a m=m_{\rm post,a}-m_{\rm pre,a},\qquad
\operatorname{DiD}(m)=\Delta_{\rm KF}m-\Delta_{\rm scratch}m.
$$

For a forgetting metric, positive $\Delta_a m$ means recovery of the forgotten behavior; positive $\operatorname{DiD}(m)$ means more recovery for K-FORGE.

For forgetting metrics such as Forget Q/A Probability, extraction, and forget ROUGE, lower is better. For model utility and retain ROUGE, higher is better.

## Reviewer 1

### Concern: K-FORGE may be too expensive; compare by FLOPs and wall-clock, not only by training steps.

We agree that step-matched comparisons alone are incomplete. K-FORGE is a one-time curvature computation, so the relevant question is whether the initialization still helps after charging the method for Fisher estimation, factorizations, and checkpoint writing. We have added both an overhead table and a compute-matched comparison. Wall-clock quantities use the same single-GPU NVIDIA RTX PRO 6000 Blackwell hardware class for the paired measurements.

For one edited `mlp.down_proj` layer with $B_{\rm cal}=32$ batches per forget/retain split, we report the measured estimator/edit runtime, an observed serialization bound, and analytical FLOPs:

| Model | Edited W | Calibration | Time, calib./edit/end-to-end | FLOPs | FLOP-equivalent NPO/SimNPO steps |
|---|---:|---:|---:|---:|---:|
| Llama-3.2-1B | 2048 x 8192 | 512 ex. / est. 47.7k input tok. | 18 / 62 / 80.2 s | $3.62\times 10^{14}$ | 7.00 / 8.19 |
| Llama-3.2-3B | 3072 x 8192 | 512 ex. / est. 47.7k input tok. | 51 / 104 / 145.0 s | $9.29\times 10^{14}$ | 6.90 / 8.08 |

The middle runtime covers calibration through edit application. End-to-end time runs from command-log creation to the final checkpoint file and includes model/data startup and serialization. An implementation audit also clarified an imprecise description in the submitted draft: every reported checkpoint used streamed K-FAC covariance factors, not a Lanczos/rearranged-Fisher MFF routine. We corrected the terminology and complexity without changing the method code or any checkpoint. The executed factor cost is $\mathcal{O}(N_F(m^2+n^2)+m^3+n^3)$ with $\mathcal{O}(m^2+n^2)$ memory. Calibration model passes account for 97.7% (1B) and 99.1% (3B) of estimated FLOPs; factor accumulation and dense factorizations account for the remainder. The width and many-layer costs remain important limitations, but here setup is one-layer and one-time. We charge it in full to every comparison rather than amortizing it across downstream budgets or optimizer seeds.

For the 1B conversion, $P=1.236$B and an effective batch contains on average $\tau_f=3042$ forget and $\tau_r=2924$ retain tokens. This gives $F_{\rm step}=5.18\times10^{13}$ for NPO and $4.42\times10^{13}$ for SimNPO; the estimated $3.62\times10^{14}$ setup is 7.00 and 8.19 steps, respectively. A late audit found that our earlier 50-step wall-clock denominator included evaluation and therefore understated the setup charge. We discarded that comparison. Under the version-aligned, evaluation-free rerun, S50 training averages 197.0 s for NPO and 130.3 s for SimNPO, so the complete setup is 40.7% and 61.5% of S50 training time. Final metrics were not inspected until the longer scratch budgets had been selected from runtime alone: S73 for NPO and S86 for SimNPO. Both also exceed the FLOP requirement, including the dynamic-padding sensitivity check.

The strict S50 comparison reuses the previously reported $\alpha=.60$ initialized checkpoint and reruns both arms under PyTorch 2.9.1+cu130, Transformers 4.51.3, and Accelerate 0.34.2 with evaluation disabled during training. It uses layer-0 `mlp.down_proj`, rank 2, factor damping $10^{-3}$, retain penalty $10^{-2}$, FP32, and seeds 0/1/2. Mean wall time includes the complete setup separately for every K-FORGE arm:

| Method | K-FORGE / scratch steps | Wall: scratch / K-FORGE+setup | Scratch FP | K-FORGE FP | Rel. FP red. | Scratch U | K-FORGE U | Delta U |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NPO | 50 / 73 | 285.5 / **277.2 s** | 0.0590 | **0.0459** | **22.3%** | 0.5486 | **0.5766** | **+0.0280** |
| SimNPO | 50 / 86 | 221.6 / **210.5 s** | 0.5527 | **0.5371** | **2.8%** | **0.5832** | 0.5722 | -0.0109 |

Every paired seed satisfies the wall inequality, with minimum margins of 3.56 s (NPO) and 3.97 s (SimNPO), and the longer scratch runs exceed the FLOP requirement. NPO improves FP in all three seeds and utility in all three; the paired FP test is descriptive ($p=.00696$), while the utility comparison meets our exploratory threshold ($p=4.91\times10^{-4}$). Its extraction and ROUGE means worsen, so this is not metric-wide dominance. SimNPO also improves FP and ROUGE in every seed, with favorable mean extraction, but its FP test is descriptive ($p=.126$) and the utility margin is missed by 0.00093. This is the distinction compute alignment was intended to expose: NPO retains a substantial residual gain after paying for curvature, whereas the SimNPO advantage largely attenuates. We therefore present strict compute matching as evidence that the NPO gain survives the charge, and SimNPO as a weaker trade-off, rather than claiming that both optimizers retain comparable utility.

We repeated the same accounting on Gemma-3-1B over four paired seeds. Its measured 35-second setup corresponds to three NPO or five SimNPO steps, so we compare K-FORGE S100 against scratch S103/S105. For NPO, K-FORGE improves Forget Probability from 0.06397 to **0.05328** ($p=0.0048$, descriptive under our $p<0.001$ threshold) and extraction from 0.05278 to **0.03298** ($p=5.5\times10^{-4}$), while utility changes from 0.40058 to 0.40122; Forget ROUGE worsens from 0.31613 to 0.36471. For SimNPO, it improves Forget Probability from 0.27223 to **0.26930** ($p=9.4\times10^{-6}$), extraction from 0.12633 to **0.12191**, and ROUGE from 0.40864 to **0.39480**, with utility changing by only -0.00090. Thus the primary compute-matched conclusion transfers to a non-Llama architecture, while the NPO ROUGE exception remains explicit.

We also tested whether the effect follows automatically from extra computation or from injecting any low-rank edit. Over four Gemma seeds, NPO K-FORGE reaches FP/extraction `0.05328/0.03298`, compared with `0.06579/0.04842` for a matched random edit, `0.06374/0.06131` for diagonal Fisher, and `0.05850/0.03530` for forget-only Fisher. Weight-SVD reaches FP `0.05297`, but its utility is `0.32702` versus `0.40122` for K-FORGE and therefore fails our prespecified -0.01 utility margin. The SimNPO result is more nuanced: diagonal Fisher has nearly identical FP (`0.26924` versus `0.26930`; the difference is not distinguishable in this sample, $p=0.45$) and slightly higher utility, whereas K-FORGE has lower extraction (`0.12191` versus `0.12420`) and ROUGE (`0.39480` versus `0.40731`). Thus an arbitrary low-rank displacement does not generally reproduce the result, and retain-aware Fisher structure is important for the NPO comparison; the controls do not support a universal advantage of full Kronecker structure on every optimizer and metric.

### Concern: model-family scope is narrow.

We agree this was a real limitation in the submitted version. We expanded the Llama-3.2-3B sanity check from two seeds to three seeds and include the full 50/100/250-step comparison:

| Method | Budget | Scratch FP | K-FORGE FP | Rel. FP red. | Scratch U | K-FORGE U | Delta U |
|---|---:|---:|---:|---:|---:|---:|---:|
| NPO | 50 | 0.0863 | **0.0649** | 24.7% | 0.5895 | **0.6393** | +0.0498 |
| NPO | 100 | **0.0342** | 0.0388 | -13.5% | 0.6480 | **0.6700** | +0.0220 |
| NPO | 250 | **0.0267** | 0.0291 | -9.3% | **0.6649** | 0.6624 | -0.0025 |
| SimNPO | 50 | 0.6911 | **0.5829** | 15.7% | 0.6365 | **0.6558** | +0.0193 |
| SimNPO | 100 | 0.5168 | **0.4550** | 11.9% | 0.6466 | **0.6546** | +0.0080 |
| SimNPO | 250 | 0.3342 | **0.3145** | 5.9% | 0.6617 | **0.6656** | +0.0039 |

This result is more nuanced than the 1B result: SimNPO improves all four reported mean metrics at each 3B budget, while NPO mainly benefits at early steps. The revised claim therefore does not present it as broad scaling evidence for both optimizers.

More importantly, we completed a non-Llama evaluation on Gemma-3-1B and then added a held-out fourth seed at 100 steps. The Gemma full and retain checkpoints use the same TOFU/OpenUnlearning fine-tuning splits and evaluation protocol as the Llama experiments. We selected the K-FORGE strength before downstream training using a one-shot grid: among the points whose utility drop from the base model was at most 0.01, we chose the point with the lowest Forget Q/A Probability. This selected $\alpha=0.8$; $\alpha=1.0$ was excluded because its utility drop was 0.0111. The matched downstream results are:

| Method | Budget | $n$ | Scratch FP | K-FORGE FP | Rel. FP red. | Scratch U | K-FORGE U | Delta U |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NPO | 50 | 3 | 0.07436 | **0.07434** | 0.02% | 0.34903 | **0.35804** | +0.00900 |
| NPO | 100 | 4 | 0.06557 | **0.05328** | **18.74%** | 0.40062 | **0.40122** | +0.00060 |
| SimNPO | 50 | 3 | 0.27334 | **0.27005** | 1.20% | **0.40401** | 0.40230 | -0.00171 |
| SimNPO | 100 | 4 | 0.27240 | **0.26930** | 1.14% | **0.41050** | 0.40896 | -0.00154 |

The held-out seed matches the probability direction for both optimizers. For NPO, its Forget Q/A Probability changes from 0.06386 to 0.05762 with a utility change of -0.00645; for SimNPO, it changes from 0.27212 to 0.26907 with a utility change of -0.00312. Both satisfy the prespecified utility margin of -0.01.

The strongest aggregate Gemma result is NPO at 100 steps: over four seeds, K-FORGE lowers Forget Q/A Probability by 18.7% and extraction by 40.0% (0.0550 to 0.0330) while leaving mean utility effectively unchanged. Forget ROUGE moves in the opposite direction (0.2980 to 0.3647), so we do not present this as uniform improvement across all forgetting metrics. SimNPO gives a smaller but highly consistent 1.14% probability reduction, together with lower extraction (0.1265 to 0.1219), lower Forget ROUGE (0.3993 to 0.3948), and a utility change of only -0.00154. Two-sided paired tests on Forget Q/A Probability give $p=0.0115$ for NPO and $p=8.8\times10^{-7}$ for SimNPO at 100 steps. Given the small sample, we treat the NPO test as descriptive; the SimNPO result remains below the $p<0.001$ exploratory threshold after adding the held-out seed.

The completed Qwen2.5-1.5B pilot was near-neutral: SimNPO Forget Q/A Probability changed by less than 0.06% relatively at both budgets. We therefore use Gemma as positive non-Llama evidence but do not claim universal transfer across architectures.

### Concern: Algorithm 1 readability.

We split the combined SVD and update lines in Algorithm 1, specifically the lines corresponding to the cross-Cholesky maps, the two SVDs, and the target/rank truncation. This is a presentation-only change; it does not alter the method.

## Reviewer 2

### Concern: scope is narrow and some claims sound broader than the evidence.

Thank you for the constructive feedback. We agree that improved Forget Q/A Probability should not be conflated with uniformly better unlearning. We address the concerns below and narrow our claims accordingly.

**Non-Llama model family.** We added a complete Gemma-3-1B comparison at 50 and 100 steps. The 50-step rows contain the original three seeds; the 100-step rows additionally include a held-out fourth seed. `S / KF` denotes scratch / K-FORGE:

| Method | Steps | $n$ | Forget Prob. S / KF ↓ | Rel. red. | Utility S / KF ↑ | Extraction S / KF ↓ | Forget ROUGE S / KF ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| NPO | 50 | 3 | 0.07436 / **0.07434** | 0.02% | 0.34903 / **0.35804** | 0.03091 / **0.03046** | **0.30962** / 0.38095 |
| NPO | 100 | 4 | 0.06557 / **0.05328** | **18.74%** | 0.40062 / **0.40122** | 0.05496 / **0.03298** | **0.29801** / 0.36471 |
| SimNPO | 50 | 3 | 0.27334 / **0.27005** | 1.20% | **0.40401** / 0.40230 | 0.12743 / **0.12278** | 0.39632 / **0.39473** |
| SimNPO | 100 | 4 | 0.27240 / **0.26930** | 1.14% | **0.41050** / 0.40896 | 0.12651 / **0.12191** | 0.39930 / **0.39480** |

The held-out seed matches the Forget Q/A Probability direction for both optimizers while satisfying a prespecified utility margin of -0.01. At 100 steps, K-FORGE reduces NPO probability by 18.7% and extraction by 40.0% at unchanged mean utility, but worsens Forget ROUGE. SimNPO shows a smaller but highly consistent probability reduction, lower extraction and ROUGE, and negligible utility cost ($p=8.8\times10^{-7}$ for paired Forget Q/A Probability over four seeds). This is why our revised claim is metric-specific rather than “better unlearning” in general.

**Scale.** We expanded the Llama-3.2-3B evaluation to three seeds and 50/100/250-step budgets. The table reports the complete metric set with separate scratch and K-FORGE columns:

| Method | Steps | Scratch FP ↓ | K-FORGE FP ↓ | Scratch Utility ↑ | K-FORGE Utility ↑ | Scratch Extraction ↓ | K-FORGE Extraction ↓ | Scratch ROUGE ↓ | K-FORGE ROUGE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NPO | 50 | 0.08625 | **0.06494** | 0.58947 | **0.63927** | **0.05733** | 0.07022 | **0.23729** | 0.29287 |
| NPO | 100 | **0.03422** | 0.03884 | 0.64796 | **0.67000** | **0.05539** | 0.07701 | **0.26001** | 0.30622 |
| NPO | 250 | **0.02665** | 0.02912 | **0.66487** | 0.66244 | **0.05672** | 0.07904 | **0.26071** | 0.30239 |
| SimNPO | 50 | 0.69107 | **0.58286** | 0.63647 | **0.65581** | 0.35088 | **0.26356** | 0.58694 | **0.51886** |
| SimNPO | 100 | 0.51676 | **0.45503** | 0.64655 | **0.65462** | 0.22328 | **0.19112** | 0.48640 | **0.46707** |
| SimNPO | 250 | 0.33421 | **0.31446** | 0.66173 | **0.66556** | 0.15079 | **0.14226** | 0.43431 | **0.42856** |

For SimNPO at 3B, K-FORGE improves the mean Forget Q/A Probability, utility, extraction, and Forget ROUGE at every tested budget. NPO benefits at 50 steps in probability and utility but pays for this in extraction and ROUGE; at 100 and 250 steps its results are mixed or unfavorable. We therefore present the 3B result as consistent scale transfer for SimNPO rather than for both downstream optimizers.

**Additional benchmark.** We evaluated MUSE-News and MUSE-Books with Llama-2-7B and SimNPO over three seeds at 50 and 100 steps. The following table gives the complete 100-step comparison; the complete 50-step results are reported in our response to R3 below and show the same mixed pattern:

| MUSE domain | Init | Extraction ↓ | Forget KnowMem ROUGE ↓ | Forget VerbMem ROUGE ↓ | Retain KnowMem ROUGE ↑ |
|---|---|---:|---:|---:|---:|
| News | Scratch | **0.3034** | 0.6286 | 0.5727 | **0.5318** |
| News | K-FORGE | 0.3097 | **0.6278** | **0.5623** | 0.5216 |
| Books | Scratch | **0.8519** | 0.3655 | **0.9554** | **0.6099** |
| Books | K-FORGE | 0.8692 | **0.3567** | 0.9632 | 0.5860 |

MUSE-News shows lower KnowMem and VerbMem forgetting metrics, but mixed extraction and retain quality. We also ran a News follow-up whose rule was fixed before downstream training: select the lowest one-shot KnowMem point subject to at most 0.01 retain drop, then run only that point downstream. The selected $\alpha=1.0$ S100 run improves extraction from 0.3034 to 0.3012 and VerbMem from 0.5727 to 0.5507, while KnowMem changes from 0.6286 to 0.6294 and retain quality from 0.5318 to 0.5197. MUSE-Books is less favorable: KnowMem ROUGE improves from 0.3655 to 0.3567, while extraction, VerbMem, and retain quality worsen. We therefore treat MUSE as evidence of partial benchmark transfer, not uniform benchmark-level dominance. The completed Qwen pilot was near-neutral, so we do not claim universal transfer across model families.

### Concern: robustness audits were written in future tense and looked incomplete.

We removed the future-tense language and completed both direct-relearning and quantization audits over three paired seeds. Crucially, we added a stricter comparison whose starting operating points are close: scratch NPO S100 versus K-FORGE-initialized NPO S50 ($\alpha=0.60$). Their pre-attack Forget Probability is 0.0472/0.0519 and utility is 0.5683/0.5754. We then apply identical supervised fine-tuning on the 400-example `forget10` QA split (AdamW, $10^{-5}$ learning rate, effective batch size 32) for one and three epochs, corresponding to 13 and 39 optimizer steps.

| Metric | Pre S / KF | Post 1 epoch S / KF | Post 3 epochs S / KF |
|---|---:|---:|---:|
| Forget Prob. $\downarrow$ | **0.0472** / 0.0519 | **0.3757** / 0.5464 | **0.7298** / 0.9110 |
| Extraction $\downarrow$ | **0.0665** / 0.0728 | **0.1122** / 0.2002 | **0.4067** / 0.8123 |
| Forget ROUGE $\downarrow$ | **0.2678** / 0.2724 | **0.3755** / 0.4741 | **0.6219** / 0.8842 |
| Utility $\uparrow$ | 0.5683 / **0.5754** | 0.4553 / **0.4926** | 0.4932 / **0.5097** |

This stricter audit changes our interpretation. K-FORGE is **not** more resistant to direct relearning at a matched initial operating point: after one epoch, its Forget Probability recovery is +0.4945 versus +0.3285 for scratch ($p=4.7\times10^{-4}$ for the paired recovery difference), and the gap increases after three epochs. SimNPO gives the same one-epoch boundary; after three epochs both SimNPO arms are almost fully recovered (FP 0.9778/0.9819). Gemma NPO and SimNPO also fail to preserve an advantage after one epoch, with post-attack FP `0.2611/0.2947` and `0.3527/0.3530` (scratch/K-FORGE). We report these negative results and remove the earlier robustness implication.

Quantization is less destructive. For the same matched pair, the within-arm changes are:

| Loading | $\Delta$ FP S / KF | $\Delta$ Extraction S / KF | $\Delta$ ROUGE S / KF | $\Delta$ Utility S / KF |
|---|---:|---:|---:|---:|
| 8-bit | +0.00181 / +0.00164 | -0.00041 / -0.00129 | +0.00236 / +0.00230 | -0.00395 / -0.00615 |
| 4-bit | +0.00843 / +0.01168 | -0.00416 / -0.00757 | -0.00287 / -0.00487 | -0.03774 / -0.04980 |

At the ordinary matched downstream budget, the initializer advantage also survives quantization for SimNPO:

| SimNPO S50 | FP S / KF $\downarrow$ | Utility S / KF $\uparrow$ | Extraction S / KF $\downarrow$ | ROUGE S / KF $\downarrow$ |
|---|---:|---:|---:|---:|
| 8-bit | 0.6936 / **0.5610** | **0.5747** / 0.5727 | 0.2844 / **0.1924** | 0.5475 / **0.4646** |
| 4-bit | 0.5629 / **0.4725** | **0.5320** / 0.5280 | 0.1777 / **0.1373** | 0.4680 / **0.4271** |

The paired Forget Probability tests give $p=1.0\times10^{-4}$ (8-bit) and $p=4.5\times10^{-4}$ (4-bit). Thus 4/8-bit loading preserves the matched-budget SimNPO advantage, but the matched-start NPO audit does not establish an intrinsic K-FORGE robustness advantage. The completed audits give a precise boundary: K-FORGE can improve optimization in a way that survives quantization, but it is not a recovery defense.

The four-seed Gemma audit shows that this stability is optimizer-dependent. For NPO, 8-bit loading erases the FP gap (`0.06989/0.06967` scratch/K-FORGE), while 4-bit reverses it (`0.05803/0.06110`) and lowers K-FORGE utility more. For SimNPO, the smaller FP advantage persists after both 8-bit (`0.26858/0.26569`) and 4-bit loading (`0.21865/0.21410`) with comparable utility. We therefore do not claim model- or optimizer-universal quantization robustness.

### Revised claim and metric alignment.

We agree that “K-FORGE improves unlearning” is too broad. We therefore replace it with the following evidence-aligned claim:

> K-FORGE improves the early Forget Q/A Probability trajectory of fixed downstream NPO and SimNPO optimizers in the tested regimes. Direct wall- and FLOP-matching supports the NPO gain; SimNPO retains a smaller probability gain with a slight utility trade-off. We do not claim uniform improvement across extraction, Forget ROUGE, model families, or benchmarks.

The revised tables report Forget Q/A Probability, extraction, and Forget ROUGE separately. We explicitly retain both the Gemma NPO case where probability/extraction improve but ROUGE worsens and the Llama NPO rows where extraction moves against the probability gain. For those rows we claim only an improved Forget Q/A Probability trajectory, not broadly better unlearning. This aligns each claim with the metric directly supported by the evidence rather than treating different notions of forgetting as interchangeable.

### Concern: no usable software.

We have expanded the artifact rather than only promising a later release. It now includes the K-FORGE trainer/config, the exact initialization harness, matched relearning and 4/8-bit quantization runners, and a structured aggregator invoked as `python open-unlearning/scripts/summarize_rebuttal_additions.py --check`. Every runner records a TSV manifest, fixes the paired seeds and attack settings, and skips only an existing summary that contains all required metrics; the aggregator exits nonzero on missing or malformed results. A compact per-seed metric snapshot reproduces the reviewer-requested aggregate tables and paired tests without checkpoints. CPU algebra tests compare the implemented full-rank Wiener edit against a direct Hessian solve and the zero-penalty rank-$r$ edit against its Eckart--Young solution. The root README gives the executable commands and identifies calibration size, edited layer, rank/strength, and evaluation dtype settings.

## Reviewer 3

### Concern: robustness against relearning or recovery attacks is important.

Thank you for highlighting this limitation. We replaced the submitted future-tense discussion with completed three-seed audits and strengthened the design by matching the initial operating point. We compare scratch NPO S100 with K-FORGE-initialized NPO S50 ($\alpha=0.60$); pre-attack Forget Probability is 0.0472/0.0519 and utility is 0.5683/0.5754. Both checkpoints then receive the same supervised `forget10` attack for 13 or 39 optimizer steps.

| Attack | FP post S / KF $\downarrow$ | Extraction post S / KF $\downarrow$ | ROUGE post S / KF $\downarrow$ | Utility post S / KF $\uparrow$ |
|---|---:|---:|---:|---:|
| 13 steps | **0.3757** / 0.5464 | **0.1122** / 0.2002 | **0.3755** / 0.4741 | 0.4553 / **0.4926** |
| 39 steps | **0.7298** / 0.9110 | **0.4067** / 0.8123 | **0.6219** / 0.8842 | 0.4932 / **0.5097** |

This audit does not support a recovery-resistance claim. After 13 steps, Forget Probability increases by 0.4945 for K-FORGE versus 0.3285 for scratch ($p=4.7\times10^{-4}$ for the paired difference); the three-epoch audit gives the same conclusion. We state explicitly that K-FORGE improves the downstream optimization trajectory but does not make the resulting model resistant to relearning.

We also completed 4/8-bit loading audits. On the matched NPO pair, 8-bit loading changes Forget Probability by only +0.00164 for K-FORGE (+0.00181 scratch); 4-bit changes it by +0.01168 (+0.00843 scratch), and extraction does not recover. On matched-budget SimNPO S50, K-FORGE retains lower FP/extraction/ROUGE after both 8-bit (`0.5610/0.1924/0.4646` versus `0.6936/0.2844/0.5475`) and 4-bit loading (`0.4725/0.1373/0.4271` versus `0.5629/0.1777/0.4680`), with utility differences below 0.005. Thus quantization preserves the optimizer advantage when present, but does not confer recovery immunity. This separates stability under quantization from vulnerability to an active recovery attack.

Gemma makes the boundary more specific: NPO's FP advantage is erased at 8-bit and reversed at 4-bit, whereas SimNPO retains a small FP advantage at both precisions. Quantization persistence is therefore optimizer-dependent rather than a general property of K-FORGE checkpoints.

### Concern: evaluation is mainly TOFU; add another benchmark, preferably MUSE or WMDP.

We added MUSE-News and MUSE-Books experiments using Llama-2-7B and SimNPO at 50 and 100 downstream steps. Every comparison uses three paired seeds. Lower extraction and forget ROUGE indicate stronger forgetting; higher retain ROUGE is better.

| Domain | Steps | Init | Extraction $\downarrow$ | Forget KnowMem $\downarrow$ | Forget VerbMem $\downarrow$ | Retain KnowMem $\uparrow$ |
|---|---:|---|---:|---:|---:|---:|
| News | 50 | Scratch | $\mathbf{0.3075\pm0.0021}$ | $0.6301\pm0.0062$ | $0.5741\pm0.0021$ | $\mathbf{0.5338\pm0.0069}$ |
| News | 50 | K-FORGE | $0.3149\pm0.0036$ | $\mathbf{0.6235\pm0.0099}$ | $\mathbf{0.5639\pm0.0120}$ | $0.5321\pm0.0056$ |
| News | 100 | Scratch | $\mathbf{0.3034\pm0.0041}$ | $0.6286\pm0.0017$ | $0.5727\pm0.0031$ | $\mathbf{0.5318\pm0.0084}$ |
| News | 100 | K-FORGE | $0.3097\pm0.0011$ | $\mathbf{0.6278\pm0.0063}$ | $\mathbf{0.5623\pm0.0123}$ | $0.5216\pm0.0063$ |
| Books | 50 | Scratch | $0.9110\pm0.0002$ | $\mathbf{0.4106\pm0.0125}$ | $\mathbf{0.9941\pm0.0049}$ | $\mathbf{0.6488\pm0.0014}$ |
| Books | 50 | K-FORGE | $\mathbf{0.9109\pm0.0002}$ | $0.4265\pm0.0047$ | $0.9958\pm0.0019$ | $0.6451\pm0.0064$ |
| Books | 100 | Scratch | $\mathbf{0.8519\pm0.0255}$ | $0.3655\pm0.0101$ | $\mathbf{0.9554\pm0.0197}$ | $\mathbf{0.6099\pm0.0057}$ |
| Books | 100 | K-FORGE | $0.8692\pm0.0203$ | $\mathbf{0.3567\pm0.0023}$ | $0.9632\pm0.0129$ | $0.5860\pm0.0166$ |

MUSE-News provides partial transfer. At 50 steps K-FORGE lowers both KnowMem and VerbMem forgetting metrics, including lower KnowMem in all three paired seeds, while extraction is worse and retain quality changes only slightly. In the S100 follow-up, we fixed the rule before downstream training and selected the lowest one-shot KnowMem point subject to at most 0.01 retain drop ($\alpha=1.0$). It improves both extraction (0.3034 to 0.3012) and VerbMem (0.5727 to 0.5507), but KnowMem and retain quality worsen (0.6286 to 0.6294 and 0.5318 to 0.5197).

MUSE-Books is less favorable. At 100 steps K-FORGE lowers KnowMem ROUGE from 0.3655 to 0.3567, but extraction, VerbMem, and retain quality worsen; at 50 steps it provides no meaningful advantage. We therefore report MUSE as evidence that individual forgetting gains can transfer beyond TOFU, not as uniform benchmark-level dominance.

Accordingly, the revised claim is:

> K-FORGE improves the early Forget Q/A Probability trajectory of fixed preference-based optimizers on TOFU. Direct wall- and FLOP-matching supports the NPO gain, while strict SimNPO matching shows a smaller utility trade-off. MUSE provides partial benchmark-transfer evidence, and matched relearning shows that K-FORGE is an optimization initializer rather than a recovery defense. We do not claim recovery immunity or uniform improvement across forgetting metrics and benchmarks.

We have added the exact attack, quantization, and aggregation scripts and documented the MUSE model, strength, budgets, seeds, dtype, and selection rule in the artifact.

## Revised Paper Claim

> K-FORGE improves the forget-probability trajectory of fixed downstream preference-based unlearning methods. On TOFU `forget10`, the NPO improvement remains after charging the one-time setup in both FLOPs and wall-clock time; SimNPO retains a smaller probability gain with a slight utility trade-off. A held-out fourth seed strengthens the Gemma-3-1B evidence: at 100 steps, K-FORGE reduces NPO Forget Q/A Probability by 18.7% and extraction by 40.0% at unchanged mean utility, and gives a smaller but highly consistent SimNPO gain. Results on Qwen and the two MUSE domains are more mixed, so we do not claim universal model- or benchmark-level transfer.
