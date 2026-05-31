# K-FORGE: Diagnostic, Method, and Positioning Strategy Report

A combined research, analysis, and strategy document for the Kronecker-Fisher closed-form unlearning project, targeting EMNLP 2026 (ARR submission deadline 25 May 2026; EMNLP commitment 2 Aug 2026; notification 20 Aug 2026; conference 24–29 Oct 2026 in Budapest). All recommendations are calibrated against the user's current results: best K-FORGE point at Utility = 0.5535, Forget Q/A Prob = 0.5442, Forget ROUGE = 0.4755, Extraction Strength = 0.2269 on TOFU forget10 with Llama-3.2-1B-Instruct, single mlp.down_proj edit, rank 2, retain-whitened forget GSVD, 2 calibration batches, strength ≈ 0.003.

---

## TL;DR

- **The strength cliff is most likely caused by retain-Fisher under-estimation (only 2 calibration batches → underdetermined retain Kronecker factors), with a secondary contribution from single-layer rank-r coarseness; once retain whitening loses fidelity, marginally larger updates step out of the model's "instruction-tuned basin," producing the abrupt utility collapse seen in the literature on basin-like LLM loss landscapes.**
- **The current K-FORGE Pareto point (0.55 utility, 0.54 forget prob) does NOT beat published OpenUnlearning baselines on Llama-3.2-1B-Instruct TOFU forget10: RMU reaches roughly 0.577 utility / 0.089 forget probability (per the WIN-U paper, arXiv:2604.13438, April 2026), and gold-standard retrain is 0.591 utility / 0.116 forget. K-FORGE is currently dominated by RMU and only competitive against badly-tuned NPO. Framing A ("one-shot SOTA") is therefore not feasible on this trajectory in 8 weeks; Framing B ("K-FORGE init for second-order/preference unlearning") is the highest-EV path.**
- **Top three actions this week: (i) sweep calibration batches from 2 → 64 (the ICLM/Grosse-style influence-function regime uses ~100k+ tokens) and re-plot the cliff; (ii) implement spectrum-aware per-layer rank allocation across all down_proj layers; (iii) implement and measure K-FORGE-as-init followed by 100–500 NPO/SimNPO steps and demonstrate the convergence-acceleration effect that mirrors the user's GFWSVD/Dobi-SVD framing in their ICML 2026 paper.**

---

## Key Findings

### 1. The published Pareto frontier on TOFU forget10 / Llama-3.2-1B-Instruct already includes Newton-style and second-order methods

The newly-released **WIN-U** paper (Zhao, Amiri, Magdon-Ismail, arXiv:2604.13438, April 2026) reports the most directly comparable head-to-head numbers using the OpenUnlearning benchmark on Llama-3.2-1B-Instruct, TOFU forget10 (interpreted from their Table 2; numbers are forget metric / retain metric, then utility). Approximate Pareto points:

| Method | Utility | Forget Prob (lower=better) | Retain-set free? |
|---|---|---|---|
| Original (no unlearn) | 0.601 | 0.881 | – |
| **Gold retrain** | **0.591** | **0.116** | – |
| **RMU** | 0.577 | **0.089** | ✗ |
| SimNPO (default) | 0.596 | 0.837 (under-unlearns) | ✗ |
| GradDiff | 0.443 | 0.057 (over-unlearns utility) | ✗ |
| NPO | 0.436 | 0.214 | ✗ |
| MC-WIN-U | 0.420 | 0.226 | ✓ |
| GradAscent | ~0.0 | 0.0 (collapse) | ✓ |
| **K-FORGE (current best)** | 0.5535 | 0.5442 | partial |

Two observations:

- **K-FORGE is currently strictly dominated by RMU on this Pareto plane.** RMU achieves higher utility (0.577 vs 0.55) AND ~6× stronger forgetting (0.089 vs 0.544). This is fatal for Framing A.
- **K-FORGE Pareto-dominates default-tuned NPO, GradAscent, and (partially) MC-WIN-U** at the equal-utility level. It is a respectable closed-form method but not a SOTA closed-form method.
- The OpenUnlearning paper (Dorna et al., arXiv:2506.12618, June 2025; NeurIPS D&B '25) explicitly notes "SimNPO and RMU as strong performers" with significant ranking sensitivity to tuning. The OpenUnlearning leaderboard exists at locuslab/open-unlearning under community/leaderboard.md, with reproducible baselines in docs/repro.md (lr 1e-5, 10 epochs, AdamW, batch 32).

### 2. Closed-form Newton/Kronecker unlearning is now an active 2026 sub-field

There is one head-on competitor and several flanking ones:

- **K-FADE** (McKinney, Thudi, Bae, Rezaei, Papernot, McIlraith, Grosse; arXiv:2602.10568, 2026): "K-FAC for Distribution Erasure" — uses K-FAC-preconditioned Gauss-Newton steps on the forget set (not closed-form; iterative). Authors claim Pareto improvement on TOFU forget quality and competitive WMDP results. **This paper is the user's most direct technical neighbor and biggest novelty risk.** K-FADE is iterative, not one-shot; this leaves the "closed-form, training-free" niche open.
- **WIN-U** (RPI, April 2026): a Woodbury-scaled Newton step using the GGN forget Jacobian, retain-free. Achieves SOTA *relearning robustness* but lower raw forget-retain trade-off than RMU. Their paper explicitly cites K-FADE as a prior approach that uses retain set access, which is conceptually adjacent to K-FORGE's retain-whitened GSVD.
- Diagonal-Fisher precedents: **SSD** (Foster et al., arXiv:2308.07707, AAAI 2024) is the canonical diagonal-Fisher dampening method but was originally vision-only; LLM SSD adaptations exist mainly in OpenUnlearning. Compression-side: **GFWSVD** (Chekalina et al., arXiv:2505.17974, the user's referenced ICML 2026 work) uses a Kronecker-factored Fisher GSVD for compression — clearly the parent algorithm of K-FORGE in spirit.
- The **LLM Surgeon** (van der Ouderaa et al., arXiv:2312.17244, ICLR 2024) and **Curvature-Weighted Capacity Allocation** (arXiv:2603.00910, 2026) work give a directly applicable spectrum-aware per-layer allocation framework via Kronecker-Fisher curvature scores.

A careful arxiv check (April–May 2026) finds **no published "Kronecker-Fisher one-shot unlearning for LLMs"**: K-FADE is iterative Gauss-Newton, WIN-U uses Jacobian × output Hessian (not Kronecker), GFWSVD is for compression. K-FORGE's specific niche — *training-free, single-pass, Kronecker-factored two-Fisher GSVD edit for unlearning* — appears genuinely novel as a closed-form contribution, but it must demonstrably beat one-shot baselines (single-step K-FADE, single-step WIN-U) to publish under that framing.

### 3. The "strength cliff" has multiple plausible causes; the dominant one is calibration scale

See "Details, Angle 1" below for the full table. The strongest evidence is from Grosse et al. (arXiv:2308.03296, Anthropic 2023), who scale EK-FAC influence functions to 50B-parameter models and use orders of magnitude more data than 2 batches — typical calibration is 10⁴–10⁵ tokens for layerwise A and S factors to converge. Two calibration batches at TOFU's typical effective batch size means roughly 100–500 sequences worth of activation/gradient outer-products feeding into A_r and B_r — far below the regime in which the "weights ~ MN(W*, B⁻¹, A⁻¹)" approximation is statistically meaningful at instruction-tuned LLM scale.

### 4. Llama-3.2-1B-Instruct sits in a small "instruction-tuned basin"

Chen et al. ("Unveiling the Basin-Like Loss Landscape in LLMs," arXiv:2505.17646, 2025) show empirically that pre-training carves a **basic capability basin**, and SFT/DPO carves a **smaller specific capability basin** inside it. The basin's *worst-case* directions are sharp; benign (most-case) directions are flat. This directly predicts a cliff: a rank-r edit aligned even partially with a worst-case direction will fall off the SFT basin abruptly when the step crosses the boundary. **The user's monotone forget metric + cliff utility is precisely the signature predicted by this paper.** Smaller models (1B) have smaller basins, which is consistent with the user's forget05/forget01 transfer probes being weaker — the basin geometry is finer at smaller forget set scales.

### 5. Closed-form rank-r MLP edits in small Llamas have a known catastrophic-collapse mode

Gupta et al. (arXiv:2401.07453) and the ROME line (Meng et al., 2022) show that *single* rank-1 MLP edits to mid-layer down_proj in Llama-2-7B can produce "disabling edits" that decapitate the model; sequential editing accumulates this rapidly. Llama-3.2-1B is more brittle than 7B because there are fewer redundant directions. This is independent corroboration that the user's symptom — single-layer rank-2 down_proj edit hitting a cliff at strength ≈ 0.003 — is structural to the *intervention class*, not to any K-FORGE-specific bug.

### 6. The empirical bar for EMNLP 2026 is rising fast

OpenUnlearning's NeurIPS D&B '25 paper, the community leaderboard, multiple SemEval 2025 Task 4 entries, and the 2025–2026 arxiv flood (Layered Unlearning, AltPO, FLAT, OBLIVIATE, NPO-SAM, BLUR-NPO, NPO+ENT, Leak@k, Selective RMU, SSPU, "Unlearning That Lasts," "Forget to Know Remember to Use," HANKER, Concept Unlearning via Knowledge Triplets, RapidUn, etc.) all use TOFU forget10/05/01 with Llama-3.2-1B-Instruct or Llama-2/3-8B as standard. The reviewer expectation is now: **multi-method baselines, MUSE results, relearning attack, optionally WMDP, and ≥ 2 model scales**. The user's current single-model, single-benchmark, no-baseline results would not pass the EMNLP main-track bar.

---

## Details

## ANGLE 1 — Diagnostic of the Strength Cliff

For each candidate cause, the table lists the assessment (subjective likelihood given the symptoms), the experiment that decisively confirms or refutes it, and the method change that addresses it.

### (a) Single-layer rank-r intrinsic coarseness

**Likelihood: Medium (~35%) as a contributory cause, low as the sole cause.** The Gupta et al. ROME-collapse literature shows that *any* single rank-r down_proj intervention in a small Llama can produce sharp basin-exit. However, the user's edit is per-token activation-conditioned (Kronecker A, B), not data-free, so this should be *less* coarse than ROME-style edits. The cliff appearing at a specific strength rather than at any non-zero step argues against pure coarseness.

- **Experiment:** Edit *all* down_proj layers simultaneously with the same total Frobenius norm budget B; compare the cliff location vs. single-layer. If the cliff disappears or shifts smoothly, single-layer was the cause.
- **Fix:** Multi-layer joint edits (Method extension #1 below).

### (b) Per-layer Kronecker MVN assumption breaking under instruction tuning

**Likelihood: Medium-low (~20%).** The MVN-Kronecker assumption (weights ~ MN(W*, B⁻¹, A⁻¹) at the optimum) is a well-known approximation; Martens & Grosse (2015), George et al. (EK-FAC, 2018), and Eschenhagen et al. (arXiv:2311.00636, 2024) all note it is more accurate near a true optimum and is violated for layers with strong cross-block correlations. Llama-3.2-1B-Instruct is not at a "natural" optimum — it has gone through SFT + DPO, which adds shortcut directions. However, this would predict a *gradual* degradation of the closed-form solution's quality with strength, not a cliff.

- **Experiment:** Compute the EK-FAC version of A_r, B_r (eigenvalue-corrected per Grosse et al. 2023). If the cliff vanishes or moves significantly, MVN-Kronecker was the bottleneck.
- **Fix:** Use EK-FAC instead of plain K-FAC for the retain factors; this is a 1–2 day code change.

### (c) Retain Fisher under-estimation at 2 calibration batches

**Likelihood: HIGH (~55%) — the leading hypothesis.** Two calibration batches in OpenUnlearning's typical TOFU configuration gives ~100–500 sequences and well under 10⁵ tokens fed into A_r, B_r. Grosse et al. (Anthropic, arXiv:2308.03296, 2023) used ~10⁵–10⁶ tokens for layerwise factors at 50B parameters; the multi-stage influence function paper (arXiv:2505.05017, 2025) uses similar scales. Cholesky-inverting an under-rank A_r promotes essentially-noise eigenvectors into "protected directions." A negative edit aligned with a true forget-only direction in σ_f / σ_r terms will be slightly mis-rotated, accumulating into a discrete failure when its component along an unprotected-but-actually-important direction crosses a threshold. **This is the cliff.**

- **Experiment (highest priority):** Sweep calibration batches B_cal ∈ {2, 4, 8, 16, 32, 64} and re-plot the strength–utility tradeoff. Predictions: (i) the cliff strength edges *upward* with B_cal (more accurate retain whitening permits stronger forget pushes); (ii) the slope of utility vs. strength near the cliff *softens*; (iii) the best Pareto point improves monotonically. If any of (i)–(iii) fails, this hypothesis is refuted.
- **Fix:** Increase B_cal to a regime that scales with parameter count of the target layer (suggestion: enough that the empirical A_r has condition number stably below ~10⁶ before damping; in practice 32–128 batches for 1B and 64–256 for 7B). Optionally use streaming Welford updates so memory does not blow up.

### (d) Damping interaction (Tikhonov on Cholesky)

**Likelihood: Medium (~25%).** With B_cal = 2 and damping 1e-3, the damping term is the same order of magnitude as the smallest nonzero singular values of A_r — effectively the damping is *defining* the retain-protected subspace rather than refining it. K-FAC literature (Martens & Grosse 2015; Clarke & Hernández-Lobato, ICML 2024 "Studying K-FAC Heuristics by Viewing Adam through a Second-Order Lens," arXiv:2310.14963) emphasizes that damping is essential for stability and is not a true regularizer in the Bayesian sense for closed-form solves: too much, and the natural-gradient direction collapses toward the gradient direction (i.e., toward diagonal-Fisher behavior); too little, and noise eigendirections dominate the inverse. This is consistent with the user's A2 ablation showing forget-only ≈ untouched: with under-estimated retain factors and damping, the GSVD's "forget-specific" subspace is poorly separated from the noisy retain-protected subspace.

- **Experiment:** Sweep damping ∈ {1e-5, 1e-4, 1e-3, 1e-2, 1e-1} at fixed B_cal = 2 and at B_cal = 32. The *interaction* matters: high damping at low B_cal should look like diagonal-Fisher (matches your A1 ablation); low damping at high B_cal should give a sharper-but-stronger Pareto.
- **Fix:** Per-layer adaptive damping by trace ratio (à la Levenberg-Marquardt in K-FAC); EKFAC's diagonal correction; or Two-Level K-FAC preconditioning (arXiv:2011.00573).

### (e) Mode-connectivity / loss-landscape sharpness near instruction-tuned checkpoints

**Likelihood: HIGH structural background factor (~50% as a *modulating* cause).** Chen et al. (arXiv:2505.17646, 2025) and the critical-sharpness paper directly predict that small instruction-tuned LLMs are in *small* basins with *sharp worst-case directions*. The user's cliff is very plausibly the boundary of the SFT-induced specific-capability basin. This factor cannot be "fixed" inside the K-FORGE algorithm; it can only be *detected* (sharpness measurement) and *avoided* (constrain edit norm to staying inside the basin via line search or trust region).

- **Experiment:** Measure max eigenvalue of the retain-Fisher at θ* and at θ* + Δθ for several Δ until the cliff. If sharpness explodes precisely at the cliff strength, this is corroborated.
- **Fix:** Add a trust-region constraint: enforce ||Δθ||_{B_r ⊗ A_r} ≤ τ rather than tuning a scalar strength. This converts the user's strength-knob into a *natural-gradient-norm* knob, which by construction tracks basin geometry.

### (f) Iterative refinement vs. one-shot

**Likelihood: HIGH structural factor (~40%) — but it's a feature of the method class, not a bug.** K-FADE explicitly takes *several* Gauss-Newton steps and re-estimates the Fisher; this is exactly the standard remedy for a closed-form quadratic-model failing far from the linearization point. The user's results suggest that one rank-r step is borderline-feasible but doesn't have margin. Iteration would re-linearize after the first step, freshly resolve forget/retain spectra, and naturally avoid the cliff.

- **Experiment:** Implement a 2–5 iteration K-FORGE: re-estimate A, B at the perturbed weights and apply another rank-r edit at smaller strength. Compare full Pareto frontier.
- **Fix:** Iterative K-FORGE (Method extension #3 below). Keeps the closed-form-per-step character.

### (g) Forget-retain Fisher generalized eigenvalue spectrum may have no clear gap

**Likelihood: Medium (~25%).** The GFWSVD paper (arXiv:2505.17974) computes generalized eigenvalues σ_f / σ_r per layer for compression and shows that the spectrum is heavy-tailed but has a long tail of small ratios (i.e., shared directions dominate). For *unlearning*, the user wants the *opposite*: a clean separation of forget-only from shared. In a small instruction-tuned model, where SFT data overlaps massively with retain knowledge structurally, the spectrum may genuinely have no gap and rank-r truncation is ill-posed.

- **Experiment:** Per layer, plot the empirical CDF of σ_f / σ_r ratios. If there is no visible gap, the method is fundamentally rank-selection-limited at this model scale. (This experiment is also a publishable artifact for Framing C.)
- **Fix:** Adaptive rank threshold (Method extension #2 below) — drop layers with no gap; allocate rank to those that have one.

### Diagnostic priorities (rank-ordered)

1. **B_cal sweep** (cause c) — single most decisive experiment, ~1 day.
2. **Spectrum CDF measurement** (cause g) — diagnostic-only, ~half day, paper-grade artifact.
3. **Damping × B_cal interaction** (cause d) — ~1 day.
4. **Trust-region/natural-gradient-norm parameterization** (cause e) — ~2 days, may by itself remove the cliff and make all method extensions easier.
5. **Iterative K-FORGE** (cause f) — ~2–3 days.
6. **EK-FAC factors** (cause b) — ~1–2 days.
7. **Multi-layer joint edit** (cause a) — covered under method extensions.

---

## ANGLE 2 — Method Extensions to Push the Pareto Frontier

Each extension is rated on (cliff-fixing likelihood / implementation cost / novelty / one-shot compatibility), 1–5 each.

### (1) Multi-layer joint GSVD across all down_proj layers

**Cliff-fix 4 / Implementation 3 / Novelty 4 / One-shot ✓.** Currently you solve a per-layer independent GSVD and apply rank-r edits; the layerwise weight perturbations compose nonlinearly, and the user's evidence (forget-only inactive, retain-whitened essential) suggests strong coupling. A joint formulation is: minimize a globally weighted forget loss subject to a joint retain budget, allocate rank-r_l to layer l with Σ r_l = R. The closed-form solution is *not* clean for nonlinear composition, but a tractable surrogate is:

- Compute σ_f / σ_r spectra at every down_proj layer.
- Pool eigenvalues globally; pick top R generalized singular vectors across the pooled spectrum.
- Layer-l's r_l = number of selected vectors that came from l.

This is exactly the **Curvature-Weighted Capacity Allocation** framework (arXiv:2603.00910) repurposed for unlearning, and aligns with **LLM Surgeon's** Kronecker-Fisher per-block scoring. Expected effect: utility improves at matched forget level because no single layer is overdriven.

### (2) Spectrum-aware per-layer rank allocation with threshold τ

**Cliff-fix 4 / Implementation 2 / Novelty 4 / One-shot ✓.** Drop the rank-r-per-layer hyperparameter entirely. At each layer compute generalized singular values; allocate rank only to directions with σ_f / σ_r > τ. Layers with no above-threshold mode get r=0 (untouched). This makes K-FORGE *parameter-free* in a meaningful sense and is the user's biggest ergonomic differentiator vs. NPO/SimNPO's lr+β grids. Aligns with AlphaPruning (arXiv:2410.10912) and AdaSVD's idea of variable-rank truncation. Same complexity as #1.

### (3) Iterative K-FORGE (re-estimate Fisher each step, k = 2–5 steps)

**Cliff-fix 5 / Implementation 2 / Novelty 2 / One-shot ✗ (becomes few-step).** Eliminates the linearization error that is the core mechanism of the cliff. Loses the "single-pass" tagline but gains a "few-step second-order Kronecker GSVD" framing that K-FADE has already established as publishable. Expected effect: cliff vanishes, Pareto improves to near K-FADE levels. **This is the safest path to numerically competitive results in 8 weeks.**

### (4) K-FORGE as initialization for NPO / SimNPO / K-FADE

**Cliff-fix 3 (sidesteps it) / Implementation 1 / Novelty 5 (in framing) / One-shot ✗.** Mirrors the user's own ICML 2026 GFWSVD-as-Dobi-init pattern. Run K-FORGE for one closed-form edit, then 100–500 NPO steps. Hypotheses: (a) forget loss starts much lower so fewer steps are needed; (b) NPO's β can be *smaller* because the model is already partially unlearned, reducing utility damage; (c) the learned manifold of the post-K-FORGE init is "closer to retrain" by construction, so NPO wanders less. There is no published precedent for closed-form Fisher init in LLM unlearning; the closest analogues are SOUL (Jia et al., arXiv:2404.18239) using second-order optimizers throughout, and LoKU (arXiv:2408.06621) using Fisher-weighted LoRA init. **This is the recommended primary framing.**

### (5) Hybrid with RMU (representation-misdirection)

**Cliff-fix 2 / Implementation 4 / Novelty 4 / One-shot ✓.** RMU achieves its forget effect by perturbing intermediate-layer activations toward random vectors. K-FORGE could output a closed-form *weight* edit that approximates this representation-space target — concretely, solve for ΔW that minimizes ||σ(W'x) − u||² over the forget set with an A_r ⊗ B_r retain penalty. This is a Tikhonov-regularized least-squares problem and remains closed-form. Could give a "RMU-equivalent without the SGD."

### (6) Targeted layer selection via spectral gap

**Cliff-fix 3 / Implementation 1 / Novelty 3 / One-shot ✓.** Use σ_f^max / σ_r^max per layer as a layer importance score; edit only top-k layers. This is what the empirical RMU literature already does manually (RMU edits a small window); the user can do it data-driven. Combine with #2 for full automation.

### (7) Adaptive damping schedule

**Cliff-fix 4 / Implementation 2 / Novelty 2 / One-shot ✓.** Per-layer damping = c · trace(A_r)/dim(A_r), as in adaptive K-FAC; add a Levenberg-Marquardt-style trust ratio update across iterations of #3. Likely cheapest direct cliff-mitigation.

### (8) Paired projection (subspace removal instead of subtraction)

**Cliff-fix 4 / Implementation 3 / Novelty 5 / One-shot ✓.** Instead of W ← W − α · u_f v_f^T, project W onto the orthogonal complement of the top-r forget-specific subspace in the retain-whitened metric: W ← (I − P_f) W, where P_f is the rank-r projector. **No magnitude knob — only a binary inclusion of each direction.** This eliminates the strength hyperparameter entirely and probably eliminates the cliff (you cannot "go too far" because each direction is fully removed or not at all). Ties closely to "Deep Unlearning" (arXiv:2312.00761) for vision and to SSPU (arXiv:2505.24428). **This is the second-most-promising single change.**

### (9) Lagrangian forget-retain balance

**Cliff-fix 3 / Implementation 4 / Novelty 4 / One-shot ✓.** Replace the (rank, strength) knob pair with an explicit Lagrange multiplier λ on the retain-loss budget. The closed-form solution becomes a Tikhonov-regularized Fisher merge: minimize Δθ^T (A_f ⊗ B_f) Δθ + λ Δθ^T (A_r ⊗ B_r) Δθ + linear forget term. Solve via generalized eigenvalue problem. This is essentially **Fisher-merging** (Matena & Raffel 2022) inverted, with one Fisher subtracted. Cleaner mathematically and gives natural λ↔budget interpretation.

### Recommended extension stack

In implementation order (week-by-week):
- **Week 1:** #7 (adaptive damping) + B_cal sweep (Angle 1c).
- **Week 2:** #2 (spectrum-aware rank) + #6 (layer selection) — these are the same code path.
- **Week 3:** #8 (paired projection) and/or #1 (multi-layer joint).
- **Week 4:** #4 (init-then-NPO/SimNPO) — *headline experiment*.
- **Optional:** #3 (iterative) only if Pareto is still off SOTA.

---

## ANGLE 3 — Positioning / Framing for EMNLP 2026

### Empirical bar by framing

| Framing | What's required | Feasibility in 8 wks | EMNLP main / Findings / Reject |
|---|---|---|---|
| **A.** "Closed-form one-shot SOTA Kronecker unlearner" | Beat NPO/SimNPO/RMU/K-FADE on TOFU + MUSE + (probably) WMDP, multiple model scales, relearning attacks | **Low.** Current 0.55/0.54 point is dominated by RMU 0.577/0.089. Closing this gap one-shot is unlikely. | Mostly reject; possible Findings if a niche claim ("best *closed-form* method") is acceptable to reviewers. |
| **B.** "K-FORGE as the canonical free init for second-order LLM unlearning" | Show K-FORGE init + few NPO/SimNPO steps > NPO/SimNPO from scratch on TOFU + MUSE on Llama-3.2-1B and Llama-2-7B; speedup ≥ 2× steps to a fixed Pareto point | **Medium-high.** Mirrors the user's own ICML paper's framing. The ablation A2 (forget-only inactive) directly motivates retain-aware init. | **Most likely Main; safe Findings.** |
| **C.** Audit/theory paper on the forget-retain Fisher generalized eigenvalue spectrum | Per-layer spectrum analysis on multiple models (1B, 7B); connection to relearning vulnerability and layer importance; smaller empirical contribution | **High.** Mostly already-collectable from the existing pipeline. | **Findings is very likely; Main is a stretch unless theory is sharp.** |
| **D.** "When does diagonal Fisher suffice vs full Kronecker?" | A1 already shows Kronecker > diagonal at matched strength. Generalize across models, datasets, layers. | **High.** A1 ablation is already done. | **Findings likely; Main risky — single methodological claim.** |

### Recommended framing: B with a C appendix

Headline: "**K-FORGE: Closed-Form Kronecker-Fisher Initialization for Stable LLM Unlearning**"

Story arc:
1. Diagonal Fisher (SSD-style) is too coarse for LLM unlearning at instruction-tuned scale (A1 ablation, expanded across models).
2. Full-Kronecker generalized SVD gives a *direction* but not a *destination*: a one-shot edit hits a strength cliff caused by basin geometry (Angle 1 diagnostic, paper-grade theoretical contribution).
3. **The cliff is not a bug; it tells us K-FORGE is the right *direction* but not the right *magnitude*.** Use it as an init for short NPO/SimNPO fine-tuning. (The user's ICML 2026 GFWSVD→Dobi-SVD pattern, applied to unlearning.)
4. Result: K-FORGE-NPO converges in N/k steps to a Pareto point ≥ NPO's, with smaller utility damage and better relearning robustness.
5. Appendix / future work: the σ_f/σ_r spectrum is a layer-importance audit signal predicting unlearning difficulty (Framing C).

Why this works:
- Sidesteps the SOTA bar (you do not need to beat RMU one-shot — only need to beat NPO+nothing).
- Aligned with the user's existing ICML 2026 storyline and code.
- Defensible against "iterative methods are needed" reviewer objection: the paper *is* iterative, just *initialized intelligently*.
- Allows MUSE / WMDP additions in the camera-ready if reviewers ask.

---

## SYNTHESIS AND RECOMMENDATIONS

### (1) Most likely cause of the strength cliff

**Retain-Fisher under-estimation at 2 calibration batches (Angle 1c)**, modulated by the **basin geometry of the SFT'd 1B model (Angle 1e)**, and amplified by **damping–calibration interaction (1d)**. Combined ~70% probability mass. Iteration (1f) and rank coarseness (1a) are real but secondary. Evidence: the cliff's discreteness, the A2 result that forget-only is inactive (which means retain whitening is doing all the work — and is therefore the bottleneck), and the literature on basin sharpness + influence-function calibration scale.

### (2) Most promising single method extension

**Adaptive damping + B_cal scaled to layer dimension + paired projection (Method extensions #7 + #8 stacked).** Implementation sketch:
- Estimate A_r, B_r with B_cal = 32 batches using Welford-streamed updates; budget ~30 minutes wall time at 1B.
- Per layer, damping = max(1e-4, 1e-3 · trace(A)/dim(A)).
- Compute generalized SVD of (Cholesky(A_r))^{-T} W^T (Cholesky(B_r))^{-T} matrix.
- Form the rank-r projection P_f from the top-r left+right singular vectors corresponding to large σ_f/σ_r.
- Apply W ← W − P_f W (no strength knob).
Expected effect: cliff vanishes (no continuous knob to overshoot), Pareto improves toward RMU range or below.

### (3) Recommended framing given trajectory + 8 weeks

**Framing B with C as appendix.** Submit to ARR by 25 May 2026 → commit to EMNLP main 2 Aug 2026. If reviews are weak on novelty/empirics, downgrade to Findings (still EMNLP brand) or pivot to a co-located workshop.

### (4) Revised 8-Week Timeline

(Calendar dates assume "now" ≈ 9 May 2026; ARR deadline 25 May; commit deadline 2 Aug 2026.)

| Week | Dates | Goals | Decision points / kill switches |
|---|---|---|---|
| 1 | May 9–15 | B_cal sweep on TOFU forget10 1B; spectrum CDF per layer; adaptive damping. **Reproduce OpenUnlearning NPO/RMU/SimNPO baselines.** | If cliff persists across all (B_cal, damping) settings, abandon one-shot framing immediately and commit to Framing B. |
| 2 | May 16–22 | Implement spectrum-aware rank + multi-layer (#1, #2). **First full Pareto sweep across baselines + K-FORGE variants.** Decide submission: ARR May 25 vs. Aug 2. | If no variant beats RMU at matched utility AND K-FORGE+NPO doesn't beat NPO alone, kill Framing A; commit to B. |
| 3 | May 23–29 | Paired projection (#8) + iterative K-FORGE (#3). Run on Llama-3.2-3B (intermediate scale) for free transfer evidence. | **Skip ARR May 25 deadline if not solidly beating one baseline. Aim for Aug 2 commit cycle.** |
| 4 | May 30–Jun 5 | **Headline experiment: K-FORGE-init + N steps NPO and SimNPO** on TOFU forget10/05/01, Llama-3.2-1B and 3B. Measure step-to-Pareto-target. | If init advantage is < 1.5× speedup, fall back to Framing C/D. |
| 5 | Jun 6–12 | Scale to **Llama-2-7B**. Time and memory budget for retain-Fisher streaming at 7B (commit decision: 7B is doable if streaming MFF works at <40GB). Begin **MUSE News + Books**. | If 7B blows up (>72h compute or memory issue), keep 1B + 3B only and frame as small-model unlearning. |
| 6 | Jun 13–19 | **Relearning attack** (1-epoch fine-tune on retain or forget subset, per WIN-U/SimNPO protocols). **Quantization-revert attack** (8-bit and 4-bit GPTQ revert). Multiple seeds (≥3). | If relearning attack fully recovers forget, K-FORGE is "shallow unlearning" — disclose explicitly; don't try to hide. |
| 7 | Jun 20–26 | Optional: WMDP. Write paper draft. Internal review. | If WMDP adds cost > 1 week, drop it. |
| 8 | Jun 27–Jul 3 | Polish, ablations, citations, supplementary. Camera-quality figures. ARR submission via June ARR cycle (typical) for Aug 2 EMNLP commit. | – |

Buffer week (Jul 4–10): final revisions for ARR. Aug 2 commit. Aug 20 notification.

### (5) Concrete top-5 actions this week

1. **B_cal calibration sweep**: 6 values × 5 strengths × 1 layer = 30 runs on existing infrastructure (1 day). Re-plot the cliff. *Single most decisive experiment.*
2. **Reproduce NPO, SimNPO, RMU, GradDiff baselines** in the user's exact environment using OpenUnlearning's repro.md scripts, on Llama-3.2-1B-Instruct TOFU forget10. This is necessary regardless of framing; will take ~1–2 days. **You cannot publish without these in the same environment.**
3. **Implement spectrum-aware rank allocation (#2)** across all down_proj layers; combine with #6 layer selection. ~2 days.
4. **Implement and run K-FORGE-as-init + 100–500-step NPO**. Measure forget quality / model utility curves vs. NPO from scratch. ~1.5 days. *This is the headline experiment for Framing B.*
5. **Compute and plot per-layer σ_f / σ_r spectra** for all linear modules (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj) at 1B and 3B. ~half day. This produces both diagnostic information and a publishable artifact for the paper appendix (Framing C support).

---

## Caveats

- The numerical baseline numbers extracted from the WIN-U paper (arXiv:2604.13438v1) are read from Table 2 in v1 of the paper as represented in the search snippets; formatting was ambiguous and the reader should re-verify by directly reading the WIN-U paper PDF before citing those numbers in K-FORGE's paper. The qualitative ranking (RMU strongest forget at high utility; SimNPO highest utility but weakest forget; NPO good forget but utility cost; GradDiff over-unlearns; GA collapses) is robust across multiple sources (OpenUnlearning paper, SimNPO paper, NPO paper).
- OpenUnlearning's published reproducibility numbers (docs/repro.md) explicitly note that "methods such as SimNPO & RMU can be significantly improved with careful tuning" and that their numbers should not be used as a competitive ceiling. The user must run them in their own environment.
- The exact numerical contents of the OpenUnlearning community/leaderboard.md could not be retrieved during this research session due to repeated 429 rate limiting from GitHub; the user should consult that page directly before submission.
- The "K-FADE" paper (arXiv:2602.10568) reports state-of-the-art forget quality in TOFU but its public numerical values in v1 are presented as Pareto plots rather than as the standard (Model Utility, Forget Quality) tabular pairs used elsewhere; the user must extract numbers from the paper figure or replicate.
- The "MFF" algorithm and the user's ICML 2026 paper are not externally indexed in this search; the user's claims about MFF properties are taken on internal authority.
- The 8-week timeline assumes a single researcher with 1× A100 or H100 access; 7B-scale work may need two GPUs or aggressive activation streaming. The user should pre-test the streaming MFF implementation at 7B during week 5, not earlier (it is the riskiest scaling step).
- Some 2026-dated arXiv preprints surfaced in search (e.g., arXiv:2604.13438, 2602.10568, 2603.00910) — these are real entries in the arXiv numbering but were posted in April 2026 and earlier; verify exact authors and dates before citing.
- Predictions of cause likelihoods in Angle 1 are subjective Bayesian estimates from the assistant given the symptoms and literature; they should be tested empirically and revised during week 1.
- "Probability of acceptance" estimates in Framing A/B/C/D are qualitative impressions of EMNLP reviewer norms 2025–2026 and are not formal probabilities.
