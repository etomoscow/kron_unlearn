# EMNLP Proposal Report: Building on MFF / GFWSVD Beyond Multilingual PEFT

## TL;DR — Recommendation Up Front

**Recommended direction: "Kronecker-Fisher Generalized SVD for One-Shot LLM Unlearning"** (working title: **K-FORGE**, "Kronecker Fisher One-shot Rank-r Generalized Erasure"). Use MFF to compute *two* Kronecker-factored empirical Fishers per layer — one on the forget set (A_f ⊗ B_f) and one on the retain set (A_r ⊗ B_r) — and prove a closed-form analog of the GFWSVD theorem: under an MVN weight prior with Kronecker FIMs, the optimal rank-r weight perturbation that maximally suppresses forget likelihood subject to a retain-loss budget is given by the *generalized* singular value decomposition of L_{B_r}^{−⊤} L_{B_f}^⊤ W L_{A_f} L_{A_r}^{−⊤} (Cholesky factors of A_•, B_•). This yields a training-free, gradient-ascent-free, closed-form rank-r edit with strong retain-side guarantees, evaluated on TOFU, MUSE, and WMDP. The Kronecker structure is *load-bearing* — the row- vs. column-cross-covariance signal is exactly what separates "Harry Potter style" from "English grammar" in the same MLP, and a diagonal Fisher provably cannot make that separation.

This is genuinely different from the user's multilingual Kronlingua plan, fits EMNLP main-conference taste (real NLP problem, careful empirical work, clean theory), is feasible inside ~3.5–4k A100-hours, and is the strongest of seven directions surveyed below.

---

## 1. Survey of NLP Areas Where MFF/GFWSVD Could Plug In

I scanned the 2023–2026 literature for each topic cluster in the brief. The table below summarizes whether someone has already used a *Kronecker-factored* Fisher (K-FAC / EK-FAC / similar), whether *only diagonal* Fisher has been used, and the resulting opportunity size.

| Area | Diagonal-Fisher work already exists? | Kronecker-Fisher work already exists? | Gap that MFF/GFWSVD uniquely addresses | Risk of being scooped |
|---|---|---|---|---|
| **LLM unlearning (TOFU/WMDP/MUSE)** | Yes — SSD, FILA, LoKU, VILA, SOUL, Fisher-Forgetting/Removal (Gu et al. 2024) | **Partially — K-FADE (McKinney, Thudi, Bae, … Grosse, arXiv 2602.10568)** uses K-FAC for *iterative Gauss–Newton uphill steps* on the forget set | **Closed-form, training-free, rank-r forget-vs-retain GSVD edit**; simultaneous diagonalization of *two* Kronecker Fishers; explicit retain-set invariance theorem | **Medium-high** — K-FADE is the closest competitor but uses iterative updates, not a closed-form low-rank Fisher-LDA. Still defensible. |
| **Model editing (ROME/MEMIT/MEND family)** | Yes (mostly activation/Hessian-style) | **Yes — CrispEdit (arXiv 2602.15823)** uses K-FAC + matrix-free projector for low-curvature constrained edits | Direct space already taken in 2026 | **High — avoid as primary direction** |
| **Task arithmetic / model merging** | Fisher-Merging (Matena & Raffel 2022), MaTS (Tam 2024 — block-diagonal Fisher via CG), DRIFT-MEDIAN (2025), AGL etc. | **Yes — "Dataless Weight Disentanglement … via K-FAC" (arXiv 2602.17385)**, plus MaTS uses K-FAC inside CG | A pure "K-FAC merge" angle is now occupied; only narrow corners remain (e.g., closed-form rank-r merge under MVN+Kronecker, GFWSVD-style) | **High** — unless you can carve out the closed-form low-rank story specifically |
| **Influence functions in LLMs** | — | **Yes — Grosse et al. 2023 (EK-FAC at 52B), LogIX/Kronfluence, Multi-Stage IF (2025)** | Almost nothing left as a *new* contribution | **Very high — avoid** |
| **Catastrophic forgetting / continual SFT** | EWC, EWCLoRA, online EWC | Mostly Ritter et al. 2018 (K-FAC Laplace) for vision; no large-scale LLM K-FAC EWC at instruction-tuning scale | A *Kronecker-EWC* regularizer for instruction-tuning / DPO that doesn't kill alignment is plausible, but story is incremental | Medium |
| **Memorization detection / extraction risk** | — | **Yes — Garg et al. (2023), Goodfire's "From Memorization to Reasoning in the Spectrum of Loss Curvature" (arXiv 2510.24256)** uses K-FAC to decompose MLP weights by curvature spectrum and suppress memorization | Direct space taken | **High — avoid as primary** |
| **Refusal/safety preservation under fine-tuning** | AlignGuard-LoRA (Fisher-decomp + geodesic regularization, diagonal Fisher), Safefreeze, SPPFT, RefusalGuard | None I can find | A *Kronecker-Fisher–constrained LoRA* that protects the refusal subspace under benign SFT could be novel | Medium |
| **PEFT initialization (single-task, non-multilingual)** | LoRA-DA (Fisher-anisotropy gradient init), some Fisher-aware LoRA papers in unlearning context | None I can find that uses a *full Kronecker* Fisher of weights/gradients to initialize LoRA | A "GFWSVD-of-gradient" Fisher-Kronecker LoRA init for SFT, instruction-tuning, math/code is plausible | Medium-high (very crowded with PiSSA/LoRA-GA/CorDA/EVA/MiLoRA/LoRA-DA) |
| **Quantization (GPTQ/AWQ/OmniQuant/YAQA)** | YAQA touches Hessian-aware quantization; OBS-style methods are well-established | Some recent K-FAC-flavored quantization work | Largely systems-flavored; not strong EMNLP fit | High (and not EMNLP-flavored) |
| **OOD detection / robustness via curvature** | EWC-style Fisher for OOD scoring | Limited | Tangential to NLP main venues | Low priority |

**Headline takeaways:**

1. **Two of the most "obvious" directions are already taken in 2026 papers** that the user must cite: **K-FADE (Gauss–Newton + K-FAC unlearning, McKinney/Thudi/Grosse 2026)** and **CrispEdit (K-FAC + matrix-free projector for editing, 2026)**. These are the two biggest scooping risks. The user's ICML paper's contribution is *also* matrix-free Kronecker-Fisher with Lanczos SVD, so positioning vs. these two papers is unavoidable.
2. **Diagonal-Fisher methods dominate unlearning, merging, alignment, and PEFT init.** The community has *not* moved to Kronecker-Fisher in those NLP-flavored applications yet — only K-FADE and CrispEdit have done so (and only with iterative second-order updates, not the closed-form low-rank GFWSVD primitive).
3. **The unique selling point of MFF + GFWSVD** is the **closed-form rank-r theorem**: SVD of L_B^⊤ W L_A. Nobody — including K-FADE and CrispEdit — has used this *closed-form low-rank* style. That is the leverage point.

---

## 2. Recommended Direction: K-FORGE — Closed-Form Kronecker-Fisher Unlearning via Forget-Retain GSVD

### 2.1 Research question and hypothesis

**RQ.** Can a single, training-free, closed-form rank-r weight edit per layer — derived from the *generalized* SVD of two Kronecker-factored Fishers (forget vs. retain) — match or beat iterative gradient-based and Gauss–Newton (K-FADE) unlearning on TOFU/MUSE/WMDP, while strictly minimizing collateral damage on retained capabilities (MMLU, GSM8K, fluency)?

**H1 (Theory).** Under an MVN weight prior and the Kronecker FIM assumption, the optimal rank-r perturbation Δ minimizing the expected retain-loss increase subject to a forget-loss-increase floor τ has a closed form given by a generalized SVD that simultaneously diagonalizes (A_f ⊗ B_f) and (A_r ⊗ B_r). This generalizes both GFWSVD (single Fisher, low-rank compression) and Fisher-Forgetting/SSD (diagonal Fisher).

**H2 (Empirical).** Because the Kronecker structure encodes row- and column-cross-covariance — which a diagonal Fisher cannot — K-FORGE will identify rank-r directions inside individual MLP/attention projections that are *forget-specific but retain-orthogonal*, yielding a strictly better Pareto frontier on (Forget Quality, Model Utility) than diagonal-Fisher methods (SSD, FILA, VILA), gradient-based methods (NPO, SimNPO, GA), and representation-based methods (RMU, KUDA), and at least competitive with K-FADE while being one-shot rather than iterative.

**H3 (Robustness).** The forget-retain *generalized eigenvalue gap* (the spectrum of the GSVD) provides a principled, per-layer signal predicting (a) how reversible a given unlearn is under relearning attacks, and (b) which layers are most fragile — yielding a built-in audit metric absent in current methods.

### 2.2 Method sketch — math

**Setup (per linear layer with weight W ∈ ℝ^{m×n}).** Run MFF twice, on a forget calibration set D_f and a retain calibration set D_r:

- I_F^{(f)}(θ) ≈ A_f ⊗ B_f, with Cholesky factors A_f = L_{A_f} L_{A_f}^⊤, B_f = L_{B_f} L_{B_f}^⊤.
- I_F^{(r)}(θ) ≈ A_r ⊗ B_r, with Cholesky factors L_{A_r}, L_{B_r}.

**Theorem (analog of the ICML paper's GFWSVD theorem).** Under the MVN-Kronecker assumption used in the ICML paper, the rank-r perturbation Δ ∈ ℝ^{m×n} minimizing

  E[L_retain(W+Δ) − L_retain(W)]   s.t.   E[L_forget(W+Δ) − L_forget(W)] ≥ τ,   rank(Δ) ≤ r,

has a closed form via the generalized SVD of the *doubly whitened* weight

  W̃ := L_{B_r}^{−⊤} (L_{B_f}^⊤ W L_{A_f}) L_{A_r}^{−1}.

Specifically, if W̃ = U Σ V^⊤, then Δ* = − L_{B_f}^{−⊤} U_r Σ_r V_r^⊤ L_{A_f}^{−1}, where (U_r, Σ_r, V_r) keep the top-r generalized singular triplets ranked by the *generalized* singular values σ_i^{(f/r)} = σ_i^{forget} / σ_i^{retain} (Fisher-LDA in weight space).

This is the analog, in the unlearning regime, of the GFWSVD result, and it is a *strict superset* of (i) diagonal-Fisher SSD-style scrubbing (recovered when A_f, A_r, B_f, B_r are diagonal) and (ii) FILA/VILA-style importance maps (recovered when only the diagonal of A_f ⊗ B_f is used).

**Implementation.** Compute W̃ via two Cholesky solves (forward whiten by L_{B_f}, L_{A_f}; back-whiten by L_{B_r}, L_{A_r}), then a thin SVD truncated at rank r (Lanczos, exactly the routine in MFF). Apply Δ* per layer. **No gradient steps, no fine-tuning, single forward pass for calibration.**

**Tunable knobs:**
- r per layer (allocated by total-σ budget across layers — itself a contribution).
- Damping factor on Σ_r (continuous "scrubbing strength").
- Optional iterative refinement: re-estimate Fishers after the edit and apply a second, smaller GSVD step (sequential unlearning over multiple requests — directly addresses MUSE's sustainability axis).

### 2.3 Experimental plan

**Benchmarks (all NLP, all single-GPU friendly).**

1. **TOFU** (Maini et al. 2024): Forget01 / Forget05 / Forget10 splits on Llama-3.2-1B, Llama-3.2-3B, Llama-2-7B, Phi-1.5B. Metrics: Forget Quality (truth ratio test), Model Utility, Real-Authors and World-Facts. Use OpenUnlearning (locuslab) for reproducibility.
2. **MUSE** (Shi et al. 2024): Books and News splits on Llama-2-7B / ICLM-7B. Metrics: VerbMem, KnowMem, PrivLeak, utility preservation, scalability, sustainability.
3. **WMDP** (Li et al. 2024): WMDP-Bio and WMDP-Cyber with Zephyr-7B. Forget MCQ accuracy vs. MMLU retention.
4. **Robustness audit.** Re-finetuning attack (Hu et al. 2024 / Lynch et al. 2024) and quantization revert attack (Quantization-Robust LLM Unlearning, arXiv 2602.13151) — increasingly required by EMNLP/ICLR reviewers.

**Baselines (must include all).**
- Gradient-based: GA, GD (gradient difference), KL+RT, IDK+RT.
- Preference-based: NPO, SimNPO, DPO+RT.
- Representation-based: RMU, Adaptive RMU, KUDA, Geometric Unlearning.
- Diagonal-Fisher: SSD, FILA, VILA, LoKU/IHL.
- Second-order: SOUL (Sophia-style first-order proxy), Fisher-Forgetting/Removal (Gu et al. 2024).
- **Most important comparison: K-FADE (McKinney/Thudi/Grosse 2026)** — this is the head-to-head paper. We must show K-FORGE wins on at least one of (Pareto frontier, wall-clock, sustainability under sequential unlearns, robustness to relearn).

**Ablations (key for EMNLP reviewers).**
- A1: Diagonal Fisher analog (zero out off-diagonal of A, B) — quantifies how much of K-FORGE's gain is the *Kronecker* part.
- A2: Single Fisher (forget only, A_f ⊗ B_f, no retain whitening) — quantifies the *contrastive forget-vs-retain* part.
- A3: Full update (no rank truncation) — quantifies the *low-rank* part.
- A4: Layer selection — sweep which transformer blocks to edit (all/MLP only/attention only).
- A5: Calibration size — does K-FORGE work with as little as 256 forget samples (TOFU regime)?
- A6: Plug K-FORGE as an *initializer* for NPO/SimNPO. Does the closed-form edit + a few NPO iterations beat both alone? This makes K-FORGE complementary to existing methods, not adversarial.

**Analysis sections (the "story" reviewers want).**
- The forget-retain generalized eigenvalue spectrum across layers: which layers carry forget-specific directions? Connection to the "knowledge neurons" / "fact-locating" line (Dai et al., Meng et al.) and to Goodfire's curvature-spectrum work — but with a forget/retain *contrast*, not absolute curvature.
- Per-author / per-document forget difficulty predicted by maxσ_i^{forget}/σ_i^{retain} *before* any editing — a free audit signal.
- Mechanistic visualization: Project U_1 V_1^⊤ to vocab-space via unembedding to show what the rank-1 forget direction *looks like* in tokens.

### 2.4 Expected results and headline claim

**Headline:** *"A closed-form, training-free rank-r weight edit derived from the generalized SVD of two Kronecker Fishers achieves the best (Forget Quality, Model Utility) Pareto frontier on TOFU, MUSE-Books/News, and WMDP across Llama-2-7B and Llama-3-family models, beating diagonal-Fisher (SSD/FILA/VILA), gradient (NPO/SimNPO), representation (RMU), and iterative K-FAC (K-FADE) baselines, while running in seconds and providing a per-layer audit signal predictive of relearning robustness."*

Quantitatively I'd expect (high confidence given precedent):
- TOFU Forget05/10: ≥ +0.05 in Forget Quality at matched Model Utility vs. NPO/SimNPO.
- MUSE-Books: VerbMem ≈ 0 with KnowMem retention degradation < 5% absolute (the key gap the MUSE paper itself flags as unsolved).
- WMDP-bio: ≥ comparable forget reduction to RMU at matched MMLU.
- 50–500× wall-clock speedup over NPO/SimNPO/K-FADE per unlearn request (closed-form vs. iterative).

### 2.5 Risk register and kill-switch criteria

| Risk | Likelihood | Mitigation | Kill switch |
|---|---|---|---|
| **K-FADE already beats us** because iterative GN > closed-form rank-r at single-shot quality. | Medium | Position as: closed-form complement, used as init for K-FADE/NPO. Compare *combined* (K-FORGE init + 1 K-FADE step) which should strictly dominate K-FADE alone. | If K-FORGE-alone is more than 0.05 Forget-Quality below K-FADE on TOFU05 and combination doesn't help, pivot the paper to "K-FORGE as a free, principled initializer for second-order LLM unlearning". |
| **MVN+Kronecker assumption is too violated** in actual LLM weights to give the predicted closed-form benefit. | Medium | The same assumption already holds well empirically in the ICML paper. Treat the theorem as a useful bias and validate empirically. | If diagonal-Fisher ablation (A1) is ≥95% as good as full K-FORGE on every benchmark, the Kronecker structure is not load-bearing → kill. |
| **Forget Fisher is ill-conditioned for tiny forget sets** (e.g., TOFU forget1%). | Medium | Damping/Tikhonov on A_f, B_f; share retain Fisher across forget queries (the retain Fisher is a one-time global computation). | If even with damping, forget1% is unstable, restrict claims to forget ≥ 2%. |
| **Robustness to relearning attacks fails** (forget knowledge resurfaces with 100 fine-tuning steps). | Medium-high (this is the dominant failure mode of *all* current LLM unlearning methods) | Combine K-FORGE with Quantization-LoRA (arXiv 2602.13151) and report under attack. Honest negative result is fine. | Not a kill — just keep claims appropriately narrow. |
| **Scooping by another arXiv-2026 paper** during the 12 weeks. | Medium | Search arXiv biweekly. The closest hit (K-FADE) is already known; CrispEdit is editing not unlearning. | Reposition or pivot to model merging runner-up. |
| **Compute overrun.** | Low | Per-layer MFF on Llama-2-7B is ~1–2 hours on 1×A100 in the ICML paper's setup. Total budget: ~3.5–4k A100h (see §2.7). | If a phase exceeds budget by 30%, drop one model size (e.g., 13B) rather than benchmarks. |

### 2.6 Why this beats the runners-up

- **Versus model merging (Runner-up R1):** The K-FAC-merge paper (arXiv 2602.17385) has already occupied the most direct angle. Closed-form low-rank merge under MVN+Kronecker is still defensible but smaller, and Fisher-merging itself has a long Pareto-saturating literature (TIES/DARE/MaTS/DRIFT-MEDIAN). Less white space.
- **Versus PEFT init (R2):** Crowded (PiSSA / LoRA-GA / CorDA / EVA / MiLoRA / LoRA-DA / EVA), and LoRA-DA already incorporates Fisher anisotropy. Hard for reviewers to feel a 5%-on-MetaMathQA delta is a "real" contribution. Diminishing-returns regime.
- **Versus alignment-preserving SFT (R3):** Real but the canonical paper is AlignGuard-LoRA which already uses Fisher; adding Kronecker is incremental and the headline metric (refusal preservation) is fragile/contested by reviewers.
- **K-FORGE wins on:** (a) load-bearing-ness of Kronecker (diagonal Fisher provably worse — tested in A1), (b) clean theorem (direct extension of the ICML GFWSVD theorem), (c) huge problem (TOFU/MUSE/WMDP are among the hottest EMNLP/ICLR benchmarks of 2024–2026), (d) novelty vs. K-FADE is in the closed-form rank-r forget-vs-retain GSVD framing, (e) easy to evaluate, (f) speed/audit-signal angles add narrative depth beyond raw numbers.

### 2.7 12-week timeline (1 lead + 0.5 collaborator, ≤ ~4k A100h)

| Week | Phase | Deliverable | Compute |
|---|---|---|---|
| **1** | Theory + scaffolding | Write & verify the GSVD-Kronecker theorem; sanity-check on a 2-layer toy MLP with synthetic forget/retain. Code MFF wrapper that returns L_A, L_B Cholesky factors. | < 50 A100h |
| **2** | Calibration plumbing | Hook MFF into HuggingFace Llama / Phi / Zephyr transformer blocks for both attention and MLP linears. Compute & cache forget/retain Fishers for TOFU forget01/05/10 on Llama-3.2-1B and Llama-2-7B. | ~200 A100h |
| **3** | First end-to-end run | K-FORGE on TOFU forget05 with Llama-2-7B. First Pareto curve. | ~150 A100h |
| **4** | Baselines re-implementation | NPO, SimNPO, RMU, SSD, FILA, VILA, K-FADE on TOFU using OpenUnlearning. | ~600 A100h |
| **5** | Scale to MUSE-Books/News | Calibrate Fishers for MUSE; first MUSE Pareto. | ~400 A100h |
| **6** | WMDP | WMDP-Bio/Cyber with Zephyr-7B; baselines + K-FORGE. | ~500 A100h |
| **7** | Ablations A1–A6 | Diagonal/single-Fisher/full-rank/layer/calib-size/init-mode ablations. | ~600 A100h |
| **8** | Robustness/attacks | Relearning attacks (100-step SFT recovery); 4-bit quantization revert attack; adversarial extraction. | ~400 A100h |
| **9** | Sequential / sustainability | MUSE sustainability axis (sequential unlearns), iterative K-FORGE refinement. | ~300 A100h |
| **10** | Analysis & viz | Forget-retain spectrum across layers, vocab-projection of top GSVD directions, layer-wise difficulty maps, audit-signal correlation with relearn vulnerability. | ~150 A100h |
| **11** | Writing v1 | Full draft. Reproducibility scripts. | ~100 A100h (mostly small reruns) |
| **12** | Polish + buffer | Address self-review, reruns, EMNLP-style table polishing, supplementary, anonymous repo. | ~100 A100h |
| | | **Total** | **~3.55k A100h**, with ~1.4k headroom |

Single GPU is sufficient for >80% of the runs; only the 7B baselines (NPO/SimNPO/K-FADE) at scale benefit from 2-A100 parallelism.

### 2.8 Key citations the user must engage with

- McKinney, Thudi, Bae, Rezaei, Papernot, McIlraith, Grosse. *Gauss-Newton Unlearning for the LLM Era* (K-FADE). arXiv:2602.10568. **Primary positioning target.**
- Ikram et al. *CrispEdit: Low-Curvature Projections for Scalable Non-Destructive LLM Editing.* arXiv:2602.15823. (For showing space awareness — different task.)
- Maini et al. *TOFU.* arXiv:2401.06121.
- Shi et al. *MUSE.* arXiv:2407.06460.
- Li et al. *WMDP / RMU.* arXiv:2403.03218.
- Zhang et al. *NPO.* arXiv:2404.05868.
- Fan et al. *SimNPO.* arXiv:2410.07163.
- Jia et al. *SOUL: Second-order optimization for LLM unlearning.* arXiv:2404.18239.
- Foster et al. *Selective Synaptic Dampening (SSD).* arXiv:2308.07707.
- Cha et al. *FILA / LoKU.* arXiv:2408.06621.
- Kim et al. *VILA.* arXiv:2508.21300.
- Gu et al. *Second-Order Information Matters: Fisher Removal/Forgetting.* arXiv:2403.10557.
- Tan et al. *Geometric-disentanglement Unlearning.* arXiv:2511.17100.
- Garg / Goodfire. *From Memorization to Reasoning in the Spectrum of Loss Curvature.* arXiv:2510.24256. (For mechanistic-interpretability tie-in.)
- Grosse et al. *Studying LLM Generalization with Influence Functions* (EK-FAC). arXiv:2308.03296. (Origin of K-FAC at LLM scale — must cite.)
- Tam et al. *MaTS.* arXiv:2312.04339. (Block-diagonal/K-FAC merging — adjacent.)
- Eshchenko-Rakhuba et al. *MFF/GFWSVD* (the user's ICML paper, arXiv:2505.17974) — load-bearing.
- Hsu et al. *FWSVD.* arXiv:2207.00112.
- OpenUnlearning (Mekala et al., NeurIPS D&B 2025). Reproducibility.

---

## 3. Strong Runner-Up Directions (for comparison)

### Runner-up R1 — "GFWSVD-Merge": Closed-form rank-r model merging under Kronecker Fisher

**Idea.** Replace diagonal Fisher in Fisher-Merging / TIES / DARE with full Kronecker Fisher per layer. Use GFWSVD-style closed form to compute the optimal rank-r task vector for each task before averaging, exploiting that low-rank task vectors interfere less (consistent with TSV / LoRE-Merging). Theorem analog: under MVN + per-task Kronecker FIM, the optimal rank-r merged task vector minimizing summed expected loss across tasks has a closed form via the simultaneous diagonalization of {A_t ⊗ B_t}_t.

**Strength.** Directly load-bearing for Kronecker (MaTS explicitly tried K-FAC merging via CG and noted instability — closed-form low-rank may be the right fix). Clean theory.

**Weakness.** "Dataless Weight Disentanglement … via K-FAC" (arXiv 2602.17385) and MaTS already occupy adjacent ground. Story is "Kronecker + low-rank" rather than purely "Kronecker", which thins the contribution. Benchmarks (MergeBench / GLUE / 8-vision-task / FLAN-T5 multi-task) are well-covered, less attention-grabbing than TOFU/MUSE. Many merging baselines (TIES/DARE/AdaMerging/EMR/LoRE) means a complicated 8–10-method comparison.

**Why not chosen as primary:** Slightly lower novelty after the 2602.17385 paper, and merging is starting to saturate.

### Runner-up R2 — "GFW-LoRA-Init": Fisher-Kronecker LoRA initialization for single-task SFT

**Idea.** Initialize LoRA A, B from the top-r left/right factors of the GFWSVD of the *gradient* matrix on the SFT calibration set, where the Fisher Kronecker factors A_l, B_l act as left/right whitening. Generalizes both PiSSA (weight-only SVD), LoRA-GA (gradient-only SVD), CorDA/EVA (activation-only SVD), and LoRA-DA (anisotropic Fisher gradient). Theorem analog: optimal rank-r LoRA initialization that minimizes first-step expected loss under MVN-Kronecker assumption is the L_B^⊤ G L_A SVD where G is the calibration gradient.

**Strength.** Plugs into a hot, well-instrumented benchmark suite (commonsense reasoning, MetaMathQA → GSM8K/MATH, code, instruction tuning). Strict superset framing of LoRA-GA / PiSSA / CorDA / EVA is reviewer-friendly.

**Weakness.** **Very crowded.** PiSSA (NeurIPS 2024 spotlight), LoRA-GA, LoRA-One, LoRA-Pro, LoRA-DA (Oct 2025, anisotropic Fisher), EVA, MiLoRA, OLoRA, NLoRA, SC-LoRA, SVFT, DoRA, LoftQ — all just from 2024–2025. To be SOTA you must beat ~8 strong baselines on 4–5 benchmarks. High effort, marginal headline. Risk of being viewed as "yet another LoRA init paper".

**Why not chosen as primary:** Saturated subfield; thin novelty margin per unit work.

### Runner-up R3 — "GFW-Refusal": Kronecker-Fisher constrained LoRA for alignment-preserving SFT

**Idea.** When fine-tuning Llama-2-Chat / Llama-3-Instruct on a benign downstream task with LoRA, project each LoRA update Δ to be Fisher-orthogonal — under the Kronecker metric of the *aligned base model's* refusal-relevant Fisher — to a target safety subspace. Generalizes AlignGuard-LoRA's diagonal-Fisher decomposition. Evaluate jailbreak ASR (HarmBench, AdvBench), DriftCaps, refusal-direction drift (Wang et al. 2025) before/after task SFT.

**Strength.** Real, headline-grabbing safety problem (Qi et al. 2024 showed 10 examples can break alignment). Clean theoretical story (Kronecker-metric projection generalizes Euclidean projection used in CrispEdit / RefusalGuard).

**Weakness.** CrispEdit (2602.15823) already does K-FAC-based capability preservation for editing; reviewers may conflate the contributions. AlignGuard-LoRA owns the "Fisher-decomp safety LoRA" frame. Safety evaluations are noisy and contested (refusal direction is ill-defined, ASR depends on judge choice). Less crisp empirical wins to demonstrate.

**Why not chosen as primary:** Meaningful, but more frame-by-frame contested than unlearning.

---

## 4. Direction Comparison Table

| Criterion | **R0: K-FORGE (recommended)** | R1: GFWSVD-Merge | R2: GFW-LoRA-Init | R3: GFW-Refusal |
|---|---|---|---|---|
| Kronecker-essential (diagonal won't suffice)? | ★★★★★ — forget vs. retain row/col cross-cov is the entire signal | ★★★★ — task interference has clear off-diagonal structure | ★★★ — diagonal-Fisher LoRA already exists (LoRA-DA), gain probably modest | ★★★★ — refusal subspace has strong row/col structure |
| Clean theorem analog of GFWSVD? | ★★★★★ — direct extension to two Fishers (forget/retain GSVD) | ★★★★ — multi-Fisher simultaneous diagonalization | ★★★★ — direct extension to gradient SVD | ★★★ — projection theorem, but needs care |
| Novelty vs. 2026 prior art | ★★★★ — K-FADE close but iterative; closed-form rank-r is open | ★★★ — 2602.17385 + MaTS occupy nearby ground | ★★ — extremely crowded | ★★★ — AlignGuard-LoRA + CrispEdit nearby |
| EMNLP fit | ★★★★★ — TOFU/MUSE/WMDP are EMNLP-flavored, current | ★★★★ — model merging fits NLP/EMNLP | ★★★★ — PEFT well-loved at EMNLP | ★★★★ — alignment hot at EMNLP |
| Compute (≤ 6k A100h) | ★★★★ — ~3.5–4k h | ★★★★ — ~3–4k h | ★★★★★ — ~2–3k h | ★★★★ — ~3–4k h |
| 12-week feasibility (1 + 0.5 person) | ★★★★ — tight but doable | ★★★★ — moderate | ★★★★ — moderate | ★★★ — safety eval overhead |
| Headline crispness | ★★★★★ — "first closed-form Kronecker-Fisher LLM unlearner; SOTA TOFU/MUSE/WMDP" | ★★★★ — "Kronecker-Fisher merging closed-form" | ★★★ — "+x% over LoRA-GA" | ★★★ — "preserves alignment under benign SFT" |
| Risk of being scooped in 12 weeks | Medium (K-FADE-2 plausible) | Medium-high (merging churns fast) | High (LoRA init is crowded) | Medium |

---

## 5. Caveats and Things to Watch

1. **K-FADE is the dominant scooping risk.** The paper exists (arXiv 2602.10568, Toronto/Vector + Grosse). The user must cite it as the most direct prior work and clearly articulate the differences: K-FORGE is (a) closed-form vs. iterative, (b) explicit forget-vs-retain GSVD vs. forget-only Gauss–Newton, (c) low-rank structured vs. full update, (d) introduces a forget/retain spectrum audit signal. Position K-FORGE as *complementary* to K-FADE: a strong, free initializer that K-FADE / NPO / SimNPO can build on.

2. **The MVN-Kronecker assumption is a modeling choice, not a fact.** It worked well in the ICML paper for compression; it may or may not work as well when *contrasting* two Fishers. Empirical ablation A1 (diagonal-Fisher version) is the make-or-break sanity check for whether Kronecker structure is truly load-bearing.

3. **Robustness to relearning and quantization-revert attacks is the failure mode of every current LLM unlearning method.** Honest negative results here are still publishable; pretending otherwise is not. Budget at least one full week (Week 8) for these.

4. **The MUSE benchmark itself is contested.** Several 2025 papers (HANKER, OpenUnlearning, Obliviate) argue MUSE's KnowMem set is too small and that VerbMem is gameable. The user should report multiple metrics (and add MIA-based privacy metrics from OpenUnlearning) rather than headlining a single MUSE number.

5. **Wall-clock claims need to include calibration time.** MFF for forget+retain Fishers per layer is the main computational cost (~1–2 h on 1×A100 for Llama-2-7B). The actual edit is seconds. Be honest.

6. **The closest "scoop-in-flight" risk is a paper combining Goodfire's K-FAC memorization spectrum with explicit retain Fisher.** No such paper exists as of May 2026 in our search; monitor arXiv weekly.

7. **Three of the surveyed sources (CrispEdit at arXiv:2602.15823, K-FADE at 2602.10568, "Dataless Weight Disentanglement via K-FAC" at 2602.17385) carry 2026 arXiv IDs that would correspond to early–mid 2026 submissions.** Verify versions and exact dates when citing; one of them may still be only on OpenReview/preprint and adjustable in the relative-novelty argument by EMNLP submission.

8. **EMNLP main vs. Findings.** The rcommended K-FORGE has main-conference theory + main-conference benchmarks; if results land within margin of K-FADE alone it is still a strong Findings paper. Frame the contribution as theorem + closed-form algorithm + audit signal so that even a non-headline empirical win is publishable.


# NEW

## Action Plan — Next 8 Weeks

Here's the operational plan, distilled from the research. Starts today.

### This week (May 9–15) — diagnose the cliff and establish baselines

The goal isn't to fix K-FORGE yet. It's to **know exactly where you stand** so every subsequent decision is grounded.

**Day 1–2: B_cal sweep.** Run the calibration-batch sweep: B_cal ∈ {2, 4, 8, 16, 32, 64} × strength ∈ {0.002, 0.0025, 0.003, 0.0033, 0.004} = 30 runs at your current best config (rank 2, single down_proj, retain-whitened, Kron). Re-plot the cliff. **Decisive test of the leading hypothesis** (retain Fisher under-estimation). If the cliff softens or shifts upward as B_cal grows, you have your diagnosis.

**Day 3–4: Reproduce baselines in your environment.** Run NPO, SimNPO, RMU, GradDiff via OpenUnlearning's `docs/repro.md` scripts on Llama-3.2-1B-Instruct TOFU forget10. Use their default hyperparameters first, then their tuned configs. **You cannot publish without these in your exact environment** — and you need to know whether your 0.5535/0.5442 point sits above NPO (probably yes) or above RMU (probably no, per the WIN-U numbers) before you can pick a framing.

**Day 5: Spectrum CDFs.** Compute per-layer σ_f / σ_r distributions for all linear modules (q/k/v/o_proj, gate/up/down_proj) at every transformer block. This is ~half a day of code, ~half a day of compute. **Two outputs**: (a) diagnostic — does any layer have a clear gap?; (b) paper-grade artifact — this figure goes in the appendix regardless of framing.

**End of week deliverable:** A one-page memo with: cliff vs. B_cal plot; baseline Pareto table in your environment; spectrum CDF heatmap. From this you commit to Framing A or B.

### Decision gate (end of Week 1)

Look at the baseline table. Two scenarios:

- **Scenario 1: Your current K-FORGE is within 5% utility / 0.1 forget-prob of RMU** → Framing A is alive. Push hard on method extensions in Week 2.
- **Scenario 2: RMU dominates you by more than that** (most likely) → **Commit to Framing B immediately**. Don't waste two weeks chasing a SOTA you can't reach. The paper becomes "K-FORGE as a free init for second-order unlearning," mirroring your ICML Dobi-SVD framing.

### Week 2 (May 16–22) — implement the two cheap wins

**Adaptive damping** (per-layer, scaled by trace ratio): half a day. **Spectrum-aware rank allocation**: drop the global rank knob, allocate rank only to directions where σ_f/σ_r > τ, layer-wise. ~2 days. These are independent of framing — both make K-FORGE a better algorithm regardless.

Re-run the Pareto sweep with these in. If the cliff vanishes (no continuous strength knob anymore), you've solved the symptom; if not, you've confirmed the cliff is structural.

### Week 3 (May 23–29) — paired projection and the Framing B headline experiment

**Paired projection** (#8 in the report): replace `W ← W − α·u v^T` with `W ← (I − P_f) W` where P_f is the rank-r projector in the retain-whitened metric. No strength knob at all. ~2 days.

**Run the headline experiment for Framing B.** K-FORGE init followed by 100/250/500 steps of NPO and SimNPO, vs. NPO/SimNPO from scratch on the same step budgets. If K-FORGE-init reaches NPO's final Pareto point in ≥1.5× fewer steps, you have a paper.

**Skip the May 25 ARR deadline.** Aim for the next ARR cycle that feeds into the Aug 2 EMNLP commit window. You will not have enough by May 25 to compete.

### Week 4 (May 30–Jun 5) — scale and lock the story

Llama-3.2-3B (cheap intermediate scale to show the trend isn't 1B-specific). Begin MUSE News and Books. Lock the paper title and outline.

### Week 5 (Jun 6–12) — Llama-2-7B

The risky scaling step. Pre-test that streaming MFF fits in memory before committing. If it doesn't fit, drop 7B and frame as a small-model paper — that's still publishable, just narrower.

### Week 6 (Jun 13–19) — robustness audits

Relearning attack (1 epoch SFT on a forget subset — see WIN-U / SimNPO protocols). Quantization-revert attack (8-bit and 4-bit GPTQ). 3 seeds minimum. Honest negative results here are fine and expected; reviewers will demand them anyway.

### Week 7 (Jun 20–26) — draft

Optional WMDP if time. Otherwise: full draft, internal review by 1-2 colleagues, 48-hour turnaround.

### Week 8 (Jun 27–Jul 3) — polish

Final figures, supplementary, anonymous repo, reproducibility checklist. Submit to ARR for the cycle that feeds Aug 2 commit.

---

### What to start *right now*

1. Open a tmux session. Queue the B_cal sweep (you already have `scripts/kforge_tofu_overnight.sh` — extend it).
2. In a separate session, clone OpenUnlearning's NPO/SimNPO/RMU recipes into your repo and queue them. They'll take ~6–12 hours each on a 1B model.
3. While those run, write the spectrum CDF code (a hundred lines on top of your existing factor estimation).

By Friday you'll have the three pieces of information that determine everything downstream. Don't write any LaTeX until the Week 1 memo is done.

### Three rules for the next 8 weeks

- **Don't chase SOTA you can't reach.** If RMU dominates you by Week 1, framing B is mandatory, not optional.
- **Every experiment runs against baselines in your environment.** Published numbers are not enough; reviewers will ask "did you run NPO yourself?"
- **Negative results are fine; hidden negative results are fatal.** Relearning recovery, MUSE underperformance, 7B blowups — disclose, don't hide. The user community is past the point of trusting one-shot unlearning claims; honesty is now the headline asset.
