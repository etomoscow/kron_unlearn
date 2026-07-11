# K-FORGE Autonomous Rebuttal Log

## Window

- Start: `2026-07-11 14:57 UTC`
- Deadline: `2026-07-12 00:57 UTC`
- Objective: maximize the probability that the three reviewers raise their scores using complete, reproducible, and evidence-aligned responses.

## Prespecified Claim Boundary

Prespecified primary hypothesis: K-FORGE is the best tested initializer for a fixed preference-based optimizer at matched downstream budget; any benefit over scratch must remain after charging setup cost. The final report retains metric- and optimizer-specific failures of this hypothesis.

Primary endpoint: Forget Q/A Probability. Utility non-inferiority margin: `-0.01`. Extraction and Forget Q/A ROUGE are mandatory secondary metrics.

## Initial State

At the start of this window, four GPUs were idle enough for new work (GPU 0-3); GPU 4 was occupied externally. The worktree contained unrelated user changes, including quantized-loading support and rebuttal edits; these are preserved.

The previous queue had already completed:

- held-out Gemma seed `3` for NPO and SimNPO at S100;
- Gemma random, weight-SVD, diagonal-Fisher, and forget-only controls at S100 over seeds `0,1,2`;
- Gemma compute-matched scratch runs (`NPO S103`, `SimNPO S105`) over seeds `0,1,2`;
- a predeclared MUSE-News strength follow-up, selecting `alpha=1.0`;
- three-seed 4-bit and 8-bit Llama NPO S50 quantization evaluation.

These artifacts are treated as completed inputs and will be re-verified before use.

## Decision Log

### 2026-07-11 15:10 UTC

The main remaining reviewer-facing weakness is the existing relearning audit's unmatched starting utility. Selected follow-up: compare Llama NPO scratch S100 against K-FORGE S50 (`alpha=0.60`), whose pre-attack utility and Forget Q/A Probability are close, under identical one-epoch and three-epoch relearning attacks. The same pair will be evaluated under 4-bit and 8-bit quantization.

### 2026-07-11 15:20 UTC

The first structured control aggregate showed that K-FORGE Pareto-dominates all four Gemma NPO controls in Forget Q/A Probability and utility. For SimNPO, diagonal-Fisher and K-FORGE differ by only `0.000054` mean Forget Q/A Probability over seeds `0,1,2` (`p=0.636`), while K-FORGE has lower extraction and Forget ROUGE. To avoid selecting only a favorable competitor, the remaining confirmation is prespecified as seed `3` for all four controls and both optimizers. It will start after the quantization-repair task releases GPU 2.

### 2026-07-11 15:30 UTC

Prespecified the next free-GPU follow-up: 4-bit and 8-bit evaluation of the four-seed Gemma S100 scratch/K-FORGE pairs for both NPO and SimNPO. This combines the two main reviewer gaps (non-Llama transfer and quantization robustness) without changing training or selecting a favorable seed.

### 2026-07-11 16:00 UTC

Prespecified an optimizer-transfer check for the positive quantization result: evaluate the existing three-seed Llama SimNPO S50 scratch/K-FORGE pair under the identical 8-bit and 4-bit protocol. This was selected before observing any SimNPO quantized result. The purpose is to distinguish an NPO-specific effect from quantization robustness of the initializer more generally; all four metrics will be reported regardless of direction.

### 2026-07-11 16:18 UTC

Prespecified a held-out compute-matching completion on Gemma: run scratch NPO S103 and scratch SimNPO S105 for seed `3`, then compare against the already completed held-out K-FORGE S100 seed. This extends the central R1 comparison to the independently added seed without changing the measured setup charge or any optimizer setting. Both results will be included regardless of direction.

### 2026-07-11 16:28 UTC

Prespecified optimizer transfer for the active recovery audit: compare SimNPO scratch S100 with K-FORGE SimNPO S50 under the same one-epoch (13-step) and three-epoch (39-step) supervised relearning protocol over seeds `0,1,2`. These checkpoints are close in the existing standard evaluation (Forget Probability `0.545/0.568`, utility `0.585/0.576`); the attack-run `checkpoint-0` evaluations will be used as the definitive matched starting measurements. Both durations and all four metrics will be reported regardless of direction.

### 2026-07-11 16:55 UTC

Prespecified a model-family recovery audit before observing any attack result: compare Gemma NPO scratch and K-FORGE at S50 over seeds `0,1,2`. These are the closest existing Gemma operating points (standard-evaluation Forget Probability `0.07436/0.07434`, utility `0.34903/0.35804`). Both arms receive identical one-epoch and three-epoch supervised `forget10` attacks; attack-run `checkpoint-0` measurements are definitive. This combines the model-family and robustness questions without selecting a favorable post-attack duration, and all four metrics will be reported regardless of direction.

### 2026-07-11 17:05 UTC

Considered repeating the 1B end-to-end timing measurement. No dedicated timing harness or preserved cold-cache protocol exists, so an ad hoc rerun would mix model-cache, download, and checkpoint-write effects with the quantity being compared. Retained the saved end-to-end measurement and its conservative five-step wall-clock charge instead of manufacturing a less comparable variance estimate. Remaining GPU capacity stays allocated to the prespecified recovery audits.

### 2026-07-11 17:08 UTC

The first Gemma relearning attempt failed during checkpoint-13 evaluation with CUDA OOM. Root cause was device mapping, not the experiment: the new runner omitted `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so `GPU_ID=2` resolved to an almost-full H200 rather than the free physical GPU 2 used by the earlier harness. Added only the deterministic PCI mapping, quarantined the partial output, and scheduled an exact retry with unchanged checkpoints, seeds, attack durations, dtype, batch size, and optimizer settings.

### 2026-07-11 17:13 UTC

Before observing any completed Gemma attack result, extended the matched recovery design to SimNPO S50 over seeds `0,1,2`. The existing standard-evaluation starting points are close (scratch/K-FORGE Forget Probability `0.27334/0.27005`, utility `0.40401/0.40230`). Both one- and three-epoch attacks are fixed in advance and will run after Gemma quantization releases GPU 1; all metrics and both durations will be retained regardless of direction.

### 2026-07-11 17:16 UTC

Implemented R1's concrete Algorithm 1 readability request by splitting only the cross-map, two-SVD, and target/truncation lines. A PDF verification attempt with `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` stops before processing the manuscript because `acl.sty` is absent from the repository/environment. This is recorded as an artifact dependency gap; no PDF-build success will be claimed unless the style file is restored.

### 2026-07-11 17:20 UTC

Checked the current [official ARR author guidance](https://aclrollingreview.org/authors). It requires a text-only response with no external links, recommends keeping discussion focused because ACs are not expected to read long threads, and permits new experiments only when they directly answer a reviewer's question. Every retained experiment maps to an explicit request here (compute matching and model family for R1; completed recovery audits and metric alignment for R2; MUSE/recovery for R3), but the current `REBUTTAL.md` is a 4.1k-word evidence dossier rather than a portal-ready comment. Decision: keep it as the complete numeric source, then produce a separate concise per-reviewer portal version after final aggregation; do not add an external artifact URL to that version.

Created `REBUTTAL_PORTAL.md` as three standalone comments: R1/R2/R3 contain 637/596/505 words (1,770 total), no external URLs, and only four balanced block-math delimiters. It retains the compute formula and central tables, complete adverse metric directions, MUSE's mixed outcome, and the negative matched-relearning conclusion while removing the dossier's repeated notation and duplicated explanations. Pending prespecified audits may update individual numbers or one boundary sentence, but not the response structure.

## Results

### Verified completed additions (`n=3` unless noted)

The structured aggregator parsed all 64 expected summaries.

**Gemma NPO S100 controls.** K-FORGE has Forget Q/A Probability `0.051835` and utility `0.403403`. The corresponding control points are:

| Initializer | Forget Prob. | Utility | Extraction | Forget ROUGE |
|---|---:|---:|---:|---:|
| Random rank-2 | 0.066970 | 0.397471 | 0.044397 | 0.354941 |
| Weight-SVD | 0.052258 | 0.327054 | 0.035598 | 0.374652 |
| Diagonal Fisher | 0.063760 | 0.396721 | 0.062102 | 0.323942 |
| Forget-only Fisher | 0.057569 | 0.391051 | 0.034985 | 0.369389 |
| **K-FORGE** | **0.051835** | **0.403403** | **0.031972** | 0.366052 |

K-FORGE Pareto-dominates every tested NPO control in Forget Probability and utility. It also has the lowest extraction, but diagonal Fisher has lower Forget ROUGE; the latter prevents a metric-uniform claim.

**Gemma compute matching.** Charging the measured setup gives scratch budgets S103 for NPO and S105 for SimNPO. Relative to these longer scratch runs:

- NPO K-FORGE S100 changes Forget Probability `0.063501 -> 0.051835`, utility `0.400552 -> 0.403403`, and extraction `0.052999 -> 0.031972`; Forget ROUGE worsens `0.320014 -> 0.366052`.
- SimNPO K-FORGE S100 changes Forget Probability `0.272284 -> 0.269373` (`p=4.28e-4`), utility `0.410011 -> 0.409476`, extraction `0.125893 -> 0.121370`, and Forget ROUGE `0.406739 -> 0.397400`.

The held-out NPO compute-matched seed `3` independently preserves the primary direction: scratch S103/K-FORGE S100 Forget Probability is `0.065376/0.057618`, utility `0.400667/0.394661` (delta `-0.006006`, inside the `-0.01` margin), extraction `0.052138/0.036018`, and Forget ROUGE `0.304473/0.360671`. Over all four seeds, means are FP `0.063970/0.053281` (`p=0.00479`), utility `0.400581/0.401218`, extraction `0.052783/0.032983` (`p=5.49e-4`), and ROUGE `0.316129/0.364707`. The probability/extraction result survives the full setup charge; ROUGE remains the explicit adverse metric.

The held-out SimNPO compute-matched seed `3` also preserves the direction: scratch S105/K-FORGE S100 Forget Probability is `0.272068/0.269072`, utility `0.409373/0.407400`, extraction `0.127627/0.123528`, and Forget ROUGE `0.414345/0.386989`. Over all four seeds, means are FP `0.272230/0.269298` (`p=9.39e-6`), utility `0.409852/0.408957`, extraction `0.126326/0.121909`, and ROUGE `0.408640/0.394797`. This is a small but highly consistent compute-matched probability gain with utility delta `-0.000895`.

**Llama FLOP matching.** The TOFU `forget10` training split averages `95.0725` tokens/example and `retain90` averages `91.3647`, measured with the exact local Llama tokenizer and training template. At effective batch size 32, this gives `tau_f=3042.32` and `tau_r=2923.67` valid tokens per optimizer step. Using the same parameter-token convention as the existing setup estimate, with `P=1,235,814,400`, SimNPO costs approximately `6P(tau_f+tau_r)=4.424e13` FLOPs/step and NPO adds a reference forward pass, for `5.176e13` FLOPs/step. The `1.435e14` K-FORGE setup is therefore `3.24` SimNPO or `2.77` NPO steps. The existing five-step wall-clock charge is stricter and thus covers both budgets. Of the setup estimate, calibration model passes account for `96.2%` at 1B and `98.3%` at 3B; covariance plus dense factorization accounts for `3.8%/1.7%`.

The central Llama compute-matched table was re-aggregated from the final FP32 summaries rather than internal training-time evaluations. NPO paired Forget-Probability comparisons at K-FORGE S50/S100/S250 versus scratch S55/S105/S255 are `0.078287/0.045373` (`p=0.00486`), `0.041117/0.030760` (`p=0.00214`), and `0.027568/0.022316` (`p=0.0126`). SimNPO values are `0.646187/0.534098` (`p=3.44e-4`), `0.503043/0.406559` (`p=0.00136`), and `0.328960/0.271863` (`p=0.00334`). Utility deltas are positive for all NPO budgets and `-0.00838/-0.00547/-0.00398` for SimNPO, all inside the `-0.01` margin. NPO extraction worsens at all three budgets; SimNPO extraction and ROUGE improve at all three, except that the S250 extraction difference is noisy (`p=0.121`).

The paired wall-clock measurements were run on the same single-GPU NVIDIA RTX PRO 6000 Blackwell setup (GPU ordinals 1/2 on this host, which are the same device class); no multi-GPU parallelism is included in the reported times.

**Artifact verification.** Added `tests/test_kforge_wiener.py`, which constructs a small random two-Fisher problem and checks the implemented full-rank Wiener edit against a direct autograd-Hessian solve of the quadratic objective. `PYTHONPATH=src python tests/test_kforge_wiener.py` passes (`Ran 1 test ... OK`) using only the standard-library test runner.

Added `tests/test_rebuttal_compat.py` for the two compatibility paths exercised by the new model/evaluation runs: tokenizer outputs normalize to flat token-ID lists, and bfloat16 probability evaluation returns finite Python floats without NumPy's unsupported bfloat16 conversion. `PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'` passes all three tests.

**Held-out Gemma seed.** The four-seed aggregates remain positive. NPO Forget Probability falls by `18.74%` with mean utility change `+0.00060`; SimNPO falls by `1.14%` with utility change `-0.00154` and paired `p=8.79e-7`.

**MUSE-News selected follow-up.** At S100, the predeclared `alpha=1.0` point improves VerbMem ROUGE `0.572742 -> 0.550708` and extraction `0.303386 -> 0.301164`, but worsens KnowMem by `0.000755` and retain ROUGE by `0.012093`. This remains partial transfer, not benchmark dominance.

**Quantization-revert audit after same-step NPO S50.** The correct attack statistic is post-minus-pre within each arm. Under 8-bit loading, Forget Probability changes by `+0.00427` for scratch and `+0.00175` for K-FORGE; extraction changes by `+0.00102` and `-0.00008`. Under 4-bit loading, scratch shows substantially more recovery: Forget Probability changes by `+0.08925` versus `+0.00719` for K-FORGE, and extraction by `+0.02317` versus `-0.00093`. K-FORGE's 4-bit utility falls by `0.03772`, whereas scratch utility rises by `0.05152`, so the recovery difference is accompanied by an opposing utility response. Forget ROUGE increases in both arms (`+0.04234` scratch, `+0.01802` K-FORGE). The defensible claim is that K-FORGE's probability/extraction forgetting is substantially less reverted by 4-bit loading, not that every metric is quantization-invariant.

**Matched-start one-epoch relearning (`n=3`).** The attack-run `checkpoint-0` evaluations provide the directly comparable starting measurements: scratch/K-FORGE Forget Probability is `0.04721/0.05191`, utility `0.56835/0.57540`, extraction `0.06652/0.07281`, and ROUGE `0.26780/0.27236`. After the same 13-step forget-set fine-tuning attack, scratch/K-FORGE values are:

| Metric | Scratch post | K-FORGE post | Recovery scratch | Recovery K-FORGE |
|---|---:|---:|---:|---:|
| Forget Probability | **0.37572** | 0.54639 | +0.32851 | +0.49447 |
| Extraction | **0.11218** | 0.20021 | +0.04567 | +0.12740 |
| Forget ROUGE | **0.37551** | 0.47405 | +0.10771 | +0.20170 |
| Utility | 0.45526 | **0.49256** | -0.11308 | -0.08284 |

K-FORGE recovers more forget behavior at the matched starting point, while retaining `+0.0373` higher post-attack utility. Therefore the earlier unmatched audit cannot support an intrinsic relearning-resistance claim; its lower post-attack forgetting was largely associated with a much stronger pre-attack forgetting point. This negative result will be disclosed as the relearning boundary, while quantization remains the positive robustness result.

At three epochs (39 attack steps), the conclusion strengthens: scratch/K-FORGE post-attack Forget Probability is `0.72977/0.91101`, extraction `0.40668/0.81227`, and Forget ROUGE `0.62192/0.88416`; K-FORGE retains slightly higher utility (`0.50967` versus `0.49324`). K-FORGE is therefore not relearning-resistant at a matched initial operating point.

**Matched-start quantization (`n=3`).** Quantizing the same scratch S100/K-FORGE S50 pair produces only small recovery in either arm. Under 8-bit loading, scratch/K-FORGE Forget Probability changes by `+0.00181/+0.00164`, extraction by `-0.00041/-0.00129`, and Forget ROUGE by `+0.00236/+0.00230`. Under 4-bit loading, the corresponding changes are `+0.00843/+0.01168`, `-0.00416/-0.00757`, and `-0.00287/-0.00487`. The 4-bit post-quantization Forget Probability is `0.05564/0.06359`. Thus the K-FORGE checkpoint itself remains largely stable under quantization, but matching the starting operating point removes the apparent comparative quantization-resistance advantage seen in the ordinary same-step S50 comparison. This is a stability result, not evidence that K-FORGE is intrinsically more quantization-resistant than scratch.

**SimNPO quantization transfer (`n=3`).** The ordinary matched-budget S50 advantage survives both precisions. After 8-bit loading, scratch/K-FORGE values are Forget Probability `0.69356/0.56101` (`p=1.03e-4`), utility `0.57474/0.57270`, extraction `0.28445/0.19243`, and Forget ROUGE `0.54751/0.46461`. After 4-bit loading they are `0.56286/0.47247` (`p=4.47e-4`), `0.53199/0.52797`, `0.17774/0.13730`, and `0.46801/0.42706`. Quantization decreases the forgetting metrics in both arms rather than reverting them; K-FORGE's optimizer advantage remains, with post-quantization utility differences of only `-0.00205/-0.00402`. This supports persistence of the matched-budget SimNPO gain, not intrinsic recovery resistance at a matched initial operating point.

**Held-out Gemma NPO control confirmation (`n=4`).** The prespecified fourth seed preserves the control conclusion without producing a metric-uniform result:

| Initializer | Forget Prob. | Utility | Extraction | Forget ROUGE |
|---|---:|---:|---:|---:|
| Random rank-2 | 0.065787 | 0.397174 | 0.048421 | 0.342786 |
| Weight-SVD | 0.052970 | 0.327018 | 0.036685 | 0.376702 |
| Diagonal Fisher | 0.063739 | 0.396396 | 0.061314 | 0.324368 |
| Forget-only Fisher | 0.058497 | 0.391344 | 0.035299 | 0.364314 |
| **K-FORGE** | **0.053281** | **0.401218** | **0.032983** | 0.364707 |

Weight-SVD is numerically `0.00031` lower on Forget Probability, but loses `0.07420` utility relative to K-FORGE and fails the prespecified `-0.01` utility non-inferiority margin. Among utility-feasible controls, K-FORGE has the lowest Forget Probability and extraction. Paired comparisons against K-FORGE give Forget-Probability `p`-values `0.0182` (random), `0.889` (weight-SVD), `0.00588` (diagonal), and `0.0989` (forget-only). Diagonal Fisher remains better on Forget ROUGE, so the evidence supports a curvature-aware, utility-constrained initialization claim rather than dominance on every forgetting metric.

**Existing Llama-3.1-8B pilot (`n=1`).** A previously completed single-seed SimNPO pilot is near-neutral and is retained here as an adverse/weak scaling result. At S50, scratch/K-FORGE Forget Probability is `0.959309/0.956319`, utility `0.634750/0.635019`, extraction `0.909606/0.901068`, and Forget ROUGE `0.947492/0.943456`. At S100, the corresponding values are `0.923775/0.921425`, `0.633971/0.635657`, `0.854938/0.860138`, and `0.902423/0.905872`. The S50 direction is uniformly favorable but tiny; at S100 extraction and ROUGE reverse. One seed cannot support a scaling claim, and the result does not alter the rebuttal's Gemma-based model-family evidence.

**Held-out Gemma SimNPO control confirmation (`n=4`).** The completed four-seed means are:

| Initializer | Forget Prob. | Utility | Extraction | Forget ROUGE |
|---|---:|---:|---:|---:|
| Random rank-2 | 0.271597 | 0.408475 | 0.126162 | 0.409018 |
| Weight-SVD | **0.263222** | 0.390092 | 0.126231 | 0.401297 |
| Diagonal Fisher | 0.269237 | **0.411584** | 0.124201 | 0.407310 |
| Forget-only Fisher | 0.269995 | 0.403567 | 0.125218 | 0.405583 |
| K-FORGE | 0.269298 | 0.408957 | **0.121909** | **0.394797** |

Weight-SVD's lower Forget Probability comes with utility `-0.018865` below K-FORGE and fails the prespecified utility margin. Diagonal Fisher is numerically `0.000061` lower on Forget Probability and `0.002627` higher on utility; the paired FP difference is not distinguishable (`p=0.445`). K-FORGE instead has the lowest extraction and Forget ROUGE. Random and forget-only are worse than K-FORGE on all three forgetting metrics, with similar or lower utility. The full Kronecker structure is therefore not universally necessary for SimNPO's probability endpoint, although it improves the secondary forgetting metrics in this comparison.

**Gemma NPO quantization (`n=4`).** This is an adverse model-specific robustness result. Under 8-bit loading, scratch/K-FORGE post values are FP `0.069887/0.069674`, utility `0.387626/0.389979`, extraction `0.053204/0.034160`, and ROUGE `0.302501/0.367899`. Relative to each arm's own pre-quantization checkpoint, FP changes by `+0.004315/+0.016394`; the original K-FORGE FP advantage is erased, although its extraction advantage remains. Under 4-bit loading, post values are FP `0.058025/0.061096`, utility `0.326869/0.295302`, extraction `0.035095/0.030459`, and ROUGE `0.329604/0.399425`. The K-FORGE arm has worse FP and `-0.03157` lower utility after 4-bit loading; only extraction remains numerically lower. Gemma NPO therefore does not support a quantization-persistence claim.

**Gemma SimNPO quantization (`n=4`).** Unlike NPO, the small matched-budget SimNPO FP advantage persists. At 8-bit, scratch/K-FORGE post values are FP `0.268583/0.265690` (`p=0.00340`), utility `0.403294/0.402467`, extraction `0.123402/0.120862`, and ROUGE `0.402164/0.402071`. At 4-bit they are FP `0.218647/0.214105` (`p=0.0238`), utility `0.388021/0.389617`, extraction `0.102340/0.100695`, and ROUGE `0.418581/0.417114`. Quantization lowers FP in both arms rather than recovering forgotten behavior. The complete Gemma matrix therefore supports an optimizer-dependent conclusion: SimNPO's small advantage is stable, whereas NPO's probability advantage is not.

**Llama SimNPO matched-start relearning, one epoch (`n=3`).** Attack-run pre values are close: scratch/K-FORGE FP `0.545205/0.568480`, utility `0.584560/0.574679`, extraction `0.183915/0.201914`, and ROUGE `0.480119/0.466382`. After the same 13 attack steps, post values are FP `0.802680/0.874356`, utility `0.559811/0.563943`, extraction `0.539263/0.694190`, and ROUGE `0.707898/0.815344`. K-FORGE has more recovery on every forgetting metric: FP `+0.305876` versus `+0.257475` (`p=0.0216` for the paired recovery difference), extraction `+0.492277` versus `+0.355347` (`p=0.00759`), and ROUGE `+0.348962` versus `+0.227778` (`p=0.00431`). Its utility falls less (`-0.010736` versus `-0.024750`). The NPO relearning limitation therefore transfers to SimNPO rather than being optimizer-specific; the prespecified three-epoch arm remains in progress.

### 2026-07-11 17:35 UTC

Replaced the stale robustness block in `main.tex`. The submitted draft still claimed that K-FORGE recovered less under relearning and left quantization as future work; both statements contradicted the stricter completed audits. The revised block now reports the matched-start NPO result as a negative recovery finding, includes post-attack values after 13 and 39 steps, reports the Llama 4/8-bit NPO and SimNPO matrix, and states the adverse Gemma NPO quantization boundary. The old layer-7 and budget-dependent tables were removed from the main narrative because they used unmatched starting points and could support a misleading robustness implication.

All three pending relearning queues remain live: Llama SimNPO three epochs, Gemma NPO one/three epochs, and Gemma SimNPO one/three epochs. The watchdog continues at ten-minute intervals. No failed or partial run is being counted as evidence.

### 2026-07-11 17:49 UTC: hostile response audit, pass 1 of 3

Read each portal response as a reviewer looking for reasons not to raise the score.

| Reviewer | Score before edits | Main remaining objection |
|---|---:|---|
| R1 | 7.5/10 | Compute matching gave a five-step charge without exposing measured per-step time/hardware; Gemma's `p=.0048` could be mistaken for passing the exploratory threshold; the manuscript still had an obsolete two-seed 3B table. |
| R2 | 8.0/10 | The response aligned metrics well, but the manuscript itself still contained the opposite relearning claim and omitted the new model/benchmark results, making the rebuttal look disconnected from the revision. |
| R3 | 8.5/10 | MUSE and recovery are directly addressed, but the distinction between passive quantization stability and active recovery resistance must remain explicit in every summary claim. |

Actions taken: added the measured single-GPU scratch times (`691.5/629.1 s` for 50-step NPO/SimNPO), derived the explicit five-step wall-clock cover (`69.2/62.9 s > 61.2 s`), marked Gemma NPO `p=.0048` descriptive and stated the `p<.001` exploratory threshold, replaced the manuscript's stale robustness section, inserted compute-matched Gemma and complete MUSE evidence, and expanded the Llama-3.2-3B appendix table to all three seeds/budgets/four metrics. The Qwen near-null result and all adverse NPO ROUGE/extraction directions remain visible.

Also corrected a mathematical overstatement near Algorithm 1: for `alpha != 1`, the scaled returned checkpoint is not itself the minimizer of the quadratic objective. The revised text attributes exactness only to the unscaled edit in the proved limits.

**Gemma NPO matched-start relearning, one epoch (`n=3`).** The completed cross-family audit starts from nearly identical mean FP (`0.074433/0.074227` scratch/K-FORGE) with K-FORGE utility `+0.00751` higher. After 13 attack steps, FP is `0.261135/0.294662` and extraction is `0.119387/0.134687`. Relative recovery is larger from K-FORGE for FP (`+0.220434` versus `+0.186703`, paired difference `p=.00213`, descriptive under the `.001` threshold) and extraction (`+0.104228` versus `+0.088373`). K-FORGE has higher post-attack utility (`0.374407` versus `0.352492`). Forget ROUGE requires separate interpretation because the K-FORGE arm starts substantially worse (`0.382705` versus `0.317574`) and ends nearly tied (`0.415302/0.413998`), so its within-arm recovery is smaller. The primary FP boundary transfers to Gemma NPO: K-FORGE is not a direct-relearning defense.

**Llama SimNPO matched-start relearning, three epochs (`n=3`).** Both arms are almost fully recovered after 39 attack steps: scratch/K-FORGE post values are FP `0.977837/0.981887`, extraction `0.967602/0.977387`, and ROUGE `0.984114/0.986968`. Paired recovery differences are not distinguishable (`p=.101/.479/.117` for FP/extraction/ROUGE). The one-epoch K-FORGE disadvantage therefore does not grow indefinitely; the longer attack saturates both checkpoints near complete recovery. This still decisively rejects a relearning-resistance claim for either initialization.

**Gemma SimNPO matched-start relearning, one epoch (`n=3`).** Pre-attack scratch/K-FORGE FP is `0.273279/0.269954` and utility is `0.404426/0.402864`. After 13 steps, the two arms converge to FP `0.352655/0.352972`, utility `0.384617/0.384812`, extraction `0.145056/0.145040`, and ROUGE `0.435721/0.433061`. K-FORGE therefore recovers slightly more FP (`+0.083019` versus `+0.079376`, paired difference `p=.00248`, descriptive), erasing its small initial advantage; other post-attack metrics are effectively tied. Together with Gemma NPO and both Llama optimizers, this confirms that K-FORGE is not a direct-relearning defense across the tested model/optimizer combinations.

### 2026-07-11 18:01 UTC: hostile response audit, pass 2 of 3

| Reviewer | Score after pass-1 edits | Hostile reading |
|---|---:|---|
| R1 | 9.2/10 | The answer now supplies hardware, end-to-end setup, measured step time, FLOP conversion, a conservative budget, 3B cost, and the many-layer limitation. Remaining uncertainty is the unavoidable single-system timing measurement and approximate FLOP convention, both labeled rather than hidden. |
| R2 | 8.8/10 | Model breadth, MUSE, negative recovery evidence, quantization boundaries, and artifact details are present. The response still needed to answer the review's exact Table-2 example rather than relying on the general revised claim. |
| R3 | 9.0/10 | Both requested experiments are complete and the negative recovery result is disclosed. MUSE remains partial rather than decisive, but the response cannot honestly strengthen that evidence. |

Action taken: added an explicit statement in both the compute discussion and R2 response that Llama NPO extraction worsens at every compute-matched budget, whereas SimNPO secondary metrics generally improve. The portal now states that the NPO compute claim is FP-plus-utility-specific. No unfavorable endpoint was removed or relabeled.

Expanded artifact validation to cover the complete three-seed Llama-3.2-3B matrix and the three-seed Qwen null pilot, both of which are cited in R1. Following a red/green check, `test_rebuttal_compat.py` first failed because `tofu_eval_paths` did not exist, then passed after the minimal path helper and aggregator rows were added. A partial `--check` now parses all prior evidence, including 3B/Qwen and the completed Llama SimNPO three-epoch audit, before stopping at the expected still-running Gemma three-epoch path. Also documented the isolated `bitsandbytes==0.49.0` install required by the tested Blackwell quantization runner; the base training dependency remains unchanged.

Rechecked the current ARR author-response guidance. It specifies text-only responses without external links and emphasizes focused discussion rather than a fixed character allowance. The portal file remains one standalone comment per reviewer, currently 772/669/548 words for R1/R2/R3, with no external URLs. R1 is longer because it answers two independent major concerns with an explicit compute derivation and model-family evidence; R2/R3 avoid repeating the full compute table.

### 2026-07-11 18:39 UTC: manuscript compression and layout audit

The revised manuscript had grown to nine main-text pages before references, and six secondary diagnostic tables produced overfull boxes of 47--85 pt. I moved the budget, split, layer-placement, layer-strength, and module-target audits to the one-column appendix while preserving every row, caption, label, and main-text scope statement. Detailed Gemma compute matching, Llama compute matching, relearning, and quantization tables were likewise moved to the appendix; their protocols, exact headline values, negative outcomes, compute formula, and measured setup table remain in the main text.

Using the available ACL template plus temporary local copies of the standard algorithm styles, a clean first-pass `pdflatex` build now places Conclusion and Limitations on page 8 and begins the appendix on page 9. The prior large overfull warnings are gone; the remaining 6.1-pt one-shot-table warning was addressed by reducing only that table to `\scriptsize` and will be rechecked in the final build. The paper's actual `custom.bib` and style bundle are absent from this checkout, so citations cannot be resolved here; this is an external bundle limitation rather than a manuscript syntax failure. No generated build products are included in the intended commit.

The main narrative now matches the rebuttal boundary: K-FORGE improves the primary Forget Q/A Probability trajectory under a full compute charge, but NPO extraction can worsen, model/benchmark transfer is uneven, and matched relearning does not support recovery resistance. This avoids spending scarce main-text space on post-hoc diagnostic grids while retaining reviewer-requested compute and robustness evidence.

### 2026-07-11 19:26 UTC: final experiment closure and snapshot

The final prespecified Gemma SimNPO three-epoch relearning run completed normally. All six seed/arm rows point to valid `checkpoint-39` summaries and the queue exited without an error signature. The three-seed aggregate is:

| Metric | Scratch post | K-FORGE post | Recovery scratch | Recovery K-FORGE | paired recovery $p$ |
|---|---:|---:|---:|---:|---:|
| Forget Probability | 0.658469 | 0.665073 | +0.385191 | +0.395120 | 0.0364 |
| Utility | 0.396184 | 0.393306 | -0.008242 | -0.009559 | 0.249 |
| Extraction | 0.283434 | 0.296008 | +0.156732 | +0.173860 | 0.0755 |
| Forget ROUGE | 0.597211 | 0.598278 | +0.191011 | +0.196158 | 0.0982 |

All differences except the already reported pre-attack FP comparison are descriptive under the $p<.001$ rule. The longer Gemma attack erases the initial SimNPO advantage and leaves K-FORGE numerically worse on all three forgetting metrics; it does not support recovery resistance.

The final structured aggregation passed with `386` expected reads spanning `349` unique per-seed summary files. Exported `open-unlearning/rebuttal_metrics_snapshot.json` contains only structured summary values (`151,378` bytes; SHA-256 `5dddca887d95c7d41f1f61a516158eece72252e94b0e60dad36d5b5f46478364`). Running the aggregator from this snapshot without raw runs produced output byte-for-byte identical to direct aggregation.

During runner review, found that the original relearning skip guard selected the numerically latest summary present. After an interrupted run, this could mistake a valid `checkpoint-0` pre-attack evaluation for a completed attack. The Llama, Gemma NPO, and Gemma SimNPO runners now require the exact final `checkpoint-$((13 * EPOCHS))` summary for both `skipped` and `ok` states. Shell syntax validation passes for all three. This does not change any completed result; it closes a retry/reproducibility failure mode.

The documented test command, run from `open-unlearning`, passes all six CPU tests. An intentionally different invocation from the repository root failed because `scripts` is not on that invocation's module path; reproducing the README command confirmed this was a command-context issue rather than a code defect.

### 2026-07-11 19:30 UTC: hostile response audit, pass 3 of 3

| Reviewer | Final response score | Remaining hostile objection |
|---|---:|---|
| R1 | 9.3/10 | Timing is measured on one hardware class and one edited layer; broad multi-layer scaling is still untested. The response exposes both limits, charges full setup without amortization, and supplies independent FLOP matching. |
| R2 | 9.3/10 | MUSE transfer is mixed and NPO secondary metrics can be adverse. The response directly narrows the claim instead of hiding these endpoints, completes both robustness audits, and supplies a reproducible snapshot. |
| R3 | 9.2/10 | The requested MUSE result is partial rather than a benchmark win. The response nevertheless answers the request with both domains/budgets and gives a clear negative recovery result across optimizers and model families. |

Final edits from this pass: state explicitly that the K-FORGE edit is folded into model weights and adds no per-step or inference overhead; soften the manuscript conclusion from “Gemma confirms” to “Gemma provides” cross-family evidence; and describe the MUSE follow-up precisely as using a selection rule fixed before downstream training rather than implying that the entire benchmark choice was preregistered. No adverse result was removed. The remaining external artifact risk is that the anonymous 4open endpoint cannot be authenticated from this environment; the owner must upload the revised package and verify the anonymous view separately.

### 2026-07-11 19:38 UTC: final verification matrix

- `PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py' -v` passes all six CPU tests.
- `py_compile` passes for the aggregator, four modified source modules, and both test files.
- `bash -n` passes for all eight new or modified rebuttal/initialization runners.
- Direct aggregation and snapshot-only aggregation both report `PASS: parsed 386 expected summaries`; their complete outputs are byte-for-byte identical.
- All 13 experiment manifests end in `ok`, `skipped`, or `skipped_existing` for 106 distinct logical rows, and every referenced summary contains finite values for the four mandatory metrics.
- The snapshot contains 349 summaries and 2,389 finite numeric aggregate values; it contains no prompts, generated text, absolute host paths, or credentials.
- Source-backed checking of the six central compute-matched rows exposed five relative percentages that had been calculated from rounded table means. Recomputing from full per-seed precision changed NPO 1B S100/S250 from `25.1/19.2%` to `25.2/19.1%`, and 3B NPO S50/S250 plus SimNPO S100 from `24.8/-9.0/12.0%` to `24.7/-9.3/11.9%`. Absolute metrics and conclusions are unchanged. The corrected central table now matches source means and exact relative reductions.
- Markdown checks find balanced display-math delimiters and tables; the portal file has exactly two display-math blocks and no external URLs.
- Static LaTeX checks find balanced braces, 47 unique labels, and 49 resolved internal references. Two fresh `pdflatex` passes produce 15 pages with no overfull boxes; main text ends on page 8 and the appendix begins on page 9. Visual inspection of all main-text pages and the appendix transition finds no overlap or clipping.
- The checkout lacks `custom.bib`, so local citation resolution remains impossible; this is recorded rather than hidden. Generated LaTeX products were removed before staging.
- No `rebuttal10_*` experiment or watchdog session remains running. Unrelated user tmux sessions were left untouched.
