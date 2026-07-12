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

- `PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py' -v` passes all seven CPU tests. The added proof-regression check verifies that the zero-penalty rank-$r$ implementation has rank at most $r$ and exactly reaches the Eckart--Young residual in the whitened basis, complementing the existing full-rank direct-Hessian test.
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

The matched-control construction was also checked against the actual saved Gemma checkpoints, not only its metadata. K-FORGE, random, and weight-SVD modify only `model.layers.13.mlp.down_proj`; all use rank 2. Relative to the same base checkpoint, their saved FP32 Frobenius norms are `3.626694`, `3.627478`, and `3.626677`, respectively (maximum difference below `0.022%`). Random and weight-SVD have zero measurable residual beyond rank 2; K-FORGE's randomized-SVD residual ratio is `5.4e-4`. This directly supports the same-module/rank/norm control claim.

### 2026-07-11 20:04 UTC: prespecified all-input-token compute confirmation

A final accounting audit found that the earlier analytical calibration estimate counted `18.6k` loss-bearing response tokens. Forward/backward computation also processes prompt tokens. Re-tokenizing the exact TOFU calibration protocol gives approximately `47.7k` input tokens across the 512 forget/retain examples. Under the same conservative `6PT` calibration convention, the corrected one-time setup estimate is approximately `3.59e14` FLOPs for Llama-3.2-1B and `9.26e14` FLOPs for Llama-3.2-3B. Calibration dominates these estimates (`98.4%` and `99.3%`, respectively); dense factor operations contribute the remainder.

Before observing any new outcomes, I fixed a stricter comparison protocol: compare K-FORGE runs at 50/100/250 downstream steps against scratch runs at 60/110/260 steps for the same optimizer, data, FP32 model, and three seeds. Ten extra scratch steps exceed both the all-input-token FLOP charge (approximately 6.95 NPO or 8.13 SimNPO steps at 1B) and the measured calibration-through-edit wall-clock charge (fewer than five steps), with no amortization across runs. The later command-to-checkpoint audit refines the complete wall-clock bound to at most six NPO or seven SimNPO steps, still below ten. All four prespecified endpoints will be retained: Forget Q/A Probability (primary), Model Utility, extraction strength, and Forget Q/A ROUGE. Relative forget reductions will be calculated from full-precision seed means; adverse directions will not be suppressed.

The original base checkpoint was no longer present locally. It was reconstructed by subtracting the deterministic saved rank-2 random initialization from its checkpoint. Independent reconstructions from seeds 1 and 2 agree with the recovered target tensor to relative Frobenius error about `7.1e-7` (maximum element error `7.5e-9`), and all non-target weights are identical. The recovered checkpoint is used only to regenerate scratch baselines; it does not affect any K-FORGE arm.

The FLOP convention was then checked against the trainer implementation. Both NPO and SimNPO run forward/backward passes on forget and retain batches; NPO additionally evaluates the frozen reference model on the forget batch. This confirms
`F_SimNPO = 6P(tau_f+tau_r)` and `F_NPO = F_SimNPO+2P tau_f`. A checked calculator at `open-unlearning/scripts/estimate_kforge_compute.py` reproduces the setup and step estimates. For 1B, the corrected setup is `6.95` NPO or `8.13` SimNPO steps, so the prespecified ten-step charge remains conservative for both FLOPs and wall-clock time.

**First all-token checkpoint (three seeds, scratch S60 versus K-FORGE S50).** NPO scratch/K-FORGE means are FP `0.072231/0.045373`, utility `0.539452/0.573835`, extraction `0.068461/0.070551`, and ROUGE `0.246537/0.269792`. This is a `37.2%` relative FP reduction and `+0.03438` utility, with adverse secondary forgetting metrics. SimNPO means are FP `0.630792/0.534098`, utility `0.580250/0.571165`, extraction `0.245438/0.188034`, and ROUGE `0.515074/0.452531`: a `15.3%` FP reduction and `-0.00909` utility, with favorable extraction and ROUGE. Every seed preserves the FP direction for both optimizers; the prespecified mean-utility margin is met, narrowly for SimNPO. The S110/S260 confirmations remain in progress.

**Estimator-description audit.** The implementation and its historical logs confirm that every reported checkpoint streams K-FAC-style factors `A=E[xx^T]` and `B=E[gg^T]` over shifted loss-bearing token rows. It does not execute the Lanczos/rearranged-Fisher MFF procedure described in the submitted draft. This mismatch had already been identified in the repository's earlier internal audit but remained in `main.tex`. The revision now matches the executed method: it defines the covariance estimator explicitly, changes Algorithm 1 from `MFF` to `KronFisher`, states accumulation/storage costs `O(N(m^2+n^2))`/`O(m^2+n^2)` and dense factor cost `O(m^3+n^3)`, and retains MFF/GFWSVD only as low-rank-factorization context. The theorem requires SPD Kronecker factors and is unchanged. I also corrected damping to the implemented trace-scaled form `C + epsilon max(tr(C)/d, 1e-12) I`. No checkpoint, metric, or experimental selection changed as a result of these documentation fixes.

### 2026-07-11 20:49 UTC: explicit factor-compute accounting

The compute calculator now exposes the factor-accumulation term instead of absorbing it into a generic dense envelope. The convention is

`F_KF ~= 6 P T_cal + 2 N_F (m^2+n^2) + 10(m^3+n^3)`,

where `T_cal=47,727.9` estimates non-padding prompt and answer tokens processed by calibration from the exact tokenizer/template length distribution, while `N_F=18,614` is the exact loss-bearing-row total (`9,515+9,099`) recorded in the headline run log. The resulting one-layer setup estimates round to `3.62e14` FLOPs at 1B and `9.29e14` at 3B, equal to `7.00/8.19` and `6.90/8.08` NPO/SimNPO steps. A red/green unit check first failed on the absent `factor_token_rows` argument, then passed after the calculator returned model-pass, factor-accumulation, and dense-algebra components separately.

As a sensitivity check, I also counted dynamic-padding overhead from the exact TOFU/tokenizer length distributions at the calibration batch size of 8 and downstream microbatch size of 4. Expected processed positions are about `58.9k` for the two calibration passes and `3,544/3,393` per forget/retain optimizer step. Applying the same formula raises the largest setup ratio to approximately `8.65` SimNPO steps (`7.39` NPO), still below the outcome-blind uniform ten-step charge fixed before the new runs. The main accounting retains the conventional non-padding parameter-token definition and records this padded-position calculation as a robustness check rather than silently switching conventions after seeing results.

### 2026-07-11 21:10 UTC: all-input-token compute confirmation complete

All 18 new scratch evaluations completed with `ok` manifests and no error/OOM signature. The three seed queues exited normally; the ten-minute watchdog was stopped only after all required summaries validated. K-FORGE at 50/100/250 steps is compared against scratch at 60/110/260 steps:

| Method | KF/scratch steps | Scratch FP | K-FORGE FP | Relative FP reduction | Utility delta | FP paired p |
|---|---:|---:|---:|---:|---:|---:|
| NPO | 50/60 | 0.072231 | 0.045373 | 37.18% | +0.034383 | 0.00155 |
| NPO | 100/110 | 0.039470 | 0.030760 | 22.07% | +0.017747 | 0.00713 |
| NPO | 250/260 | 0.027547 | 0.022316 | 18.99% | +0.011089 | 0.0136 |
| SimNPO | 50/60 | 0.630792 | 0.534098 | 15.33% | -0.009086 | 0.000671 |
| SimNPO | 100/110 | 0.491669 | 0.406559 | 17.31% | -0.006830 | 0.000481 |
| SimNPO | 250/260 | 0.327671 | 0.271863 | 17.03% | -0.004944 | 0.00260 |

Every one of the 18 paired-seed FP differences is favorable. All six mean utility changes satisfy the prespecified `-0.01` margin, although one SimNPO seed at each of the first two budgets is below that margin; the claim is correctly stated for the prespecified mean endpoint, not per seed. Only the first two SimNPO FP tests meet the exploratory `p<.001` threshold. The NPO rows and SimNPO S250 remain descriptive despite seed consistency.

The secondary outcomes remain visible. NPO extraction worsens at all three budgets (`+0.002091/+0.008803/+0.010247`); ROUGE worsens at S50 (`+0.023255`) and is effectively tied at S100/S250 (`-0.000912/-0.000832`). SimNPO improves extraction (`-0.057404/-0.020096/-0.003843`) and ROUGE (`-0.062543/-0.041796/-0.034322`) at every budget. This supports an FP-plus-utility initializer claim, not metric-wide dominance.

The expanded direct aggregator now passes `422` expected reads over `367` unique summaries. The regenerated snapshot is `159,309` bytes with SHA-256 `f15fde9f849e1dfa6283fda338bcae5f28186a98f06444ca37596cbf31b2730d`. Snapshot-only aggregation reports the same expected count and is byte-for-byte identical to direct aggregation.

### 2026-07-11 21:18 UTC: post-confirmation hostile audit

I reread each final portal response against the original review rather than against our intended narrative.

| Reviewer | Final hostile score | Residual concern and treatment |
|---|---:|---|
| R1 | 9.4/10 | The submitted estimator description did not match the released K-FAC implementation. The response now discloses this directly, states the corrected complexity, notes that all old/new checkpoints used the same implementation, and explains that the SPD-factor derivation is unchanged. Compute matching uses a uniform outcome-blind ten-step charge, reports adverse secondary metrics, and includes Gemma plus a Qwen null result. Remaining limits are one hardware class and one edited layer, both explicit. |
| R2 | 9.5/10 | MUSE is mixed and active relearning is unfavorable, but these are now completed evidence rather than promises. The exact Table-2 extraction objection is answered: NPO extraction worsens even when FP and utility improve, and the revised claim is metric-specific. The artifact response names executable runners, failure checks, proof regressions, and the structured snapshot. |
| R3 | 9.4/10 | The requested additional benchmark is not a uniform win, and recovery resistance is decisively negative. The answer nevertheless reports both MUSE domains/budgets and matched-start attacks across optimizers/model families, making the boundary credible rather than evasive. |

No endpoint was removed after this audit. The only response edit was to make R1's opening say explicitly that experiments used the originally released implementation and that the estimator terminology correction follows, preventing the later disclosure from appearing inconsistent with the opening claim.

### 2026-07-11 21:27 UTC: effective-configuration audit

Hydra's composed configuration revealed one remaining documentation mismatch. The base trainer config lists `damping=1e-4`, but `configs/experiment/unlearn/tofu/kforge.yaml` overrides it to `1e-3`; the K-FORGE launchers do not override that field. The saved headline Hydra config confirms rank 2, strength 0.60, fixed trace-scaled damping `1e-3`, retain penalty 0.01, 32 calibration batches, one module, and the Wiener-v2 edit. Its log confirms layer 0, 256 forget plus 256 retain examples, and `9,515/9,099` factor rows. The Gemma primary log independently confirms layer 13 and 16 examples per split; Qwen/MUSE launchers set 32 one-example batches. I corrected `main.tex` and the root README from `1e-4` to the executed `1e-3`. No run, selection, or metric changed.

### 2026-07-11 21:42 UTC: calibration-count wording audit

The `47.7k` calibration input-token quantity is an analytical estimate obtained from the exact local tokenizer/template length distribution, whereas `18,614` is the exact loss-bearing-row count recorded by the estimator. A dataloader replay with seed 0, batch size 8, and the saved tokenizer reproduced the headline forget side exactly: 256 examples, 9,515 loss rows, and 23,720 non-padding input tokens. Historical retain sampling depends on the older runtime's RNG state, so its exact prompt-token total cannot be reconstructed solely from the saved checkpoint. I therefore changed every current-facing description from an exact count to an estimate. The prespecified ten-step charge remains conservative under the separate dynamic-padding sensitivity estimate (8.65 SimNPO steps, the largest ratio). No compute-matched outcome or budget changed.

### 2026-07-11 21:47 UTC: anonymous artifact availability

An external request to `https://anonymous.4open.science/r/kforge-C710/` redirects to its repository API and returns HTTP 401 with `{"error":"not_connected"}`. The manuscript URL is therefore not currently usable from a fresh unauthenticated client. I did not replace it with the public GitHub remote because that would compromise submission anonymity. The local artifact itself has executable runners, a 159 KB structured metric snapshot, CPU proof tests, and reproduction commands; after the final commit is pushed, the repository owner must reconnect or recreate the anonymous 4open snapshot and verify it from a logged-out browser before posting the rebuttal. This is the only remaining manual release action.

### 2026-07-11 22:02 UTC: end-to-end wall-clock correction

Code inspection showed that K-FORGE's `train_runtime` timer ends before `train.py` calls `save_model`. A stricter filesystem audit uses the runner log's birth time as command start and the final checkpoint file's timestamp as completion. This gives complete command-to-checkpoint intervals of `80.2 s` at 1B and `145.0 s` at 3B, including model/data startup and serialization; the paper rounds these upward to `<=81/<=145 s`. Relative to measured S50 scratch runs, the bounds are `11.6/12.7%` at 1B and `13.9/15.1%` at 3B for NPO/SimNPO. The 1B bound costs at most six NPO or seven SimNPO steps; the prespecified uniform ten-step charge still exceeds both this wall-clock bound and the stricter 8.65-step dynamic-padding FLOP estimate. The compute CSV, paper, full rebuttal, and portal response now use this same end-to-end definition.

### 2026-07-11 22:18 UTC: release-candidate verification

The release candidate was verified from a clean temporary build and from both raw summaries and the portable metric snapshot:

- A four-pass ACL LaTeX build with BibTeX produces a 16-page PDF. The final log has no unresolved citation/reference, rerun, overfull-box, or fatal-error warning. The main text still ends on page 8. Visual checks of the anonymous first page and the revised compute-overhead page show no clipping or overlap; all PDF fonts are embedded.
- `PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py' -v` and the equivalent focused `pytest` invocation pass all nine CPU tests. These cover both theorem limits, BF16 metric serialization, path construction, snapshot reads, token normalization, compute-component arithmetic, CSV/calculator agreement, end-to-end wall percentages, and the conservative dynamic-padding bound.
- `py_compile` passes for all modified/new Python scripts, `bash -n` passes for the all-token runner, and `git diff --check` reports no whitespace errors.
- Direct aggregation over local experiment summaries and snapshot-only aggregation are byte-for-byte identical and both parse 422 expected reads. Regenerating the snapshot is byte-for-byte identical to the committed candidate; its SHA-256 remains `f15fde9f849e1dfa6283fda338bcae5f28186a98f06444ca37596cbf31b2730d`.
- All 18 all-token compute-matched TSV manifests contain exactly one `started` and one terminal `ok` row, with no other status. All 18 corresponding `TOFU_SUMMARY.json` files parse and contain only finite numeric metric fields.
- The portal copy has balanced display-math delimiters and no URL. The snapshot has no absolute workspace path, bearer token, API key, password, or secret pattern. Current-facing files contain none of the superseded Llama `50/55`, `100/105`, or `250/255` comparisons, `<=66/<=121 s` bounds, or 9--11% setup claims (Gemma's independently derived S100/S105 comparison remains valid).

Two release operations remain external to the verified checkout. First, the final commit must be pushed with an ordinary fast-forward-safe push; the current environment cannot reach the GitHub remote through its proxy. Second, the anonymous 4open URL currently returns `401 not_connected`; the repository owner must reconnect or recreate it after the push and test it in a logged-out browser. The public GitHub remote must not be substituted in the anonymous manuscript.

### 2026-07-11 23:06 UTC: critical same-stack wall-clock correction

The 22:18 release candidate above was reopened rather than shipped. Two late
checks invalidated its *wall-clock interpretation* (not its saved metrics):

1. The old 50-step `train_runtime` values used as denominators included costly
   start/end evaluation, whereas K-FORGE setup was being compared with pure
   training work. This understated the setup charge.
2. The host environment had drifted to Transformers 5.12.0 and Accelerate
   1.14.0, while the repository protocol uses Transformers 4.51.3 and
   Accelerate 0.34.2. In particular, logged gradient-accumulated losses have
   different scaling semantics. Model outcomes were close, but a timing claim
   cannot mix these stacks.

The uniform `+10` results are therefore retained only as diagnostic history and
removed from every current-facing claim and from the portable snapshot. I
created a system-site environment with Transformers 4.51.3, Accelerate 0.34.2,
and the host's PyTorch 2.9.1/CUDA 13 stack; the fixed runner now rejects other
Transformers/Accelerate versions and prints PyTorch explicitly. Both arms were
rerun in FP32 with training evaluation disabled.

The complete one-layer setup remains `80.157 s`. Evaluation-free S50 training
averages `197.009 s` for NPO and `130.319 s` for SimNPO, so setup is
`40.687%/61.509%` of S50 training, not `11.6%/12.7%`. Before final metrics were
inspected, direct runtime matching selected scratch S73 for NPO and scratch S86
for SimNPO. The FLOP charge is smaller (`7.00/8.19` nominal steps; maximum
dynamic-padding sensitivity `8.65`), so wall time determines both budgets.

| Method | Seed | KF S50 train | KF + setup | Scratch train | Wall margin |
|---|---:|---:|---:|---:|---:|
| NPO | 0 | 202.439 s | 282.596 s | 293.648 s | +11.053 s |
| NPO | 1 | 187.313 s | 267.470 s | 271.034 s | +3.564 s |
| NPO | 2 | 201.276 s | 281.433 s | 291.964 s | +10.532 s |
| SimNPO | 0 | 133.612 s | 213.769 s | 228.788 s | +15.019 s |
| SimNPO | 1 | 125.155 s | 205.312 s | 209.285 s | +3.973 s |
| SimNPO | 2 | 132.188 s | 212.345 s | 226.788 s | +14.443 s |

Final direct wall/FLOP-matched outcomes (`n=3`) are:

| Method | KF/scratch steps | Scratch/KF FP | Relative FP reduction | Scratch/KF utility | Utility delta | FP paired p |
|---|---:|---:|---:|---:|---:|---:|
| NPO | 50/73 | 0.059022/0.045852 | 22.31% | 0.548584/0.576600 | +0.028016 | 0.00696 |
| SimNPO | 50/86 | 0.552696/0.537060 | 2.83% | 0.583151/0.572220 | -0.010931 | 0.126 |

NPO preserves the favorable FP and utility directions in every seed. Its FP
test is descriptive under the `p<.001` rule, while utility meets that threshold
(`p=4.91e-4`). Extraction worsens `0.067004 -> 0.071685` and ROUGE worsens
`0.262573 -> 0.271766`; both remain explicit. SimNPO preserves favorable FP and
ROUGE directions in every seed and improves mean extraction
`0.196417 -> 0.187717`, but its mean utility change misses the prespecified
`-0.01` margin by `0.000931`. The strict claim is consequently NPO-specific;
SimNPO is reported as a weaker trade-off rather than comparable-utility
confirmation.

All three final queues exited with code 0. Direct aggregation now parses `362`
expected reads. The regenerated snapshot contains `325` structured summaries
and has SHA-256
`d4c60defc8f80ab051076976905b18f2bb10472664c309a7939ae7ac4baa3f9f`.
The obsolete `+5` comparisons were also removed from the current aggregator and
snapshot rather than relabeled: they no longer satisfy the corrected compute
definition and would create a second, incompatible result table.

### 2026-07-11 23:18 UTC: final three-pass hostile audit

I reread the revised responses three times from the position of a reviewer
looking for reasons not to raise the score.

| Pass | Hostile objection | Response change | R1/R2/R3 score |
|---|---|---|---:|
| 1: correctness | The old wall denominator was invalid, SimNPO misses the utility margin, and NPO FP has only three seeds with $p=.00696$. | Removed all old `+5/+10` publication claims; made the strict conclusion NPO-specific; reported SimNPO's exact 0.00093 margin miss and all adverse metrics. | 8.5/9.2/9.1 |
| 2: significance | A second-order method can still look better simply because setup is hidden or amortized. | Added the direct per-seed wall inequality, full non-amortized 80.157 s charge, analytical FLOP inequality, minimum timing margins, and evaluation-free same-stack protocol. | 9.1/9.3/9.2 |
| 3: scope/reproducibility | One positive Llama result may be family-specific, recovery evidence was promised rather than completed, and a reader may not reconstruct tables. | Kept four-seed Gemma evidence, Qwen null and mixed MUSE outcomes, completed relearning/quantization audits, explicit estimator correction, checked calculator, fixed runner, proof tests, and a self-contained metric snapshot. | 9.3/9.5/9.4 |

Residual risks are now substantive rather than presentational: strict Llama
compute confirmation has only three seeds; SimNPO does not meet the utility
margin; MUSE is mixed; relearning is unfavorable; and the anonymous 4open
endpoint still requires owner reconnection. None is hidden or described as a
positive result.

### 2026-07-11 23:24 UTC: final release verification

- All three exact-compute queues exited with code 0; no rebuttal experiment or
  watchdog process remains.
- Direct and snapshot-only aggregation both parse `362` expected reads and are
  byte-for-byte identical. The snapshot contains `325` source summaries and
  has SHA-256
  `d4c60defc8f80ab051076976905b18f2bb10472664c309a7939ae7ac4baa3f9f`.
- All nine CPU tests pass with the documented command
  `cd open-unlearning && PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'`.
  A bare `pytest` invocation does not add `src` and the repository root to
  `sys.path`; the equivalent focused command is
  `PYTHONPATH=.:src pytest -q tests/test_kforge_wiener.py tests/test_rebuttal_compat.py`.
  The tests cover the two theorem limits, compute
  arithmetic/CSV consistency, BF16 metric serialization, snapshot reads, path
  construction, and token normalization.
- `py_compile` passes for every modified/new Python file; `bash -n` passes for
  the exact-compute runner; `git diff --check` reports no whitespace errors.
- The official ACL style and bibliography-style files are now bundled in the
  repository. A clean standalone ACL build in a temporary directory, without
  an external `TEXINPUTS`, produces a 16-page PDF with resolved citations and
  references, no overfull boxes, and all fonts embedded. Visual inspection of
  pages 1, 8, and 12 confirms that the abstract, compute section, equation, and
  exact comparison table are legible and unclipped.
- `aclpubcheck==0.2.0` reports `All Clear` for its page-size, page-limit, and
  font checks. Its raster margin pass cannot run under the host's standard
  ImageMagick PDF security policy; I did not weaken that global policy. Margins
  were instead inspected from rendered pages 1, 2, 6, 8, and 12. The complete
  Conclusion ends on page 8; the required Limitations section follows, which
  ARR explicitly excludes from the content-page limit.
- Portal responses have balanced display-math delimiters and no external URLs.
  The snapshot and release-facing text contain no credentials or absolute host
  paths.
- The three standalone portal responses contain `994/746/566` words (`2306`
  total). A final selection audit now states explicitly that the strict
  compute-matched runs reused the previously reported `alpha=.60` checkpoint;
  only the longer scratch budget was chosen from evaluation-free runtime, and
  it was fixed before final metrics were inspected. The main results section
  also distinguishes downstream-step matching from the later full-setup
  comparison so that the small strict SimNPO utility trade-off is not obscured.
- A final primary-source novelty audit caught an over-broad taxonomy in the
  submitted prose: FILA is already Fisher-weighted LoRA initialization, VILA
  refines that line, and the newly published ReGLU/RILA is a
  representation-guided LoRA initializer. The revision now cites and credits
  these direct precedents and explicitly does *not* claim informed or
  Fisher-guided initialization itself as novel. The narrower distinction is
  the two-Fisher Kronecker Wiener checkpoint edit, exact full-rank solution,
  stated low-rank relaxation, and use with an otherwise unchanged NPO/SimNPO
  optimizer. K-FADE is also correctly described as few-step Gauss--Newton,
  rather than as a one-shot method. After compressing this context and the
  compute prose, the complete Conclusion remains on main-text page 8; the PDF
  remains 16 pages with Limitations/references/appendix following the main
  limit.
- The audited evidence release is committed as `b99d434`; the final
  text/novelty/format refinements and this audit are recorded in the current
  `dev` HEAD. A proxy-free fetch confirmed that `dev` is zero commits behind
  and, after the final SVD-tie proof clarification, eight commits ahead of
  `origin/dev`, so the update is a fast-forward.
  Pushing through the configured proxy failed with `Proxy CONNECT aborted`;
  direct HTTPS push reached GitHub but this non-interactive environment has no
  credential helper, and SSH has no authorized key. No force push or alternate
  identity-bearing remote was attempted. The remaining Git action is an
  ordinary authenticated `git push origin dev` from the owner's shell.
- A fresh logged-out check of the anonymous 4open endpoint still returns
  `401 not_connected`. Reconnecting it and verifying the logged-out view
  remains the other manual publication action after pushing from an
  authenticated shell.

### 2026-07-12 00:18 UTC: final theorem-domain audit

A final theorem-to-proof consistency pass found one formal omission in the
statement, not in the implementation or experiments. Strict convexity of the
quadratic objective requires the retain-penalty domain
`lambda_ret >= 0`; the proof used that domain implicitly, but the theorem had
not stated it. The theorem and algorithm discussion now make the condition
explicit. The zero-penalty rank-r proposition and limit-consistency proposition
also say *an* exact minimizer rather than implying uniqueness when the
truncation singular values are tied. Finally, the forget-only description now
states precisely that identity retain factors remove retain-set curvature
awareness while leaving the isotropic penalty. No equation, checkpoint, metric,
or experimental conclusion changed.

### 2026-07-12 00:24 UTC: full-tree anonymity scan

The final source-package audit scanned every tracked text file rather than only
the metric snapshot and rebuttal. It found one historical changelog sentence
and one dormant queue script containing an author-specific absolute home path.
The changelog now uses neutral wording, and the script takes a configurable
`MODEL_PATH` with a public model identifier as its default. Shell syntax
validation passes, and a full-tree scan finds no remaining author-specific home
path or repository-account string. Upstream project links and ordinary
bibliographic author names remain intact because they are citations, not
repository identity leaks.

### 2026-07-12 00:30 UTC: final claim-language audit

An independent per-seed recalculation reproduced the strict-compute means,
relative reductions, utility-margin slack, and paired t statistics from the
portable snapshot. The only remaining language issue was in the expanded
response, not the portal copy or manuscript: one held-out seed was described
as independently confirming the result, and weaker controls were described as
not reproducing it. These now say that the held-out seed *matches the direction*
and that the controls are *weaker in the Gemma NPO comparison*. The numerical
claims and portal word counts are unchanged.

### 2026-07-12 00:33 UTC: SVD-tie proof clarification

Visual inspection of the final appendix prompted one last degeneracy check.
With tied singular values, an arbitrary implementation of
`SVD_r(-R)` need not return the literal negative of an independently
computed `SVD_r(R)`, although every such truncation attains the same
Eckart--Young optimum. The rank-r proof now says that the compatible negative
truncation *may be chosen* and that any truncation of `-R` is optimal.
The limit-consistency proof uses optimality rather than an unnecessary literal
matrix identity. This changes no proposition, algorithm, checkpoint, or
experiment.
