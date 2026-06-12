# K-FORGE Implementation Changelog

## 2026-06-12

### Finalized Frontier And Relearning Audit Artifacts

- Finalized compact aggregate summaries for the extended layer-15 strength frontier
  (`alpha=0.55,0.56,0.58,0.60,0.62,0.65,0.66,0.68`, seeds `0,1,2`) and the
  layer-15 budget-relearning audit (`S25/S100/S250`, seeds `0,1,2`).
- Stored the durable aggregate JSONs under
  `open-unlearning/logs/review_extra_strength/` and
  `open-unlearning/logs/review_budget_relearn/`.
- Paper-safe takeaway: the extra-strength frontier remains a broad plateau,
  while budget relearning still recovers answer probability but shows lower
  extraction recovery at `S100`/`S250` than at `S25`.

## 2026-06-08

### Review Experiment Results Integrated

- Added manuscript results for the layer-15 `alpha=0.62` budget-sensitivity
  audit on TOFU `forget10`, comparing scratch vs. K-FORGE-initialized NPO at
  budgets `S25`, `S100`, and `S250` over seeds `0,1,2`.
- Added manuscript results for alternate-split generalization on TOFU
  `forget05` and `forget01`, comparing scratch vs. K-FORGE-initialized NPO at
  `S50` over seeds `0,1,2`.
- Recorded aggregate JSON artifacts for these review audits under
  `open-unlearning/logs/review_budget_sweep/` and
  `open-unlearning/logs/review_split_generalization/`.
- Added the completed target-module audit for layer-15 `alpha=0.62`, comparing
  `mlp.down_proj`, `mlp.up_proj`, `mlp.gate_proj`, and `self_attn.o_proj` over
  three seeds. The down-projection target remains the safest setting, with the
  lowest privacy leakage/extraction and the best aggregate Forget Quality.
- Added the completed budget-dependent direct-relearning audit for layer-15
  `alpha=0.62` at `S25`, `S100`, and `S250`. The `S100`/`S250` checkpoints show
  lower extraction recovery than the undertrained `S25` checkpoint, but direct
  forget-set fine-tuning still recovers answer probability.
- Expanded the layer-15 strength frontier with additional `alpha` points
  `0.56`, `0.58`, `0.66`, and `0.68` over seeds `0,1,2`. The extended frontier
  is a broad plateau: utility and extraction are nearly unchanged across the
  range, privacy leakage is lowest near `0.65`, and mean Forget Quality peaks at
  `0.68` within large seed variance.
- Paper-safe interpretation: the conservative layer-15 `alpha=0.62` point is a
  privacy-side audit setting rather than the headline NPO initializer in the
  `forget10` budget sweep, while the alternate-split runs give cleaner
  generalization evidence with improved Forget Quality and lower extraction on
  both smaller forget splits.

## 2026-05-24

### Manuscript Updated With Latest 3B Results

- Updated `main.tex` to incorporate the completed Llama-3.2-3B-Instruct
  minimal-v2 sanity check on TOFU `forget10`.
- Added a paired 3B table for NPO and SimNPO scratch vs. K-FORGE
  initialization over completed seeds `1` and `2`.
- Revised the abstract, introduction, setup, limitations, and conclusion so
  the manuscript no longer describes all 3B experiments as future work.
- Explicitly scoped the 3B result as a scale sanity check rather than a full
  scaling study: SimNPO preserves the matched-budget K-FORGE improvement,
  while NPO shows a strong 50-step gain but mixed longer-budget behavior.
- Clarified that MUSE and robustness audits remain incomplete.

## 2026-05-15

### Numerical And Theory Correction Pass

- Audited the K-FORGE path against the updated derivation and identified that
  the earlier Week-1/Week-2 measurements were based on the historical
  `legacy_v1` path with bf16 checkpoint edits, unshifted token masking,
  mean-reduced calibration gradients, and an overly large damping scale floor.
- Added side-by-side edit variants in `src/trainer/unlearn/kforge.py`:
  - `legacy_v1` preserves the original asymmetric heuristic for ablation.
  - `wiener_v2` implements the theorem-aligned retain-basis relaxation:
    simultaneous diagonalization of the forget/retain Fisher pairs, Wiener
    filtering in the generalized-eigen basis, rank truncation in rescaled
    `Y`-space, and retain-factor unwhitening.
- Corrected calibration and numerical behavior:
  - Fisher calibration now runs in eval mode.
  - Causal-LM calibration masks activation/gradient rows using the shifted
    valid-label positions, excluding pad and non-loss tokens.
  - Calibration backpropagates summed token loss so gradient covariance is not
    biased by per-batch valid-token counts.
  - Damping now scales with `clamp_min(damping_floor)` instead of forcing
    factor diagonals to be at least `1.0`.
  - Per-module raw Fisher diagonal means and valid-token counts are logged for
    future auditability.
  - Edited checkpoints now default to fp32 weights so small edits remain
    representable after save.
- Added new method knobs:
  `damping_floor`, `edit_variant`, `lambda_tradeoff`, and
  `edit_weight_dtype`.
- Updated the K-FORGE docs and community wrapper notes to distinguish
  historical `legacy_v1` from theorem-backed `wiener_v2`.
- Updated the TOFU sweep and Week-2 init scripts so training dtype is explicit
  through `MODEL_TORCH_DTYPE`, defaulting to `float32` for corrected reruns.
- Added `scripts/kforge_corrected_validation_sweep.sh`, which queues the
  repaired `legacy_v1` B-calibration rerun plus a compact `wiener_v2` lambda
  pilot from one manifest.
- Added `scripts/kforge_corrected_ablation_sweep.sh` for the repaired A1
  diagonal-Fisher and A2 forget-only reruns at the corrected calibration scale.
- Added `scripts/kforge_corrected_adaptive_damping_sweep.sh` so the damping
  interaction can be rerun after the scale-floor fix without overwriting the
  historical adaptive-damping artifacts.
- Updated `scripts/kforge_week1_spectrum.sh` to use explicit fp32 loading and
  the local Hugging Face settings for the corrected spectrum rerun.
- Marked the existing Week-2 findings documents as provisional pending reruns
  with the corrected code path.

### Verification And Queue Handoff

- Re-ran static checks:
  `python -m py_compile src/trainer/unlearn/kforge.py src/trainer/__init__.py`
  and `bash -n` over the edited sweep scripts.
- Ran synthetic algebra checks for `wiener_v2`:
  - Full-rank `lambda=0.7` matched the exact vectorized quadratic solve with
    max absolute error `7.355e-16`.
  - Rank-2 `lambda=0` matched the exact rescaled `Y`-space solution with max
    absolute error `0.000e+00`.
  - The causal-LM mask test selected the shifted valid-label positions only.
  - The edit path upcast a bf16 toy checkpoint to fp32 as intended.
- Stopped the stale 3B smoke queue on GPU 1 because it was still running the
  pre-correction measurement path and was lower value than corrected reruns.
- Started tmux session `kforge_corrected_gpu1_20260515T1249Z` on GPU 1.
  - Manifest:
    `open-unlearning/logs/kforge_corrected_validation_20260515T1249Z.tsv`
  - Phase 1: corrected `legacy_v1` `B_cal in {2,4,8,16,32,64}` strength sweep.
  - Phase 2: `wiener_v2` pilot at `B_cal=32` over
    `lambda in {0,1e-4,1e-3,1e-2}` and five small strengths.
- Added idle-gated follow-on queues so newly free GPUs are claimed only after
  they stay genuinely idle for five checks:
  - GPU 2 -> corrected A1/A2 ablation rerun after the legacy RMU tail exits.
  - GPU 0 -> corrected adaptive-damping rerun if the currently hidden external
    load clears.
  - GPU 3 -> corrected 112-module spectrum dump if its external resident load
    clears.
- Stopped the remaining legacy `forget01` RMU tail on GPU 2 after confirming it
  still had another stale seed pair queued under the pre-correction K-FORGE
  path; this frees GPU 2 for corrected ablations instead of spending several
  more hours on non-headline measurements.
- Measured corrected edit scales at `B_cal=32`:
  - `legacy_v1`, strength `0.002`: relative layer Frobenius delta `6.075x`.
  - `wiener_v2`, strength `0.004`, `lambda=0`: relative layer Frobenius delta
    `2.10e-4`.
  - diagonal control, strength `0.004`: relative layer Frobenius delta `0.309x`.
- Added `scripts/kforge_corrected_retune_sweep.sh` with norm-aware follow-up
  grids: micro strengths for `legacy_v1` and macro strengths for `wiener_v2`.
- Completed the first norm-aware retune:
  - `legacy_v1` micro strengths up to `3.2e-4` remained weak at the
    high-utility end (`0.8503` forget probability at utility `0.5503`).
  - `wiener_v2` recovered a real operating band:
    strength `0.3` retained utility `0.583-0.595` with forget probability
    `0.827-0.837`, strength `1.0` reached forget probability `0.133-0.142`
    with lower utility `0.229-0.370`, and strengths `>=3` collapsed.
- Started a finer `wiener_v2` sweep on GPU 2 over strengths
  `{0.35,0.45,0.6,0.8}` and `lambda in {0,1e-3,1e-2}` to resolve the corrected
  Pareto frontier between the no-op and over-edit regimes.
- Updated `scripts/kforge_week2_init_experiment.sh` with `KFORGE_INIT_TAG` so
  multiple corrected initializers can be evaluated under one fair scratch
  baseline without task-name collisions.
- Added `scripts/kforge_corrected_v2_transfer_sweep.sh` for corrected
  `wiener_v2` transfer sweeps on `forget05` / `forget01` when extra GPUs become
  available.

### 24-Hour Wiener-v2 Schedule

- Started the corrected headline experiment on `2026-05-17`:
  - GPU 1: `NPO`, `forget10`, fp32 scratch vs. `wiener_v2`
    initializer `lambda=0.01, strength=0.45`, steps `{50,100,250}`, seeds
    `{0,1,2}`, followed by an initialized-only `strength=0.6` block.
  - GPU 2: same schedule for `SimNPO`.
- Replaced the old idle-gated adaptive-damping / spectrum sidecars with
  corrected transfer work more relevant to the new paper hypothesis:
  - GPU 0 if idle -> `forget05` `wiener_v2` one-shot transfer sweep.
  - GPU 3 if idle -> `forget01` `wiener_v2` one-shot transfer sweep.
- On 2026-05-18, after `forget01` transfer completed and GPUs 2/3 became free,
  launched the corrected `forget01` headline init experiment:
  - GPU 2: `NPO`, scratch vs. `wiener_v2 s=0.45`, followed by initialized-only
    `s=0.6`.
  - GPU 3: same schedule for `SimNPO`.
- Moved the waiting `forget05` transfer off persistently busy GPU 0 onto GPU 1,
  then queued corrected `forget05` NPO init work behind it so GPU 1 does not go
  idle after the transfer checkpoints are produced.

## 2026-05-08

### Planning Intake

- Read `PLAN.md` and selected the recommended implementation target:
  Kronecker-Fisher Generalized SVD for one-shot LLM unlearning (`K-FORGE`).
- Inspected the OpenUnlearning trainer stack, Hydra configuration layout,
  unlearning dataset wrapper, collator behavior, and existing method registry.
- Chose a trainer-based integration because `src/train.py` already delegates
  training, saving, and evaluation through registered trainer classes.

### Code Added

- Added `src/trainer/unlearn/kforge.py`.
  - Registers a one-shot `KFORGE` trainer derived from `UnlearnTrainer`.
  - Finds target `torch.nn.Linear` modules by full-match regex.
  - Adds module-count and module-size guards to bound calibration memory.
  - Collects empirical Kronecker factors on forget and retain batches:
    `A = E[x x^T]` from module inputs and `B = E[g g^T]` from output gradients.
  - Applies damping before Cholesky factorization for numerical stability.
  - Forms the retain-whitened forget-weight matrix, runs thin SVD, and applies
    a negative rank-r edit scaled by `strength`.
  - Supports planned ablations through `factor_mode: diagonal` and
    `use_retain_fisher: false`.
  - Logs edited/skipped module counts, rank, strength, and summed retained
    singular value mass.
- Updated `src/trainer/__init__.py` to import and register `KFORGE`.

### Configuration Added

- Added `configs/trainer/KFORGE.yaml`.
  - Defaults to one-shot execution with `max_steps: 1`.
  - Targets `.*mlp\.down_proj$` modules by default.
  - Exposes `rank`, `strength`, `damping`, calibration batch count, factor
    normalization, and large-module skipping knobs.
  - Sets `max_target_modules: 4` as a conservative default memory guard; users
    can set it to `null` for all matched modules.
  - Exposes diagonal-vs-Kronecker and retain-Fisher ablation toggles.
- Added `configs/experiment/unlearn/tofu/kforge.yaml`.
  - Builds on the existing TOFU unlearning experiment.
  - Overrides the trainer to `KFORGE`.
  - Uses conservative defaults: rank 8, strength 0.25, damping 1e-3, and 8
    calibration batches.

### Documentation Added

- Added `docs/kforge.md` with method summary, run command, configuration knobs,
  and current scope.
- Updated `docs/components.md` with a short K-FORGE component entry.
- Added `community/methods/KFORGE/README.md` and executable
  `community/methods/KFORGE/run.sh` so K-FORGE appears alongside other community
  method wrappers.
- Added `scripts/kforge_tofu_sweep.sh` to run the PLAN-aligned TOFU strength
  sweep with local cache overrides, accessible OpenUnlearning model/tokenizer
  paths, fp32 evaluation, and toggles for `factor_mode` and
  `use_retain_fisher`.
- Added `scripts/kforge_tofu_overnight.sh` to run a detached overnight queue
  over TOFU `forget10`, `forget05`, and `forget01`; Kronecker vs diagonal
  factors; retain-Fisher vs forget-only; ranks 2/4/8; 1- and 2-module edits;
  and 2- vs 8-batch calibration.

### Verification

- Ran `python -m py_compile src/trainer/unlearn/kforge.py src/trainer/__init__.py`;
  both files compiled successfully.
- Performed lightweight configuration text checks for the new K-FORGE trainer
  and TOFU experiment YAML files.
- Installed missing runtime dependencies needed for experiments:
  Hydra/OmegaConf, Transformers, Datasets, Accelerate, bitsandbytes,
  DeepSpeed, TensorBoard, W&B, and `lm-eval`.
- Set local cache paths for Triton, Torch extensions, Hugging Face, and XDG
  cache data to avoid the non-writable `/home/d.moskovskiy/.triton` path.
- Verified that the trainer registry imports and includes `KFORGE`.
- Fixed `configs/experiment/unlearn/tofu/kforge.yaml` so Hydra resolves the
  sibling TOFU default via `/experiment/unlearn/tofu/default`.

### Experiment Setup

- Verified CUDA availability: 5 visible GPUs.
- Downloaded OpenUnlearning TOFU/MUSE evaluation logs with
  `python setup_data.py --eval_logs`.
- Downloaded and cached `open-unlearning/tofu_Llama-3.2-1B-Instruct_full`.
- Avoided the gated Meta tokenizer path by overriding
  `model.tokenizer_args.pretrained_model_name_or_path` to the OpenUnlearning
  checkpoint.
- Avoided missing FlashAttention by overriding
  `model.model_args.attn_implementation=sdpa`.
- Avoided the OpenUnlearning bfloat16-to-NumPy metric issue during evaluation by
  loading evaluated checkpoints with `model.model_args.torch_dtype=float32`.

### Experiments Run

- Ran `KFORGE_TOFU_F10_R4_M1_B2`.
  - Benchmark: TOFU `forget10` / `retain90`, Llama-3.2-1B-Instruct target.
  - K-FORGE settings: Kronecker mode, retain Fisher enabled, rank 4, strength
    0.25, 1 target `mlp.down_proj` module, 2 calibration batches per split.
  - Training result: 1 edited module, 0 skipped modules, top singular mass
    19.1721, runtime 17.82s.
  - Evaluation output:
    `saves/eval/KFORGE_TOFU_F10_R4_M1_B2_EVAL_FP32/TOFU_SUMMARY.json`.
  - Summary: `forget_Q_A_Prob=3.292e-06`,
    `forget_Q_A_ROUGE=0.0965`, `extraction_strength=0.0325`,
    `forget_quality=3.151e-15`, `model_utility=3.407e-05`,
    `privleak=-20.5457`.
- Ran `KFORGE_TOFU_F10_R4_M1_B2_S001`.
  - Same setup as above, but strength 0.01.
  - Training result: 1 edited module, 0 skipped modules, top singular mass
    19.1721, runtime 16.73s.
  - Evaluation output:
    `saves/eval/KFORGE_TOFU_F10_R4_M1_B2_S001_EVAL_FP32/TOFU_SUMMARY.json`.
  - Summary: `forget_Q_A_Prob=0.00185`,
    `forget_Q_A_ROUGE=0.1191`, `extraction_strength=0.0326`,
    `forget_quality=0.00383`, `model_utility=0.0247`,
    `privleak=4.9645`.
- Ran `KFORGE_TOFU_F10_R4_M1_B2_S0p001_kron` through
  `scripts/kforge_tofu_sweep.sh`.
  - Same setup as above, but strength 0.001.
  - Training result: 1 edited module, 0 skipped modules, top singular mass
    19.1721, runtime 16.54s.
  - Evaluation output:
    `saves/eval/KFORGE_TOFU_F10_R4_M1_B2_S0p001_kron_EVAL_FP32/TOFU_SUMMARY.json`.
  - Summary: `forget_Q_A_Prob=0.8557`,
    `forget_Q_A_ROUGE=0.7661`, `extraction_strength=0.6302`,
    `forget_quality=1.399e-20`, `model_utility=0.5991`,
    `privleak=-99.3470`.
- Ran A1 diagonal-Fisher ablation
  `KFORGE_TOFU_F10_R4_M1_B2_S0p001_diagonal`.
  - Same setup as the strength 0.001 run, but `factor_mode=diagonal`.
  - Training result: 1 edited module, 0 skipped modules, top singular mass
    16.5684, runtime 16.65s.
  - Evaluation output:
    `saves/eval/KFORGE_TOFU_F10_R4_M1_B2_S0p001_diagonal_EVAL_FP32/TOFU_SUMMARY.json`.
  - Summary: `forget_Q_A_Prob=0.8682`,
    `forget_Q_A_ROUGE=0.7946`, `extraction_strength=0.6766`,
    `forget_quality=3.905e-22`, `model_utility=0.6019`,
    `privleak=-99.4351`.
- Compared against cached target-model TOFU `forget10` summary:
  `forget_Q_A_Prob=0.8805`, `forget_Q_A_ROUGE=0.8201`,
  `extraction_strength=0.7063`, `model_utility=0.5992`,
  `privleak=-99.4574`.

### Experiment Findings

- The current one-module K-FORGE edit strongly suppresses forget-set probability,
  ROUGE, and extraction strength.
- Retention/utility collapses at strength 0.25 and remains poor at strength
  0.01, so this implementation needs a smaller strength sweep before increasing
  rank, calibration size, or layer count.
- Strength 0.001 preserves model utility almost exactly while producing a small
  forget-side reduction (`forget_Q_A_Prob` 0.8805 -> 0.8557,
  `forget_Q_A_ROUGE` 0.8201 -> 0.7661, `extraction_strength` 0.7063 -> 0.6302).
- At matched strength 0.001, the full Kronecker version suppresses forget
  metrics more than the diagonal-Fisher ablation while preserving similar model
  utility. This is preliminary positive evidence for PLAN ablation A1.
- The next PLAN-aligned run should test the midpoint `STRENGTHS="0.003"` and
  then run `USE_RETAIN_FISHER=false` at the best strength for ablation A2.

### Overnight Sweep Queue

- Prepared an overnight sweep with 16 blocks and 41 train/eval runs:
  - `forget10/retain90`, rank 4, 1 module, strengths
    `0.0015 0.002 0.003 0.005` for full Kronecker.
  - Matched diagonal-Fisher A1 block.
  - Matched forget-only retain-Fisher ablation A2 block at strengths
    `0.001 0.002 0.003`.
  - 2-module probes at strengths `0.0005 0.001` for Kronecker and diagonal.
  - `forget05/retain95` Kronecker and diagonal blocks at strengths
    `0.001 0.002 0.003`.
  - Rank 2 and rank 8 probes on `forget10/retain90`.
  - 8-calibration-batch probes on `forget10/retain90`.
  - Tiny forget-set `forget01/retain99` Kronecker and diagonal probes.
- The queue writes logs to `logs/`, run artifacts to `saves/unlearn/`, eval
  summaries to `saves/eval/`, and a TSV manifest named
  `logs/kforge_overnight_<timestamp>.tsv`.
- Started the overnight queue in detached tmux session `kforge_overnight` at
  `2026-05-08T20:44:10Z`.
  - Main log:
    `open-unlearning/logs/kforge_overnight_tmux_20260508T204410Z.log`.
  - Manifest:
    `open-unlearning/logs/kforge_overnight_20260508T204410Z.tsv`.
  - Verified the first run is active:
    `KFORGE_TOFU_forget10_R4_M1_B2_S0p0015_kron_retain`.
  - `nohup` was attempted first but exited immediately in this shell; tmux is
    the active runner.

### Overnight Sweep Results

- The tmux overnight sweep completed cleanly at `2026-05-09T00:42:28Z`.
- All 15 manifest blocks completed with status `ok`.
- The run produced 41 train/eval pairs from the overnight queue and 41 matching
  summary rows in `open-unlearning/saves/eval/kforge_overnight_summary.csv`.
- Best utility-preserving forget reduction among runs with
  `model_utility >= 0.55`:
  - `KFORGE_TOFU_forget10_R2_M1_B2_S0p003_kron_retain`
  - `model_utility=0.5535`, `forget_Q_A_Prob=0.5442`,
    `forget_Q_A_ROUGE=0.4755`, `extraction_strength=0.2269`.
- Next-best utility-preserving Kronecker points:
  - `KFORGE_TOFU_forget10_R8_M1_B2_S0p002_kron_retain`:
    `model_utility=0.5568`, `forget_Q_A_Prob=0.6271`,
    `forget_Q_A_ROUGE=0.5294`, `extraction_strength=0.2759`.
  - `KFORGE_TOFU_forget10_R4_M1_B2_S0p002_kron_retain`:
    `model_utility=0.5771`, `forget_Q_A_Prob=0.6896`,
    `forget_Q_A_ROUGE=0.5670`, `extraction_strength=0.3223`.
- Strongest overall forget-quality run was
  `KFORGE_TOFU_forget10_R4_M1_B2_S0p005_kron_retain`, but utility dropped:
  `forget_quality=0.6405`, `model_utility=0.2406`,
  `forget_Q_A_Prob=0.0379`, `extraction_strength=0.0387`.
- The overall pattern is still strength-sensitive:
  Kronecker runs dominate the best high-utility forget reduction points, while
  stronger edits can erase aggressively but damage utility.

### Stage-2 Sweep Queue

- Analyzed the overnight CSV and found:
  - A2 forget-only runs were effectively no-ops at strengths `0.001-0.003`,
    confirming the retain-Fisher contrast is necessary for the observed edit.
  - Best high-utility frontier point was rank 2, one `down_proj`, strength
    `0.003`.
  - Rank 8 at strength `0.002` and rank 4 at strength `0.002` were the next
    best high-utility Kronecker points.
- Updated `scripts/kforge_tofu_sweep.sh` to accept
  `TARGET_MODULES_REGEX` and `RUN_SUFFIX`, so layer-selection probes do not
  overwrite prior runs.
- Added `scripts/kforge_tofu_stage2.sh`.
  - Refines the rank-2 down-projection frontier at strengths
    `0.00325 0.0035 0.00375 0.004`.
  - Tests rank-2 two-module edits at strengths `0.0015 0.002 0.0025`.
  - Runs A4 layer-selection probes for `gate_proj`, `up_proj`, and
    `self_attn.o_proj` at strengths `0.001 0.002 0.003`.
- Started detached tmux session `kforge_stage2` at `2026-05-09T14:25:02Z`.
  - Main log: `open-unlearning/logs/kforge_stage2_tmux_20260509T142502Z.log`.
  - Manifest: `open-unlearning/logs/kforge_stage2_20260509T142502Z.tsv`.
  - First active run:
    `KFORGE_TOFU_forget10_R2_M1_B2_S0p00325_kron_retain_down`.

### Stage-3 Parallel Queue

- Stage 2 completed cleanly with all 5 manifest blocks marked `ok`.
- Aggregated 57 total K-FORGE TOFU summaries into
  `open-unlearning/saves/eval/kforge_all_summary.csv`.
- Stage-2 result highlights:
  - Best high-utility point remained the overnight rank-2/down-projection run:
    `KFORGE_TOFU_forget10_R2_M1_B2_S0p003_kron_retain` with
    `model_utility=0.5535`, `forget_Q_A_Prob=0.5442`,
    `forget_Q_A_ROUGE=0.4755`, `extraction_strength=0.2269`.
  - Stage-2 frontier points at strengths `0.00325-0.004` improve forgetting
    further but cross below the `model_utility >= 0.55` utility threshold.
  - A4 layer probes show `down_proj` is still the strongest target; `gate_proj`,
    `up_proj`, and `self_attn.o_proj` are weaker at matched rank/strength.
- Added `scripts/kforge_stage3_worker.sh` for two-GPU parallel runs.
- Started Stage 3 in two tmux sessions at `2026-05-09T15:25Z`:
  - `kforge_stage3_gpu0`: frontier refinement and matched diagonal/forget-only
    controls on GPU 0.
  - `kforge_stage3_gpu2`: transfer probes for `forget05`/`forget01` and a
    gate-projection probe on GPU 2.
- GPU selection note:
  - Initial GPU 0 launch OOMed because CUDA ordinal mapping did not match
    `nvidia-smi`; `CUDA_DEVICE_ORDER=PCI_BUS_ID` was added to
    `scripts/kforge_tofu_sweep.sh`, verified GPU 0 maps to the free H200, and
    the GPU 0 worker was relaunched.
  - GPUs 1, 3, and 4 were not used because they showed high memory use and/or
    high utilization.
- Stage 3 completed by `2026-05-09T16:01:57Z`.
- Re-aggregated 75 total K-FORGE TOFU summaries into
  `open-unlearning/saves/eval/kforge_all_summary.csv`.
- Stage-3 manifest outcomes:
  - Retry frontier manifest
    `open-unlearning/logs/kforge_stage3_frontier_20260509T152611Z.tsv`:
    all 3 blocks `ok` (`frontier_tight`, `frontier_diag`,
    `frontier_forgetonly`).
  - Transfer manifest
    `open-unlearning/logs/kforge_stage3_transfer_20260509T152513Z.tsv`:
    `forget01_r2` and `forget05_gate` completed `ok`; `forget05_r2` ended
    `failed:2`, but several usable `forget05` summaries were produced before
    the failure.
  - The earlier stale frontier manifest from the first GPU 0 launch contains
    CUDA/OOM failures and is superseded by the retry manifest above.
- Stage-3 result highlights:
  - Best overall utility-preserving point at `model_utility >= 0.55` is now
    `KFORGE_TOFU_forget10_R2_M1_B2_S0p00305_kron_retain_stage3down` with
    `model_utility=0.5508`, `forget_Q_A_Prob=0.5292`,
    `forget_Q_A_ROUGE=0.4662`, and `extraction_strength=0.2134`.
  - The previous overnight best remains essentially tied:
    `KFORGE_TOFU_forget10_R2_M1_B2_S0p003_kron_retain` with
    `model_utility=0.5535`, `forget_Q_A_Prob=0.5442`,
    `forget_Q_A_ROUGE=0.4755`, and `extraction_strength=0.2269`.
  - Matched diagonal controls at the same frontier strengths preserve utility
    but forget substantially less, e.g. `S0p0032_diagonal` has
    `model_utility=0.5728`, `forget_Q_A_Prob=0.7886`,
    `forget_Q_A_ROUGE=0.6519`, and `extraction_strength=0.4480`.
  - Forget-only control remains effectively inactive:
    `S0p0032_kron_forgetonly` has `model_utility=0.5981`,
    `forget_Q_A_Prob=0.8808`, `forget_Q_A_ROUGE=0.8195`, and
    `extraction_strength=0.7099`.
  - Transfer to `forget05` is directionally successful but weaker at matched
    settings: `S0p003_kron_retain_stage3down` has `model_utility=0.5644`,
    `forget_Q_A_Prob=0.6778`, `forget_Q_A_ROUGE=0.5846`, and
    `extraction_strength=0.3727`.
  - Transfer to `forget01` is milder at matched utility:
    `S0p003_kron_retain_stage3down` has `model_utility=0.5695`,
    `forget_Q_A_Prob=0.6746`, `forget_Q_A_ROUGE=0.5959`, and
    `extraction_strength=0.4355`.
- Added initial report at
  `open-unlearning/docs/kforge_initial_report.md`.
  - The report summarizes the implemented K-FORGE method, TOFU/Llama-3.2-1B
    experiment scope, baseline comparison, best utility-preserving frontier,
    Kronecker/diagonal and retain/forget-only ablations, transfer probes, and
    recommended next steps.

## 2026-05-09: Week-1 diagnostic experiments from updated PLAN.md

- Read the new `RESEARCH.md` and the `PLAN.md` `#NEW` action plan.
- Added `open-unlearning/scripts/kforge_week1_bcal_sweep.sh`.
  - Runs the Day 1-2 calibration-batch diagnostic:
    `B_cal in {2,4,8,16,32,64}` by
    `strength in {0.002,0.0025,0.003,0.0033,0.004}` using the current best
    rank-2 single-`down_proj` retain-whitened Kronecker setup.
  - Splits the queue into two serial tmux workers to avoid oversubscribing
    GPUs.
- Started the B_cal sweep in tmux at `2026-05-09T20:11:41Z`.
  - Manifest:
    `open-unlearning/logs/kforge_week1_bcal_20260509T201141Z.tsv`.
  - GPU 0 worker: batches `2 8 32`.
  - GPU 2 worker: batches `4 16 64`.
- Added `open-unlearning/scripts/tofu_week1_baselines.sh`.
  - Queues same-environment OpenUnlearning baselines for
    `NPO`, `SimNPO`, `RMU`, and `GradDiff` on
    `Llama-3.2-1B-Instruct`, TOFU `forget10`.
  - Uses the same local cache, tokenizer, `sdpa`, and fp32-eval overrides used
    by the K-FORGE runs.
  - The baseline tmux session is waiting for the B_cal sweep to finish before
    consuming GPUs 0 and 2.
- Extended `src/trainer/unlearn/kforge.py` with diagnostic spectrum support.
  - New method args: `spectrum_output_path`, `spectrum_top_k`, and `skip_edit`.
  - `skip_edit=true` now avoids saving an unchanged model.
  - Spectrum output stores top generalized singular values and quantiles for
    each calibrated module.
- Added `open-unlearning/scripts/kforge_week1_spectrum.sh`.
  - Queues per-layer/per-module spectrum dumps for Llama-3.2-1B after the B_cal
    and baseline sessions complete.
  - Default target set covers layers `0-15` and
    `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and
    `down_proj`.
- Added `open-unlearning/scripts/kforge_collect_summary.py` to regenerate
  `open-unlearning/saves/eval/kforge_all_summary.csv` from all K-FORGE TOFU
  summary JSON files, including the new B_cal runs as they complete.

## 2026-05-10: Week-1 status and reruns

- Checked overnight Week-1 jobs.
- Completed artifacts:
  - Spectrum dump completed for all 112 layer/module targets:
    `open-unlearning/saves/spectrum/`.
  - Spectrum manifest:
    `open-unlearning/logs/kforge_week1_spectrum_20260509T201447Z.tsv`.
  - B_cal completed for batches `2` and `4` across all requested strengths.
- B_cal observations so far:
  - `B=2` reproduces the earlier cliff: utility drops from `0.5836` at
    `s=0.002` to `0.5044` at `s=0.004`, while forget probability improves
    from `0.7736` to `0.2779`.
  - `B=4` preserves utility much better but forgets less: at `s=0.004`,
    `model_utility=0.5780`, `forget_Q_A_Prob=0.7737`, and
    `extraction_strength=0.4315`.
- Baseline queue failed on the first NPO run because the environment's
  `bitsandbytes` installation is incompatible with the installed Triton
  (`No module named triton.ops`), causing `paged_adamw_32bit` optimizer
  construction to fail.
- Updated `open-unlearning/scripts/tofu_week1_baselines.sh`:
  - Default optimizer changed to `adamw_torch`.
  - Supports single-GPU `python src/train.py` launch when `GPU_IDS` contains
    one GPU, while preserving Accelerate launch for comma-separated GPU lists.
- Relaunched missing B_cal batches `8`, `16`, `32`, and `64` in tmux session
  `kforge_bcal_missing_20260510T111448Z` on GPU 2.
- Relaunched baselines in queued tmux session
  `tofu_week1_baselines_retry_20260510T111448Z`; it waits for the missing
  B_cal session to finish, then runs `NPO`, `SimNPO`, `RMU`, and `GradDiff`
  with `adamw_torch`.
- User requested proceeding with missing baselines.
  - Stopped `kforge_bcal_missing_20260510T111448Z` so it no longer competes
    for GPU 2.
  - Confirmed `tofu_week1_baselines_retry_20260510T111448Z` is actively running
    the NPO baseline with `adamw_torch` on GPU 2.
  - Missing B_cal batches `8`, `16`, `32`, and `64` are paused until the
    baseline queue finishes or another free GPU becomes available.
- Same-environment baseline queue completed successfully:
  `open-unlearning/logs/tofu_week1_baselines_20260510T111448Z.tsv`.
  - NPO: `model_utility=0.4016`, `forget_Q_A_Prob=0.2189`,
    `forget_Q_A_ROUGE=0.1796`, `extraction_strength=0.0955`.
  - SimNPO: `model_utility=0.5973`, `forget_Q_A_Prob=0.8443`,
    `forget_Q_A_ROUGE=0.7366`, `extraction_strength=0.5622`.
  - RMU: `model_utility=0.5793`, `forget_Q_A_Prob=0.0878`,
    `forget_Q_A_ROUGE=0.3026`, `extraction_strength=0.0555`.
  - GradDiff: `model_utility=0.4446`, `forget_Q_A_Prob=0.0639`,
    `forget_Q_A_ROUGE=0.3626`, `extraction_strength=0.0860`.
  - These baseline results confirm the `RESEARCH.md` warning: current one-shot
    K-FORGE is dominated by RMU on the utility/forget-probability plane in this
    environment.

## 2026-05-10: Week-2 framing and init-experiment setup

- Accepted the revised paper framing:
  `K-FORGE: A Closed-Form Kronecker-Fisher Initialization that Accelerates
  Second-Order LLM Unlearning`.
- Restarted the missing B_cal sweep for batches `8`, `16`, `32`, and `64`.
  - Session: `kforge_week2_bcal_missing_20260510T131331Z`.
  - Manifest:
    `open-unlearning/logs/kforge_week2_bcal_missing_20260510T131331Z.tsv`.
  - GPU: `2`.
- Added explicit adaptive damping support to K-FORGE.
  - New method args:
    `trainer.method_args.damping_mode=fixed|adaptive` and
    `trainer.method_args.adaptive_damping_coeff`.
  - In adaptive mode, damping uses the retain-factor trace scale:
    `coeff * trace(A_r) / dim(A_r)` and the analogous output-gradient factor.
  - Config defaults preserve existing behavior with `damping_mode=fixed`.
- Added `open-unlearning/scripts/kforge_week2_adaptive_damping.sh`.
  - Sweeps `adaptive_damping_coeff in {0.01, 0.1, 1.0}` at a selected
    calibration batch setting.
  - Reuses `scripts/kforge_tofu_sweep.sh`.
- Updated `scripts/kforge_tofu_sweep.sh` with `EXTRA_TRAIN_ARGS` so method
  variants can be swept without duplicating the script.
- Added original-reference support for init experiments.
  - `GradDiff` now accepts `method_args.ref_model_path`.
  - NPO/SimNPO/RMU inherit this path, allowing the trainable model to start from
    a K-FORGE checkpoint while the preference/reference model remains the
    original full model.
- Added `open-unlearning/scripts/kforge_week2_init_experiment.sh`.
  - Runs scratch vs. K-FORGE-init for `NPO` and `SimNPO`.
  - Supports step budgets `50 100 250 500 1000` and seeds `0 1 2`.
  - Evaluates each run with fp32 TOFU eval.
- Verification:
  - `python -m py_compile src/trainer/unlearn/kforge.py
    src/trainer/unlearn/grad_diff.py` passed.
  - Bash syntax checks passed for `kforge_tofu_sweep.sh`,
    `kforge_week2_adaptive_damping.sh`, and
    `kforge_week2_init_experiment.sh`.
- Status check:
  - The restarted B_cal worker completed all five `B=8` strengths.
  - The worker exited before running `B=16`, `B=32`, and `B=64`, without an
    error trace in the corresponding run logs.
  - Relaunched the remaining queue in session
    `kforge_week2_bcal_remaining_20260510T144440Z` on GPU 2.
  - New manifest:
    `open-unlearning/logs/kforge_week2_bcal_remaining_20260510T144440Z.tsv`.
- GPU utilization check:
  - GPU 2 is the only currently usable GPU for project runs; it is running
    `kforge_week2_bcal_remaining_20260510T144440Z`.
  - GPUs 0, 1, 3, and 4 show high utilization and/or high memory use from other
    jobs, so no new project jobs were launched there.
- Queued follow-up GPU-2 jobs behind the active B_cal worker:
  - `kforge_week2_adaptive_queued_20260510T145718Z`: adaptive damping sweep with
    `MAX_CALIBRATION_BATCHES=32`.
  - `kforge_week2_init_smoke_queued_20260510T145718Z`: small NPO init smoke
    experiment with `STEP_BUDGETS=100 250 500`, `SEEDS=0`, scratch vs.
    K-FORGE-init.
- Status update:
  - B_cal `B=16` completed all five strengths.
  - Adaptive damping sweep completed for coefficients `0.01`, `0.1`, and
    `1.0`.
  - Adaptive damping at `B=32` behaved like a near-no-op at the tested
    coefficients: forget probability stayed near the base model
    (`~0.880`) across strengths.
  - The first init-smoke launch failed immediately because Hydra requires
    appending `max_steps` with `+trainer.args.max_steps`.
  - Patched `open-unlearning/scripts/kforge_week2_init_experiment.sh` to use
    `+trainer.args.max_steps`.
  - Relaunched the remaining B_cal batches `32` and `64` in session
    `kforge_week2_bcal_final_20260510T162635Z`.
  - Relaunched the NPO init smoke queue behind it in session
    `kforge_week2_init_smoke_retry_20260510T162635Z`.
- Interim results:
  - B_cal `B=32` completed all five strengths.
    - At `s=0.004`: `model_utility=0.5843`, `forget_Q_A_Prob=0.8285`,
      `extraction_strength=0.5436`.
    - Compared with `B=2`, larger calibration continues to smooth the utility
      cliff but substantially weakens one-shot forgetting at the same strength.
  - B_cal `B=64` is still missing.
  - NPO init smoke produced first paired results:
    - 100 steps, scratch: `model_utility=0.3441`,
      `forget_Q_A_Prob=0.2320`, `extraction_strength=0.0939`.
    - 100 steps, K-FORGE init: `model_utility=0.5490`,
      `forget_Q_A_Prob=0.1200`, `extraction_strength=0.0930`.
    - 250 steps, scratch: `model_utility=0.5710`,
      `forget_Q_A_Prob=0.1022`, `extraction_strength=0.1117`.
    - 250 steps, K-FORGE init: `model_utility=0.5721`,
      `forget_Q_A_Prob=0.0709`, `extraction_strength=0.0936`.
  - Early init-smoke conclusion: K-FORGE init dominates scratch at 100 steps
    and improves forget probability/extraction strength at matched utility by
    250 steps. The 500-step scratch run is currently active.
- Overnight status:
  - NPO init smoke completed through the 500-step pair.
  - 500 steps, scratch: `model_utility=0.5911`,
    `forget_Q_A_Prob=0.0555`, `forget_Q_A_ROUGE=0.3057`,
    `extraction_strength=0.1115`.
  - 500 steps, K-FORGE init: `model_utility=0.5737`,
    `forget_Q_A_Prob=0.0449`, `forget_Q_A_ROUGE=0.3009`,
    `extraction_strength=0.0893`.
  - Summary of NPO init smoke:
    - At 100 steps, K-FORGE init strongly dominates scratch on utility and
      forget probability.
    - At 250 steps, K-FORGE init keeps matched utility while improving forget
      probability and extraction strength.
    - At 500 steps, scratch recovers higher utility, but K-FORGE init still
      gives better forget probability and extraction strength.
  - B_cal is complete through `B=32`; `B=64` remains missing.

## 2026-05-11: Remaining task queue

- Added `open-unlearning/scripts/kforge_spectrum_summary.py`.
  - Aggregates the 112 spectrum JSON files into
    `open-unlearning/saves/spectrum/kforge_spectrum_summary.csv`.
  - Writes top-ranked modules to
    `open-unlearning/saves/spectrum/kforge_spectrum_top20.csv`.
  - Current top-1 spectrum leaders include layer 0 `self_attn.q_proj`, layer 15
    `mlp.down_proj`, layer 0 `self_attn.k_proj`, and early-layer
    `self_attn.q_proj` modules.
- Updated `open-unlearning/scripts/kforge_week2_init_experiment.sh` to skip
  runs with existing eval summaries when `SKIP_COMPLETED=true`.
- Queued all remaining GPU work serially on GPU 2:
  - `kforge_week2_bcal_b64_20260511T084100Z`: final `B=64` calibration slice.
  - `kforge_week2_init_full_20260511T084100Z`: full init grid for
    `NPO` and `SimNPO`, step budgets `50 100 250 500 1000`, seeds `0 1 2`,
    scratch vs. K-FORGE init, skipping completed smoke runs.
  - `kforge_week2_adaptive_aggressive_20260511T084100Z`: follow-up adaptive
    damping sweep with smaller coefficients `1e-5`, `1e-4`, and `1e-3`.
- Status update:
  - B_cal `B=64` completed all five strengths.
    - At `s=0.004`: `model_utility=0.5842`,
      `forget_Q_A_Prob=0.8295`, `extraction_strength=0.5433`.
  - Full init grid started after B=64.
    - Completed NPO 50-step runs for seeds `0`, `1`, and scratch seed `2`;
      the K-FORGE seed-2 eval is active.
    - Early NPO 50-step result:
      K-FORGE init improves utility and forget probability across the first
      completed seeds compared with scratch.
- User reported one more free GPU.
  - Verified GPU 3 was free.
  - Stopped the previously queued adaptive-aggressive waiter.
  - Started `kforge_week2_adaptive_gpu3_20260511T135640Z` on GPU 3 for the
    aggressive adaptive-damping sweep.
  - Queued `kforge_week2_simnpo_smoke_gpu3_20260511T135640Z` behind it on GPU 3
    for SimNPO init smoke at steps `50 100 250`, seeds `0 1 2`, scratch vs.
    K-FORGE init.
  - The active full NPO grid continues on GPU 2.

## 2026-05-12: Init-grid overnight status

- SimNPO smoke on GPU 3 completed for steps `50`, `100`, and `250`, seeds
  `0`, `1`, and `2`, scratch vs. K-FORGE init.
  - At 50 steps, K-FORGE init improves forget probability from roughly
    `0.869-0.870` to `0.670-0.679`, with utility decreasing from roughly
    `0.596-0.598` to `0.580-0.582`.
  - At 100 steps, K-FORGE init improves forget probability from roughly
    `0.852` to `0.654-0.664`, with utility around `0.585-0.586`.
  - At 250 steps, K-FORGE init improves forget probability from roughly
    `0.751-0.754` to `0.544-0.553`, with utility around `0.589-0.591`.
  - Conclusion: K-FORGE init consistently accelerates SimNPO forgetting at a
    modest utility cost.
- NPO grid status:
  - Completed all seeds for `50`, `100`, `250`, and `500` steps.
  - At 50 and 100 steps, K-FORGE init strongly improves both utility and forget
    probability versus scratch.
  - At 250 steps, K-FORGE init preserves matched utility and improves forget
    probability across all three seeds.
  - At 500 steps, scratch reaches higher utility, but K-FORGE init gives lower
    forget probability and lower extraction strength across all three seeds.
  - The active run is `NPO kforge S1000 seed0`; `NPO scratch S1000 seed0`
    completed with `model_utility=0.5808`, `forget_Q_A_Prob=0.0302`,
    `forget_Q_A_ROUGE=0.2907`, and `extraction_strength=0.1021`.
- Aggressive adaptive damping completed for coefficients `1e-5`, `1e-4`, and
  `1e-3`.
  - Coefficients `1e-5` and `1e-4` collapse utility while driving forget
    probability near zero.
  - Previous larger coefficients were near no-op.
  - Conclusion: the current adaptive damping parameterization is not a useful
    balanced method and should be deprioritized.
- User noted GPU 1 was almost free.
  - Verified GPU 1 had enough free memory and no active compute process.
  - Started `kforge_week2_simnpo_long_gpu1_20260512T104006Z` on GPU 1.
  - This queue runs the long SimNPO init grid for steps `500` and `1000`,
    seeds `0`, `1`, and `2`, scratch vs. K-FORGE init, with
    `SKIP_COMPLETED=true`.
  - The active NPO grid continues on GPU 2.

## 2026-05-12: Proceeding with remaining Week 2/3 queues

- Refreshed live status at `2026-05-12T15:50Z`.
  - Completed Week 2 init eval summaries increased to `50`.
  - `NPO scratch S1000 seed2` completed; `NPO kforge S1000 seed2` started on
    GPU 2.
  - `SimNPO scratch S500 seed0/1` and `SimNPO kforge S500 seed0` completed;
    `SimNPO kforge S500 seed1` started on GPU 1.
- Added a guard tmux session,
  `kforge_stop_gpu2_after_npo_20260512T155644Z`, to stop the older mixed
  `NPO SimNPO` queue after the final NPO eval is written.
  - Rationale: GPU 1 already owns the long SimNPO `500/1000` queue, so allowing
    the older GPU 2 script to enter its SimNPO loop could duplicate active runs.
- Queued the next planned downstream baseline/init check behind GPU 2:
  `kforge_rmu50_gpu2_20260512T155731Z`.
  - It waits for `NPO kforge S1000 seed2` to finish and for the mixed GPU 2
    queue to stop.
  - Then it runs `RMU` at `50` steps, seeds `0 1 2`, scratch vs. K-FORGE init,
    using `scripts/kforge_week2_init_experiment.sh` with
    `TRAINERS=RMU STEP_BUDGETS=50`.
- Refreshed status at `2026-05-12T18:40Z`.
  - `NPO kforge S1000 seed2` completed at `2026-05-12T18:19:04Z`.
  - GPU 2 was free, but the RMU waiter was blocked on a defunct old bash PID.
  - Replaced the stuck waiter with direct session
    `kforge_rmu50_gpu2_20260512T1841_RMU50`.
  - Verified GPU 2 is now running `RMU scratch S50 seed0`.
- Scheduled the 12-hour overnight queue at `2026-05-12T19:34Z`.
  - Added `open-unlearning/scripts/kforge_wait_for_idle_gpu.sh`, a conservative
    idle-GPU waiter that requires sustained low memory and low utilization
    before launching a queued run.
  - GPU 1 remains assigned to
    `kforge_week2_simnpo_long_gpu1_20260512T104006Z`, which will continue the
    SimNPO `500/1000` scratch-vs-K-FORGE grid.
  - GPU 2 will continue `RMU S50` and then chain into
    `kforge_rmu_long_gpu2_20260512T1934Z`, running `RMU` step budgets
    `100 250 500`, seeds `0 1 2`, scratch vs. K-FORGE init.
  - GPU 3 has opportunistic session
    `kforge_forget05_gpu3_overnight_20260512T1934Z`; if the current non-K-FORGE
    job frees the GPU, it will run `forget05` NPO/SimNPO init transfer at steps
    `50 100 250`, seeds `0 1 2`.
  - GPU 0 has opportunistic session
    `kforge_forget01_gpu0_overnight_20260512T1934Z`; it will only start if the
    resident jobs fully clear, then run the analogous `forget01` transfer grid.

## 2026-05-13: Overnight queue status

- Status snapshot at `2026-05-13T07:25Z`.
  - Total `week2` eval summaries reached `101`.
  - GPU 1 continued the SimNPO long queue through `S1000`.
    - Completed `S1000` scratch/kforge seeds `0` and `1`.
    - Active run: `SimNPO scratch S1000 seed2`.
  - GPU 2 completed all `RMU S50`, all `RMU S100`, all `RMU S250`, and most
    `RMU S500`.
    - Active run: `RMU scratch S500 seed2` eval.
    - Remaining in that queue: `RMU kforge S500 seed2`.
  - GPU 3 became idle enough for the opportunistic waiter and launched the
    `forget05` transfer queue.
    - Completed all `forget05 NPO` steps `50 100 250`, seeds `0 1 2`, scratch
      vs. K-FORGE.
    - Began `forget05 SimNPO`; active run is `SimNPO scratch S50 seed1`.
  - GPU 0 did not meet the idle threshold; the `forget01` waiter is still
    waiting.
- Early overnight observations:
  - `forget05 NPO S50`: K-FORGE init improves utility from about `0.46` to
    about `0.54-0.55` and forget probability from about `0.28-0.32` to about
    `0.093-0.095`.
  - `SimNPO S1000 forget10`: K-FORGE init still gives lower forget probability
    and extraction strength than scratch, with similar utility.
  - `RMU S500 forget10`: scratch has higher utility (`~0.589`) than K-FORGE
    init (`~0.574`), while both drive forget probability very low; K-FORGE is
    not showing an RMU acceleration win at this budget so far.
- Status snapshot at `2026-05-13T13:09Z`.
  - The `forget10 SimNPO S1000` queue completed all seeds.
  - The `RMU` long queue completed all scheduled budgets through `S500`.
  - The only remaining K-FORGE queue is `forget05 SimNPO S250` on GPU 3.
    - Active run: `forget05 SimNPO kforge S250 seed0`.
    - Completed in that block: `forget05 SimNPO scratch S250 seed0`.
  - GPUs 0, 1, and 2 are occupied by separate `e1_single_language` jobs, so the
    remaining `forget05` work cannot currently be split without interrupting
    those jobs.
- At `2026-05-13T14:40Z`, GPU 1 became usable.
  - Started `kforge_f05_simnpo_s250_seed2_gpu1_20260513T1440Z` to run
    `forget05 SimNPO S250 seed2` scratch vs. K-FORGE on GPU 1.
  - Added `kforge_stop_f05_gpu3_after_seed1_20260513T1440Z`, which waits for
    the GPU 3 queue to finish `forget05 SimNPO kforge S250 seed1` and then
    stops the old queue before it can duplicate seed 2.
  - Verified GPU 1 is running `forget05 SimNPO scratch S250 seed2`; GPU 3 is
    still running `forget05 SimNPO scratch S250 seed1`.
- Status snapshot at `2026-05-13T16:53Z`.
  - All active K-FORGE experiment queues completed; no `kforge` tmux server or
    active `src/train.py`/`src/eval.py` process remains.
  - Total `week2` eval summaries reached `120`.
  - Completed `forget05 SimNPO S250` for all seeds:
    - Scratch: utility about `0.595-0.598`, forget probability about
      `0.611-0.624`, extraction strength about `0.266-0.268`.
    - K-FORGE init: utility about `0.586-0.587`, forget probability about
      `0.387-0.398`, extraction strength about `0.180-0.188`.
  - Completed `forget10 SimNPO S1000` for all seeds:
    - Scratch forget probability about `0.291-0.302`; K-FORGE init about
      `0.263-0.269`, at similar utility.
  - Completed `forget10 RMU S500` for all seeds:
    - Scratch has higher utility (`~0.589`) than K-FORGE init (`~0.574`), with
      both reaching very low forget probability.

## 2026-05-13: Week 2 analysis and report

- Added `open-unlearning/scripts/kforge_week2_analyze.py`.
  - Parses all `saves/eval/*week2_EVAL_FP32/TOFU_SUMMARY.json` summaries.
  - Writes per-run and aggregate CSVs to `open-unlearning/saves/analysis/week2/`.
  - Writes `week2_summary.md` with seed means/stds.
  - Generates forget10 plots for NPO, SimNPO, RMU, and a Pareto scatter.
- Ran the analysis script on the completed 120-run Week 2 batch.
  - Output directory: `open-unlearning/saves/analysis/week2/`.
  - Key files:
    - `week2_runs.csv`
    - `week2_aggregate.csv`
    - `week2_summary.md`
    - `forget10_pareto_scatter.png`
    - `forget10_npo_forget_prob.png`
    - `forget10_simnpo_forget_prob.png`
    - `forget10_rmu_forget_prob.png`
- Regenerated the spectrum summary with
  `open-unlearning/scripts/kforge_spectrum_summary.py`.
- Added `open-unlearning/docs/kforge_week2_report.md`.
  - Main conclusion: K-FORGE is best framed as a closed-form initializer.
  - NPO and SimNPO show consistent acceleration/stronger forgetting.
  - RMU is mixed: K-FORGE lowers forget metrics but costs utility, so it is not
    a headline RMU win.
- Updated `open-unlearning/docs/kforge_initial_report.md` to point to the new
  Week 2 report.

## 2026-05-13: Next publishability experiments queued

- Started the missing `forget01` transfer grid.
  - `kforge_f01_npo_gpu1_20260513T2004Z`: NPO on GPU 1, steps `50 100 250`,
    seeds `0 1 2`, scratch vs. K-FORGE init.
  - `kforge_f01_simnpo_gpu2_20260513T2004Z`: SimNPO on GPU 2, steps
    `50 100 250`, seeds `0 1 2`, scratch vs. K-FORGE init.
  - K-FORGE init checkpoint:
    `saves/unlearn/KFORGE_TOFU_forget01_R2_M1_B2_S0p003_kron_retain_stage3down`.
- Patched `open-unlearning/scripts/kforge_tofu_sweep.sh` to pass
  `model=${MODEL_ID}` and `model.model_args.pretrained_model_name_or_path`
  explicitly during K-FORGE training.
  - This is required for the planned Llama-3.2-3B smoke; previous 1B runs
    inherited the 1B model from the TOFU experiment config.
- Patched `open-unlearning/scripts/kforge_week2_init_experiment.sh` to accept
  `RUN_TAG` while preserving the default `week2` tag.
- Queued `kforge_3b_smoke_gpu1_after_f01_20260513T2006Z`.
  - Waits for the GPU 1 `forget01 NPO` queue to finish.
  - Generates a Llama-3.2-3B `forget10` K-FORGE init:
    `KFORGE_TOFU_forget10_R2_M1_B2_S0p003_kron_retain_3bsmoke`.
  - Then runs 3B NPO/SimNPO smoke at steps `50 100 250`, seed `0`, scratch vs.
    K-FORGE init.
- GPU 3 showed sustained SM activity and resident memory without a visible
  compute-app PID, so it was not used for the 3B run.
- Status snapshot at `2026-05-14T07:03Z`.
  - `forget01` transfer has completed all `S50` and `S100` rows for both NPO
    and SimNPO, scratch vs. K-FORGE, seeds `0 1 2`.
  - Active runs:
    - GPU 1: `forget01 NPO scratch S250 seed0`.
    - GPU 2: `forget01 SimNPO scratch S250 seed0`.
  - 24 `forget01` eval summaries are complete so far.
  - The 3B smoke has not started yet because it is queued behind the GPU 1
    `forget01 NPO` queue.
  - Early `forget01` results are strong:
    - NPO `S50/S100`: K-FORGE lowers forget probability from roughly
      `0.06-0.10` to `0.01-0.015`.
    - SimNPO `S50/S100`: K-FORGE lowers forget probability from roughly
      `0.55-0.73` to `0.012-0.027`.

## 2026-05-14: Consolidated findings report

- Added `open-unlearning/docs/kforge_current_findings_2026-05-14.md`.
  - Summarizes one-shot K-FORGE, B_cal/adaptive-damping diagnostics,
    `forget10` init results, `forget05` transfer, partial `forget01` transfer,
    RMU findings, spectrum diagnostics, and publishability assessment.
  - Separates completed evidence from currently running experiments.
  - Current conclusion: K-FORGE is best framed as a Kronecker-Fisher
    initializer for NPO/SimNPO, not as a standalone one-shot SOTA unlearner.
- Updated `open-unlearning/docs/kforge_week2_report.md` to link to the current
  findings report.

## 2026-05-14: Paper figure generation pass

- Added `open-unlearning/scripts/kforge_make_figures.py`.
  - Reads saved TOFU summaries, `saves/eval/kforge_all_summary.csv`, and
    `saves/spectrum/kforge_spectrum_summary.csv`.
  - Rebuilds paper figures from disk without rerunning experiments.
  - Writes PNG and PDF outputs plus per-figure source CSVs.
- Generated figures in `open-unlearning/saves/figures/kforge/`.
  - Main text:
    - `fig1_strength_cliff_calibration.{png,pdf}`
    - `fig2_steps_to_target_acceleration.{png,pdf}`
    - `fig3_pareto_frontier_forget10.{png,pdf}`
  - Appendix:
    - `figA1_spectrum_heatmap.{png,pdf}`
    - `figA2_diagonal_vs_kronecker.{png,pdf}`
    - `figA3_forget_only_vs_retain_whitened.{png,pdf}`
  - Manifest:
    - `MANIFEST.md`
  - Reproducibility data:
    - `week2_runs_used.csv`
    - `week2_aggregate_used.csv`
    - `fig1_strength_cliff_calibration_data.csv`
    - `fig2_steps_to_target_data.csv`
    - `fig3_pareto_frontier_data.csv`
    - `figA1_spectrum_heatmap_data.csv`
    - `figA2_diagonal_vs_kronecker_data.csv`
    - `figA3_forget_only_vs_retain_whitened_data.csv`
- Current figure data snapshot at generation time:
  - Parsed `152` completed Week 2 init/eval runs.
  - Aggregated `52` trainer/forget/schedule rows.
  - `forget01` S250 panels are still marked partial because only two seeds
    were complete for S250 at this snapshot.
  - Figure A4 was intentionally not generated: the layer-inclusion acceleration
    sweep needed for that correlation plot is not present yet.
- Adjusted figure layout after visual inspection.
  - Figure 1 uses alpha scaled as `x 10^-3` to avoid overlapping tick labels.
  - Figure 2 hides top-row x tick labels and annotates each panel with the
    strongest observed `scratch steps to match / K-FORGE-init steps used`
    ratio. Labels with `k>` are lower bounds where scratch does not reach the
    K-FORGE forget probability within the available budget.

## 2026-05-14: GPU utilization and remaining baseline scheduling

- Diagnosed the queued Llama-3.2-3B smoke as blocked on HuggingFace model
  fetch rather than GPU compute.
  - Both the 3B K-FORGE-init run and an attempted 3B scratch run stalled at
    `Fetching 2 files: 0%`.
  - The local cache had a stale incomplete shard for
    `open-unlearning/tofu_Llama-3.2-3B-Instruct_full`.
  - Stopped the stalled 3B tmux sessions and removed only the incomplete shard
    and lock files for that model cache.
- Patched the wrappers to avoid the same fetch path.
  - `open-unlearning/scripts/kforge_tofu_sweep.sh` now exports
    `HF_HUB_DISABLE_XET=1` for K-FORGE train/eval jobs.
  - `open-unlearning/scripts/kforge_week2_init_experiment.sh` now exports
    `HF_HUB_DISABLE_XET=1` for train/eval jobs.
  - Added `INIT_MODES` to `kforge_week2_init_experiment.sh`, defaulting to
    `scratch kforge`, so scratch-only and init-only batches can be scheduled
    independently.
- Started the missing RMU transfer grids to keep available GPUs occupied while
  the 3B model cache is repaired.
  - `kforge_f05_rmu_gpu1_20260514T2035Z`: GPU 1, TOFU `forget05`, RMU,
    steps `50 100 250`, seeds `0 1 2`, scratch vs. K-FORGE init.
  - `kforge_f01_rmu_gpu2_20260514T2035Z`: GPU 2, TOFU `forget01`, RMU,
    steps `50 100 250`, seeds `0 1 2`, scratch vs. K-FORGE init.
  - Both sessions started with active GPU processes.
- Started CPU-side 3B model prefetch with Xet disabled.
  - Session: `kforge_3b_prefetch_20260514T2035Z`.
  - Once the 3B cache completes, restart the 3B smoke from the cached model.

## 2026-05-15: Overnight experiment status

- Completed the missing `forget05` RMU transfer grid.
  - Session `kforge_f05_rmu_gpu1_20260514T2035Z` finished all
    scratch/K-FORGE rows for steps `50 100 250`, seeds `0 1 2`.
- Continued the `forget01` RMU transfer grid.
  - Session `kforge_f01_rmu_gpu2_20260514T2035Z` completed all `S50` and
    `S100` rows and entered the final `S250` block.
- Repaired the 3B smoke launch path.
  - The successful prefetch landed in
    `.cache/hf/hub/models--open-unlearning--tofu_Llama-3.2-3B-Instruct_full/`
    while the older stalled jobs were looking in the legacy cache layout.
  - Relaunched the 3B smoke on GPU 1 as
    `kforge_3b_smoke_gpu1_local_20260515T0816Z`, using the completed local
    snapshot path directly.
  - The relaunched run loaded both checkpoint shards immediately and began
    K-FORGE calibration, confirming the network-fetch blocker is resolved.
- Refreshed Week 2 analysis after the overnight completions.
  - `open-unlearning/scripts/kforge_week2_analyze.py` now parses `187`
    completed runs at this snapshot.

## 2026-05-18: Paper draft corrected to match validated Wiener results

- Rewrote `main.tex` to remove the obsolete pre-audit paper story.
  - Replaced the incorrect closed-form rank-r GSVD claim with the corrected
    K-FORGE Wiener formulation:
    - exact unconstrained/full-rank solution,
    - exact rank-r solution only at zero retain penalty,
    - explicit low-rank relaxation for positive retain penalty.
  - Updated the method text to distinguish numerical Fisher damping from the
    retain-budget penalty and to use retain-basis unwhitening.
- Replaced stale experimental claims and tables with corrected results.
  - Added the validated one-shot `wiener_v2` frontier on TOFU `forget10`.
  - Added corrected matched-budget NPO and SimNPO initialization tables for
    strengths `0.45` and `0.60`.
  - Added corrected one-shot transfer summaries for `forget05` and `forget01`.
  - Removed the stale RMU table and the unsupported acceleration-factor claims.
- Reframed the draft conclusion conservatively.
  - Current evidence supports a curvature-aware initializer on `forget10`.
  - Transfer of initialization gains, benchmark generalization, larger-model
    scaling, and robustness audits remain open empirical work.

## 2026-05-19: Added completed transfer rows to the paper draft

- Updated `main.tex` with completed corrected TOFU transfer results.
  - Added a compact three-seed table for `forget05` and `forget01` at
    50/100 steps for NPO and SimNPO.
  - Reported only completed rows; still-running 250-step transfer rows are
    explicitly excluded from the table.
- Updated the paper framing based on the new evidence.
  - SimNPO transfer is consistently positive on `forget05` and `forget01`.
  - NPO transfer is positive on `forget05` but largely saturated on
    `forget01`, where scratch NPO already reaches about 0.01 forget
    probability.
  - Limitations and conclusion now reflect that transfer is positive but more
    nuanced than the `forget10` headline.

## 2026-05-20: Added partial S250 SimNPO transfer rows

- Updated `main.tex` with newly completed S250 transfer summaries.
  - Added two-seed S250 rows for SimNPO on `forget05` and `forget01`.
  - Marked those rows with a dagger instead of presenting them as final
    three-seed aggregates.
- Updated framing to reflect the current S250 trend.
  - Current S250 SimNPO partial aggregates preserve the same direction as the
    completed 50/100 rows: K-FORGE-init has lower forget probability than
    scratch with similar utility.
  - The final S250 seeds are still running and should replace the dagger rows
    once complete.

## 2026-05-21: Filled free GPUs with matched-init controls

- Parsed the corrected TOFU transfer grid from disk.
  - The previously partial S250 rows for `forget05` and `forget01` are now
    complete for scratch and K-FORGE init.
  - The random matched-norm NPO init-control on `forget10` is complete for
    `S50` and `S100`, seeds `0 1 2`.
- Added matched-init control launch scripts.
  - `open-unlearning/scripts/run_init_control_weight_svd_npo.sh` runs a
    weight-SVD rank-2 matched-norm control for NPO on `forget10`.
  - `open-unlearning/scripts/run_init_control_random_simnpo.sh` runs the
    random rank-2 matched-norm control for SimNPO on `forget10`.
- Started the new controls on available GPUs.
  - `kforge_initctl_weightsvd_npo_f10_gpu1_20260521T0734Z`: GPU 1, NPO,
    weight-SVD matched init, steps `50 100`, seeds `0 1 2`.
  - `kforge_initctl_random_simnpo_f10_gpu2_20260521T0736Z`: GPU 2, SimNPO,
    random matched init, steps `50 100`, seeds `0 1 2`.

## 2026-05-22: Updated paper tables with init-control ablations

- Updated `main.tex` with completed corrected transfer and control results.
  - Replaced the partial S250 transfer rows with final three-seed aggregates.
  - Added a matched-init control table on TOFU `forget10` comparing scratch,
    random matched-norm, weight-SVD matched-norm, diagonal, forget-only, and
    K-FORGE for both NPO and SimNPO.
  - Added a transfer-control table for `forget05` and `forget01`, including
    random and weight-SVD matched-norm controls where available.
- Refreshed the abstract/introduction and analysis text.
  - Removed stale partial-result language.
  - Reframed the empirical claim around K-FORGE as a curvature-aware
    initializer whose gains are not reproduced by generic matched low-rank
    perturbations.
- Added seed variability to the new control tables.
  - `tab:init_controls_forget10` and `tab:transfer_controls` now report
    Model Utility / Forget Probability as mean plus standard deviation over
    three seeds.
- Expanded the experimental baseline description in `main.tex`.
  - Added explicit rationale for scratch NPO/SimNPO, matched-norm random and
    weight-SVD controls, and structural diagonal/forget-only ablations.
  - Clarified why RMU/GradDiff are treated as future method-level comparisons
    rather than initializer controls in the current draft.
  - Added positioning against initialization-method competitors such as
    PiSSA, LoRA-GA, CorDA, and LoRA-DA, explaining why the direct-edit
    weight-SVD control is the closest current analogue.
  - Added an explicit discussion of the actual experimental initializer
    competitors: random matched, weight-SVD matched, diagonal Fisher, and
    forget-only Fisher.

## 2026-05-22: Scheduled minimal Llama-3.2-3B upgrade

- Verified that the OpenUnlearning TOFU Llama-3.2-3B-Instruct weights are
  already present locally, along with `retain90` evaluation logs.
  - Because the TOFU-tuned 3B checkpoint is available, the scheduled runs use
    `open-unlearning/tofu_Llama-3.2-3B-Instruct_full` rather than falling back
    to `unsloth/Llama-3.2-3B-Instruct`.
- Added and launched two queued 3B scripts.
  - `open-unlearning/scripts/run_3b_minimal_gpu1_kforge.sh`: waits for GPU 1,
    creates the corrected v2 K-FORGE 3B initializer, then runs NPO S50/S100
    K-FORGE-init seeds `0 1 2`.
  - `open-unlearning/scripts/run_3b_minimal_gpu2_scratch.sh`: waits for GPU 2,
    then runs NPO S50/S100 scratch seeds `0 1 2`.
  - Sessions: `kforge_3b_minimal_kforge_gpu1_20260522T1158Z` and
    `kforge_3b_minimal_scratch_gpu2_20260522T1158Z`.

## 2026-05-22: Regenerated corrected paper figures

- Added `open-unlearning/scripts/kforge_make_corrected_figures.py`.
  - Parses corrected `wiener_v2` one-shot summaries, `v2lam0p01_corr`
    downstream summaries, and `initctrl` initializer-control summaries
    directly from `open-unlearning/saves/eval`.
  - Writes regenerated figure source CSVs alongside the images so the plotted
    data are auditable.
- Generated corrected figure set under
  `open-unlearning/saves/figures/kforge_corrected`.
  - `fig1_wiener_v2_strength_sweep.{png,pdf}` replaces the stale
    pre-correction B-calibration cliff plot for the current story.
  - `fig2_corrected_steps_to_target.{png,pdf}` regenerates the
    scratch-versus-K-FORGE matched-budget plot from the corrected three-seed
    runs.
  - `fig3_corrected_pareto_forget10.{png,pdf}` regenerates the Pareto frontier
    using corrected one-shot, downstream, and init-control points.
  - `figA1_spectrum_heatmap.{png,pdf}` and
    `figA2_init_controls_forget10.{png,pdf}` refresh appendix/supporting plots.
- Updated `main.tex` to include the corrected figure directory and insert the
  regenerated strength-sweep, steps-to-target, and Pareto figures in the
  experiments section.

## 2026-05-22: Fixed corrected Pareto figure operating-region view

- Updated `fig3_corrected_pareto_forget10` generation to avoid letting
  catastrophic one-shot over-edits define the visual frontier.
  - The full source CSV still includes all points.
  - The plotted CSV excludes one-shot points with utility below `0.45`.
  - The excluded low-utility one-shot failures are written separately to
    `fig3_corrected_pareto_forget10_excluded_low_utility_data.csv`.
- Regenerated `fig3_corrected_pareto_forget10.{png,pdf}` and updated the
  figure caption in `main.tex` to state this operating-region filter.

## 2026-05-22: Reconciled manuscript tables with latest corrected results

- Updated `main.tex` tables against the current corrected figure/data CSVs.
  - Refreshed the one-shot \texttt{forget10} table from
    `fig1_wiener_v2_strength_sweep_data.csv`, including the corrected ROUGE
    values and the $\alpha=.10/.30$ operating-region rows.
  - Added mean$\pm$std for Utility and Forget Probability in the main NPO and
    SimNPO \texttt{forget10} initialization tables.
  - Added the completed three-seed \texttt{forget05}/\texttt{forget01}
    scratch-versus-K-FORGE transfer table through S250.
  - Restored the \texttt{forget10} matched initializer-control table with
    scratch, random matched, weight-SVD matched, diagonal, forget-only, and
    K-FORGE rows.
- Updated nearby manuscript text that still described corrected transfer runs
  or ablations as unfinished.

## 2026-05-23: Prepared repository hygiene for commit

- Added root repository metadata.
  - `README.md` describes the K-FORGE project, important files, experiment
    entry points, figure generation, and commit hygiene.
  - `.gitignore` excludes local settings, caches, raw logs, checkpoints,
    raw eval outputs, and model-weight artifacts.
  - `.gitattributes` normalizes text files and marks generated figures/model
    checkpoint formats appropriately.
- Preserved curated corrected paper figures for version control.
  - Raw `open-unlearning/saves/` artifacts remain ignored.
  - `open-unlearning/saves/figures/kforge_corrected/` remains trackable.
- Flattened the vendored OpenUnlearning tree for the new root repository.
  - Moved the nested `open-unlearning/.git` metadata to the ignored backup
    directory `.open-unlearning.git.backup/`.
  - This lets the root repository track OpenUnlearning source files normally
    instead of committing `open-unlearning/` as an embedded git repository.

## 2026-05-31: Review by the Hermes Agent

    I reviewed the repo and the paper in main.tex, and ran several concrete checks:

    - Read main.tex fully.
    - Checked active citations, labels, refs, and figure paths.
    - Attempted LaTeX compilation: blocked because latexmk/pdflatex are not installed in this environment.
    - Ran static Python compilation for:
      - open-unlearning/src/trainer/unlearn/kforge.py
      - open-unlearning/scripts/kforge_make_corrected_figures.py
    - Regenerated corrected figures into /tmp/kforge_figcheck.
    - Checked tracked figure/data artifacts and README/script consistency.
    - Visually inspected the two figures used in main.tex.

    Bottom line:
    The paper has a coherent and much improved story: K-FORGE is framed as a Fisher-guided initializer rather than overclaiming one-shot unlearning. The empirical table in the main paper is compelling for early-budget NPO/SimNPO on TOFU. However, the repo/paper is currently not submission-ready because of several hard reproducibility and LaTeX blockers, plus a few scientific-positioning risks reviewers will likely hit.

    1. Hard blockers before submission / sharing

    1.1 main.tex cannot compile from the repo as-is

    Current main.tex uses:

    - \usepackage[review]{acl}
    - \bibliography{custom}
    - figures under latex/figures/...

    But the repo appears to lack:

    - acl.sty
    - custom.bib
    - latex/figures/fig2_improved.pdf
    - latex/figures/fig_matched_init_arrows.pdf

    I verified:

    - search_files("*.bib") found zero .bib files.
    - search_files("*.sty") found zero .sty files.
    - Figure files exist under:
      - open-unlearning/saves/figures/kforge_corrected/fig2_improved.pdf
      - open-unlearning/saves/figures/kforge_corrected/fig_matched_init_arrows.pdf
      but not under latex/figures/.

    Fix:
    Either create a proper paper directory:

    text
    paper/
      main.tex
      custom.bib
      acl.sty
      figures/
        fig2_improved.pdf
        fig_matched_init_arrows.pdf


    and update paths to figures/..., or change the current \includegraphics paths to the tracked figure location.

    At minimum:

    latex
    \includegraphics[...]{open-unlearning/saves/figures/kforge_corrected/fig2_improved.pdf}
    \includegraphics[...]{open-unlearning/saves/figures/kforge_corrected/fig_matched_init_arrows.pdf}


    But for submission cleanliness, I would not reference open-unlearning/saves/... from the paper. Copy curated figures into latex/figures/ or paper/figures/.

    1.2 Missing bibliography is the most serious paper artifact issue

    main.tex actively cites 29 distinct citation keys, including:

    text
    tofu, rmu, muse, npo, simnpo, openunlearning, foster24, fila,
    kfade, winu, chekalina_2026_gfwsvd, crispedit, pissa,
    loraga, corda, lorada, ...


    But custom.bib is absent. This means the paper cannot be independently built, and none of the citations are currently verifiable from the repo.

    Fix:
    Add custom.bib and verify every key in main.tex resolves. Given this is a research paper, do not hand-write from memory; fetch BibTeX programmatically from arXiv/Semantic Scholar/CrossRef.

    1.3 README reproduction command does not reproduce the paper’s actual K-FORGE initializer

    README gives:

    bash
    INIT_MODES="scratch kforge" \
    RUN_TAG=v2lam0p01_corr \
    bash scripts/kforge_week2_init_experiment.sh


    But scripts/kforge_week2_init_experiment.sh defaults to:

    bash
    KFORGE_INIT_MODEL=saves/unlearn/KFORGE_TOFU_forget10_R2_M1_B2_S0p00305_kron_retain_stage3down
    KFORGE_INIT_TAG=kforge


    That does not match the paper’s corrected headline setup, which is described as:

    - wiener_v2
    - rank r=2
    - lambda_ret = 1e-2
    - strengths alpha=.45 and .60
    - corrected run tags like kforge_s045, kforge_s06, v2lam0p01_corr

    Fix:
    README needs exact commands for producing the paper rows, e.g. something like:

    bash
    First create the corrected K-FORGE initializer checkpoint
    python src/train.py ... \
      trainer=KFORGE \
      trainer.method_args.rank=2 \
      trainer.method_args.strength=0.45 \
      trainer.method_args.edit_variant=wiener_v2 \
      trainer.method_args.lambda_tradeoff=0.01 \
      trainer.method_args.max_target_modules=1 \
      trainer.method_args.target_modules_regex='.*mlp\.down_proj$' \
      ...

    Then run downstream initialization experiments
    KFORGE_INIT_MODEL=saves/unlearn/<actual_checkpoint> \
    KFORGE_INIT_TAG=kforge_s045 \
    RUN_TAG=v2lam0p01_corr \
    ...
    bash scripts/kforge_week2_init_experiment.sh


    Right now a fresh user following README would likely run the old/default initializer, not the paper one.

    1.4 Paper and implementation defaults are inconsistent

    open-unlearning/configs/trainer/KFORGE.yaml defaults to:

    yaml
    rank: 8
    strength: 1.0
    edit_variant: legacy_v1
    lambda_tradeoff: 0.0
    max_target_modules: 4


    But the paper’s main claims use corrected wiener_v2, rank 2, lambda 0.01, and one selected mlp.down_proj layer.

    This is fine internally if documented, but dangerous for reviewers/users.

    Fix:
    Either:

    - make the default config the paper config, or
    - add a separate KFORGE_paper_wiener_v2.yaml, and make README use that.

    Suggested:

    text
    open-unlearning/configs/trainer/KFORGE_wiener_v2_paper.yaml


    with:

    yaml
    method_args:
      target_modules_regex: .*mlp\.down_proj$
      rank: 2
      strength: 0.45
      damping: 1e-4
      max_calibration_batches: 32
      max_target_modules: 1
      factor_mode: kron
      use_retain_fisher: true
      edit_variant: wiener_v2
      lambda_tradeoff: 1e-2
      edit_weight_dtype: float32


    2. Scientific evaluation

    2.1 The main story is good

    The paper’s current framing is much stronger than the original plan implied by PLAN.md.

    Original planned story:
    “closed-form, one-shot K-FORGE beats iterative unlearning.”

    Current paper story:
    “one-shot K-FORGE is useful but insufficient; its real value is as a curvature-aware initializer for NPO/SimNPO.”

    That is the right pivot. The current results support this narrower claim better.

    The strongest evidence is Table 2 / tab:init_forget10:

    For 1B TOFU forget10:

    - NPO S50:
      - scratch: utility 0.509, forget prob 0.093
      - K-FORGE .60: utility 0.574, forget prob 0.045
    - SimNPO S50:
      - scratch: utility 0.577, forget prob 0.662
      - K-FORGE .60: utility 0.571, forget prob 0.534

    This is a clean and understandable contribution: K-FORGE moves early optimization to a better part of the forget-retain frontier.

    2.2 The main reviewer objection will be “matched steps are not matched compute”

    You report K-FORGE overhead in Table tab:kforge_compute, which is good. But the main comparisons are matched by downstream training steps, not by wall-clock/FLOPs.

    You state K-FORGE overhead is about 9–11% of a 50-step scratch run. Reviewers may ask:

    - Would scratch NPO with 55 steps close the gap?
    - Would scratch SimNPO with 55/110/275 steps close the gap?
    - Is the advantage still present under wall-clock matched budgets?

    Fix:
    Add one small compute-matched control. You do not need a full sweep. For the headline case, run:

    - NPO scratch at 55 or 56 steps vs K-FORGE+50
    - SimNPO scratch at 55 or 56 steps vs K-FORGE+50

    If expensive, at least interpolate from extra checkpoints if available. Then phrase claims as:

    “K-FORGE improves matched downstream-step budgets; the advantage remains under an approximate wall-clock-matched check at S50+overhead.”

    Without this, keep saying “matched downstream optimization budgets,” not “matched budgets” unqualified.

    2.3 Statistical testing is currently too light

    The paper reports means/std over 3 seeds, but no confidence intervals or paired tests. Three seeds is common but fragile; reviewers may still ask for significance.

    Fix:
    Add a compact statistical paragraph:

    - Use paired bootstrap over seeds/tasks where possible.
    - Report 95% CI for key delta:
      - Δ forget probability
      - Δ utility
    - Or at least report per-seed deltas in appendix.

    Example wording:

    “Across three seeds, K-FORGE reduces forget probability for SimNPO S50/S100/S250 by X/Y/Z with no utility loss; bootstrap 95% CIs are ...”

    2.4 The scope is honest, but maybe too narrow for a main conference unless positioned carefully

    Current scope:

    - TOFU only
    - mostly Llama-3.2-1B
    - limited 3B sanity check
    - NPO/SimNPO only
    - no MUSE/WMDP
    - no relearning/robustness audit

    The paper admits this in limitations, which is good. But reviewers may ask whether this is enough.

    Best framing:
    This is not a new full unlearning method. It is an initializer/primitive. The correct review category is closer to “optimization/initialization for unlearning” than “SOTA unlearning method.”

    Strengthen contribution wording:
    Avoid implying it competes with RMU/K-FADE as a full unlearning system. The current text mostly does this well, but some parts still invite SOTA comparison.

    For example, line 126 says:

    “one-shot or closed-form methods ... tend to either under-forget or sharply damage retained capabilities.”

    Fine.

    But the introduction should explicitly say:

    “We do not aim to replace full unlearning algorithms; instead, we test whether curvature information can improve their early optimization trajectory.”

    You say this later; I’d move it earlier.

    2.5 Need a stronger explanation of why the objective corresponds to unlearning

    The method objective is:

    latex
    J(Δ) = ||W + Δ||^2_{F_f} + λ_ret ||Δ||^2_{F_r}


    This is mathematically clear, but the paper needs a more intuitive bridge:

    - Why does shrinking the forget-Fisher norm of the weight reduce forget answer probability?
    - Is this derived from a local quadratic approximation to forget loss, influence, memorization, or Fisher-weighted weight energy?
    - Why is W + Δ ≈ 0 in forget-sensitive coordinates the right target?

    Right now the paper says what it does, but a skeptical reviewer may see it as a heuristic dressed in second-order notation.

    Fix:
    Add a short paragraph before Eq. 4:

    “Following Fisher-weighted compression/editing, we interpret high forget-Fisher directions as coordinates in which the current weight most supports forget-set likelihood. A one-shot erasure edit should shrink the component of the weight that lies in these forget-sensitive directions, while measuring collateral movement in the retain Fisher. This yields the quadratic surrogate...”

    2.6 The method description may overstate “MFF” relative to the implementation

    main.tex says the empirical Fisher factors are recovered through “matrix-free Fisher factorization (MFF)” and a truncated SVD of the rearranged Fisher.

    But the implementation in kforge.py directly collects K-FAC-style factors:

    python
    A = x.T @ x
    B = g.T @ g


    from activations and output gradients.

    That is a standard Kronecker empirical Fisher/K-FAC estimator, but not obviously the same as “MFF via Lanczos rearranged Fisher SVD” as described in lines 188–196.

    This mismatch could be serious if a reviewer checks code.

    Fix one of these:

    Option A: Change paper wording to match code:
    “we estimate Kronecker empirical Fisher factors from activation and output-gradient covariances.”

    Option B: Change code/docs if you truly use MFF elsewhere.

    I would choose A unless you have a separate MFF implementation path. The current paper can still cite MFF/GFWSVD as inspiration, but should not imply the implementation uses the exact MFF algorithm if it does not.

    2.7 7B-scale tractability claim is unsupported by experiments

    Line 196 says the construction is tractable for 7B-scale models.

    But experiments are 1B and a limited 3B sanity check, and only one mlp.down_proj layer is edited in the paper setup.

    Fix:
    Change “making the procedure tractable for 7B-scale models” to a more cautious sentence:

    “making the procedure tractable for the 1B–3B settings and single-layer edits studied here, and potentially scalable to larger settings with the same matrix-free machinery.”

    3. Paper structure and writing

    3.1 Abstract is solid but could be sharper

    Current abstract is accurate but somewhat long and generic. It says:

    - preference unlearning has early optimization problem
    - K-FORGE uses forget/retain Fisher
    - one-shot frontier
    - initializer improves matched-budget runs
    - controls show not arbitrary low-rank perturbation
    - 3B sanity check

    Good.

    Suggested sharper 5-sentence version:

    1. Preference-based LLM unlearning can achieve useful forget-retain tradeoffs, but early training spends many steps discovering directions that suppress forgotten content without damaging retained capability.
    2. We introduce K-FORGE, a Kronecker-Fisher initializer that estimates separate forget and retain empirical Fishers, simultaneously diagonalizes them, and applies a closed-form low-rank Wiener edit before NPO or SimNPO training.
    3. The full-rank edit has an exact solution; for practical low-rank edits we use a stated separable relaxation that is exact in the full-rank and zero-retain-penalty limits.
    4. On TOFU with Llama-3.2-1B, one-shot K-FORGE traces a smooth forget-utility frontier but does not replace iterative unlearning; as an initializer, it consistently lowers forget probability at matched downstream step budgets while preserving utility.
    5. Matched random, weight-SVD, diagonal-Fisher, forget-only, additional-split, and 3B sanity checks support the view that K-FORGE provides a useful curvature-aware starting direction for preference-based unlearning.

    3.2 “Scratch training” is misleading terminology

    The paper uses “scratch” to mean “pretrained checkpoint initialized without K-FORGE.” In ML, “scratch training” often means random initialization.

    Fix:
    Use:

    - “pretrained initialization”
    - “base-checkpoint initialization”
    - “uninitialized NPO/SimNPO”
    - “standard NPO/SimNPO initialization”

    For table labels, “scratch” is okay if defined, but in prose I’d avoid it.

    3.3 Section “Why K-FORGE Initialization Helps” is too short

    This section is currently one paragraph and mostly restates the result. Since it has a strong title, reviewers will expect analysis.

    Possible additions:

    - Show/edit spectrum:
      - Are there a few high forget/retain generalized singular directions?
      - Does K-FORGE move mostly in those?
    - Show gradient alignment:
      - cosine between K-FORGE delta and first NPO/SimNPO gradients
      - does K-FORGE reduce early gradient norm or change direction?
    - Show initial model after K-FORGE is already closer to final NPO/SimNPO checkpoint.
    - Use existing figA1_spectrum_heatmap if it is meaningful.

    If no new analysis is available, rename section to “Interpretation” and keep it modest.

    3.4 Related work is good but needs K-FADE positioning to be extra crisp

    K-FADE is likely the most dangerous related work. The paper mentions it, but should make the distinction unmissable:

    - K-FADE: iterative Gauss-Newton/second-order unlearning update.
    - K-FORGE: closed-form low-rank two-Fisher edit used as initializer.
    - K-FORGE’s downstream objective remains NPO/SimNPO; the intervention is only the initial checkpoint.

    Add a sentence like:

    “Unlike K-FADE, which uses curvature to define the unlearning update itself, K-FORGE uses curvature once, before training, to choose an initial displacement; the downstream optimizer and preference loss are unchanged.”

    4. Figures and tables

    4.1 Figure path issue aside, Figure 2 is visually strong

    fig2_improved is clear:

    - 2 rows: NPO / SimNPO
    - 3 columns: forget10 / forget05 / forget01
    - x-axis training steps
    - y-axis forget probability
    - scratch vs K-FORGE init

    It supports the claim well.

    Minor visual issues:

    - “less desired / more desired” background text is useful but a little large.
    - For forget01 NPO, the y-range is tiny and error bars overlap; the caption correctly says it is near the noise floor. Good.
    - Legend inside first panel uses space but is acceptable.

    4.2 Figure fig_matched_init_arrows is attractive but less necessary

    The arrow figure is readable and intuitive, but:

    - It duplicates Table 2’s message.
    - The orange arrows are visually dominant.
    - The SimNPO x-axis range is narrow, making utility changes look more dramatic than they are.
    - It only shows forget10, while Figure 2 has broader split coverage.

    Recommendation:
    If page budget is tight, keep Figure 2 and move arrow figure to appendix. If you keep both, make the arrow figure the main intuitive “Pareto shift” visual and reduce table size.

    4.3 Table 1 bolding is potentially misleading

    In one-shot table, the best forget metrics occur at alpha 0.8, while best utility is base / alpha 0.1. That is technically correct but can encourage the reader to compare across trade-off points.

    Fix:
    Consider replacing bolding with a Pareto-frontier indication or say:

    “Bold indicates column optimum, not a preferred operating point.”

    4.4 Table 2 is strong but dense

    The main table has many numbers. Consider adding a delta column:

    text
    Δ Forget Pr.
    Δ Utility


    For each K-FORGE row vs scratch. Reviewers should not have to mentally subtract.

    Example:

    text
    S50 NPO K-FORGE .60: ΔForget -0.048, ΔUtility +0.065


    5. Repo/reproducibility review

    5.1 Good aspects

    The repo has many good signs:

    - Root README explains the current story.
    - CHANGELOG.md records the important correction from legacy_v1 to wiener_v2.
    - Corrected figure data CSVs are tracked under:
      - open-unlearning/saves/figures/kforge_corrected/
    - The figure script successfully runs:
      - parsed 162 corrected downstream runs
      - parsed 102 controls
      - parsed 9 one-shot points
    - Static compilation passed for the K-FORGE implementation and figure script.

    5.2 But clean-clone reproduction is currently incomplete

    The figure script works in this local checkout because ignored eval outputs are present under open-unlearning/saves/eval. A clean clone would have only the tracked CSVs/figures, not the raw eval summaries.

    Fix:
    Add one of these:

    Option A:
    Track minimal TOFU_SUMMARY.json files needed to regenerate figures/tables.

    Option B:
    Make kforge_make_corrected_figures.py support a --from-csv mode using tracked CSVs.

    Option C:
    Add a paper_data/ directory containing curated CSVs for every table/figure, and make all paper plots/tables generated from it.

    I recommend C:

    text
    paper_data/
      one_shot_forget10.csv
      init_forget10.csv
      init_controls_forget10.csv
      transfer_controls.csv
      init_3b_sanity.csv
      compute_overhead.csv


    Then add:

    bash
    python scripts/make_paper_tables.py
    python scripts/kforge_make_corrected_figures.py --from-paper-data paper_data/


    5.3 Add a table-generation script

    Main paper tables are manually embedded in LaTeX. This is risky.

    Fix:
    Add:

    text
    open-unlearning/scripts/kforge_make_paper_tables.py


    that produces .tex table fragments from the tracked CSVs:

    text
    paper/tables/oneshot_forget10.tex
    paper/tables/init_forget10.tex
    paper/tables/init_controls_forget10.tex
    ...


    Then in main.tex use:

    latex
    \input{tables/init_forget10}


    This prevents transcription errors.

    5.4 MANIFEST.md is stale or inconsistent

    Tracked manifest says:

    text
    Corrected downstream seed runs parsed: 153


    But I regenerated and got:

    text
    Parsed 162 corrected downstream runs, 102 controls, 9 one-shot points.


    Fix:
    Regenerate and commit the manifest, or make it clear which run set the tracked figures were generated from.

    6. Code review notes

    6.1 K-FORGE implementation compiles

    This passed:

    bash
    cd open-unlearning
    python -m py_compile src/trainer/unlearn/kforge.py scripts/kforge_make_corrected_figures.py


    No syntax errors.

    6.2 Implementation has useful safety choices

    Good:

    - single-process guard
    - eval mode during calibration
    - shifted causal-LM token mask
    - summed loss scaling by valid token count
    - fp32 checkpoint edit option
    - separate legacy_v1 and wiener_v2
    - logs Fisher factor scales
    - supports diagonal / forget-only ablations

    6.3 Potential issue: paper says MFF, code looks K-FAC

    As noted above, code estimates direct activation/gradient covariances, not a Lanczos MFF SVD. Align wording.

    6.4 Potential issue: default save_strategy: no

    KFORGE.yaml has:

    yaml
    save_strategy: 'no'


    This may be okay if the training harness saves at the end anyway, but for a one-shot checkpoint-producing trainer it is worth verifying from a clean run. If a user runs trainer=KFORGE, they expect an edited checkpoint. README should show a tested command that leaves a checkpoint at a known path.

    7. Priority fix list

    P0 — must fix before anyone else can build/evaluate

    1. Add custom.bib.
    2. Add/copy ACL style or document TeX template setup.
    3. Fix \includegraphics paths or create latex/figures/.
    4. Add exact reproduction commands for the paper’s wiener_v2 K-FORGE checkpoints.
    5. Make config defaults or paper-specific config match the paper.

    P1 — must fix before submission

    6. Add compute-matched S50+overhead control or carefully qualify “matched downstream steps.”
    7. Add statistical CIs / paired seed deltas for headline claims.
    8. Align paper’s “MFF” wording with actual K-FAC-style implementation.
    9. Add table-generation or table-verification script.
    10. Update README figure list and manifest.

    P2 — strong quality improvements

    11. Expand “Why K-FORGE Initialization Helps” with spectrum/gradient/direction analysis.
    12. Move arrow figure to appendix unless it is central to narrative.
    13. Add delta columns to main initialization table.
    14. Replace “scratch training” terminology with “base-checkpoint initialization.”
    15. Add more explicit K-FADE distinction.

    8. Overall assessment

    Current readiness:

    - Scientific idea: promising.
    - Paper story: good and appropriately scoped.
    - Main empirical result: credible and useful, especially for early-budget NPO/SimNPO.
    - Repo state: not yet reproducible from clean checkout.
    - Submission readiness: not ready until LaTeX/bib/figure/config/reproduction blockers are fixed.

    My confidence:
    This can become a solid workshop or main-conference submission if positioned as “curvature-aware initialization for preference-based unlearning,” not as a full unlearning SOTA method. The most important immediate work is not more writing; it is making the artifact buildable and making the exact paper experiments reproducible from documented commands.
