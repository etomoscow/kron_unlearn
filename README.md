# K-FORGE

K-FORGE is a Kronecker-Fisher Wiener initialization method for LLM unlearning.
This repository contains the paper draft, experiment notes, and the modified
OpenUnlearning code used for the TOFU and MUSE experiments.

The current paper story is:

- one-shot K-FORGE gives a controllable retain-forget frontier, but is not the
  final unlearning method;
- K-FORGE is used as an initializer for NPO and SimNPO;
- matched controls test whether the gain is explained by a generic low-rank
  perturbation, a weight-SVD direction, diagonal Fisher, or forget-only Fisher;
- the Llama-3.2-1B Forget Q/A Probability gain remains after charging the full
  setup in FLOPs and wall-clock time, and a held-out Gemma seed preserves the
  primary direction;
- MUSE transfer is metric-dependent, and matched relearning shows that
  K-FORGE is an optimization initializer rather than a recovery defense.

## Repository Layout

```text
.
├── main.tex                         # current manuscript draft
├── CHANGELOG.md                     # implementation and experiment log
├── PLAN.md                          # working experiment plan
├── RESEARCH.md                      # research notes
└── open-unlearning/
    ├── src/trainer/unlearn/kforge.py
    ├── configs/trainer/KFORGE.yaml
    ├── configs/experiment/unlearn/tofu/kforge.yaml
    ├── configs/model/gemma-3-1b-it.yaml
    ├── scripts/kforge_make_corrected_figures.py
    ├── scripts/kforge_week2_init_experiment.sh
    ├── scripts/summarize_rebuttal_additions.py
    ├── rebuttal_metrics_snapshot.json
    ├── tests/
    ├── scripts/make_matched_init_checkpoint.py
    └── saves/figures/kforge_corrected/
```

Most generated checkpoints, raw evals, logs, and caches are intentionally
ignored. The curated corrected figures and the CSVs used to draw them are kept
under `open-unlearning/saves/figures/kforge_corrected/`.

## Environment

The code is based on OpenUnlearning. From `open-unlearning/`:

```bash
conda create -n unlearning python=3.11
conda activate unlearning
pip install ".[lm-eval]"
pip install --no-build-isolation flash-attn==2.6.3
python setup_data.py --eval
```

The experiments use Llama-3.2-1B/3B-Instruct, Gemma-3-1B-IT,
Qwen2.5-1.5B-Instruct, and Llama-2-7B checkpoints through the same
OpenUnlearning harness.
Reported paired wall-clock measurements use single-GPU NVIDIA RTX PRO 6000
Blackwell hardware; no multi-GPU speedup is included.

## Rebuttal Configurations

Unless a row overrides it, K-FORGE uses rank 2, factor damping `1e-4`, retain
penalty `0.01`, 32 forget and 32 retain calibration batches, and one
`mlp.down_proj` module. TOFU training uses effective batch size 32; MUSE uses
effective batch size 16.

| Evaluation | Model / target | Strength | Budgets | Seeds | Eval dtype |
|---|---|---:|---|---|---|
| TOFU primary | Llama-3.2-1B, layer 15 | 0.60 | 50, 100, 250 | 0, 1, 2 | FP32 |
| TOFU scale | Llama-3.2-3B, layer 15 | 0.45 | 50, 100, 250 | 0, 1, 2 | FP32 |
| TOFU family | Gemma-3-1B, layer 13 | 0.80 | 50, 100 | 0, 1, 2, 3 | FP32 |
| TOFU null pilot | Qwen2.5-1.5B, layer 15 | 0.45 | 50, 100 | 0, 1, 2 | FP32 |
| MUSE transfer | Llama-2-7B, layer 15 | 0.45 | 50, 100 | 0, 1, 2 | BF16 |

Gemma strength was selected from the one-shot sweep before downstream
training by minimizing Forget Q/A Probability subject to at most `0.01`
utility loss; seed 3 was then evaluated without retuning. The MUSE follow-up
used the analogous one-shot retain constraint and selected strength `1.0`.
The primary endpoint is Forget Q/A Probability with a prespecified mean-utility
margin of `-0.01`; extraction and Forget Q/A ROUGE are always reported as
separate secondary endpoints.

## K-FORGE Entry Points

Core implementation:

```text
open-unlearning/src/trainer/unlearn/kforge.py
```

Main config:

```text
open-unlearning/configs/trainer/KFORGE.yaml
open-unlearning/configs/experiment/unlearn/tofu/kforge.yaml
```

Corrected figure generation:

```bash
cd open-unlearning
python scripts/kforge_make_corrected_figures.py
```

Initialization experiment harness:

```bash
cd open-unlearning
GPU_ID=0 \
MODEL_ID=Llama-3.2-1B-Instruct \
FORGET_SPLIT=forget10 \
RETAIN_SPLIT=retain90 \
TRAINERS=NPO \
STEP_BUDGETS="50 100 250" \
SEEDS="0 1 2" \
INIT_MODES="scratch kforge" \
RUN_TAG=v2lam0p01_corr \
bash scripts/kforge_week2_init_experiment.sh
```

Matched-norm initialization controls are generated through:

```text
open-unlearning/scripts/make_matched_init_checkpoint.py
open-unlearning/scripts/run_init_control_random_npo.sh
open-unlearning/scripts/run_init_control_random_simnpo.sh
open-unlearning/scripts/run_init_control_weight_svd_npo.sh
open-unlearning/scripts/run_init_control_weight_svd_simnpo.sh
```

## Rebuttal Reproduction

The added model-family, compute-matched, benchmark-transfer, and robustness
results are parsed directly from their structured evaluation summaries:

```bash
python open-unlearning/scripts/summarize_rebuttal_additions.py --check
cd open-unlearning && PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'
```

The repository also includes a compact snapshot of the per-seed aggregate
metrics, so the reviewer-requested aggregate tables and paired tests can be
verified without model checkpoints or generated text:

```bash
python open-unlearning/scripts/summarize_rebuttal_additions.py \
  --snapshot-in open-unlearning/rebuttal_metrics_snapshot.json \
  --check
```

To regenerate that snapshot from local evaluation summaries, add
`--snapshot-out open-unlearning/rebuttal_metrics_snapshot.json` to the first
aggregation command.

Given the scratch and K-FORGE checkpoints produced by the initialization
harness above, the matched recovery audits can be rerun from
`open-unlearning/` with:

```bash
# Required only by the 4/8-bit runners on the tested Blackwell environment.
python -m pip install --target .cache/quant_bnb_049 bitsandbytes==0.49.0

GPU_ID=0 EPOCHS=1 bash scripts/kforge_rebuttal_matched_relearning.sh
GPU_ID=0 EPOCHS=3 bash scripts/kforge_rebuttal_matched_relearning.sh
GPU_ID=0 METHOD=SimNPO EPOCHS=1 bash scripts/kforge_rebuttal_matched_relearning.sh
GPU_ID=0 EPOCHS=1 bash scripts/kforge_rebuttal_gemma_relearning.sh
GPU_ID=0 EPOCHS=3 bash scripts/kforge_rebuttal_gemma_relearning.sh
GPU_ID=0 EPOCHS=1 bash scripts/kforge_rebuttal_gemma_simnpo_relearning.sh
GPU_ID=0 EPOCHS=3 bash scripts/kforge_rebuttal_gemma_simnpo_relearning.sh
GPU_ID=0 bash scripts/kforge_rebuttal_matched_quantization.sh
GPU_ID=0 bash scripts/kforge_rebuttal_simnpo_quantization.sh
GPU_ID=0 bash scripts/kforge_rebuttal_gemma_quantization.sh
```

Each runner writes a TSV manifest, skips only a valid existing summary, and
uses fixed seeds and identical attack settings for the paired arms. The
aggregator exits nonzero if a required summary or metric is missing.

## Current Figures

Regenerate the current paper figures with:

```bash
cd open-unlearning
python scripts/kforge_make_corrected_figures.py
```

Outputs:

```text
open-unlearning/saves/figures/kforge_corrected/
├── fig1_wiener_v2_strength_sweep.{png,pdf}
├── fig2_corrected_steps_to_target.{png,pdf}
├── fig3_corrected_pareto_forget10.{png,pdf}
├── figA1_spectrum_heatmap.{png,pdf}
└── figA2_init_controls_forget10.{png,pdf}
```

## Commit Hygiene

The root `.gitignore` excludes large generated artifacts:

- `open-unlearning/saves/` except curated corrected figures,
- `open-unlearning/logs/`,
- `open-unlearning/outputs/`,
- `open-unlearning/runs/`,
- `open-unlearning/.cache/`,
- model checkpoint formats such as `.safetensors`, `.bin`, `.pt`, and `.ckpt`.

Before committing, check:

```bash
git status --short
git status --ignored --short
```

## License

The OpenUnlearning code under `open-unlearning/` retains its upstream MIT
license. See `open-unlearning/LICENSE`.
