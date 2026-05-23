# K-FORGE

K-FORGE is a Kronecker-Fisher Wiener initialization method for LLM unlearning.
This repository contains the paper draft, experiment notes, and the modified
OpenUnlearning code used for the TOFU experiments.

The current paper story is:

- one-shot K-FORGE gives a controllable retain-forget frontier, but is not the
  final unlearning method;
- K-FORGE is used as an initializer for NPO and SimNPO;
- matched controls test whether the gain is explained by a generic low-rank
  perturbation, a weight-SVD direction, diagonal Fisher, or forget-only Fisher.

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
    ├── scripts/kforge_make_corrected_figures.py
    ├── scripts/kforge_week2_init_experiment.sh
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

Local experiments used the OpenUnlearning TOFU model releases, including
Llama-3.2-1B-Instruct and Llama-3.2-3B-Instruct checkpoints.

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
