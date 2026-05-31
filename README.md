# K-FORGE

K-FORGE is a Kronecker-Fisher Wiener initialization method for preference-based LLM unlearning. This anonymous release contains the modified OpenUnlearning code, K-FORGE configs, experiment scripts, and curated plotting data needed to reproduce the figures used for the submitted paper.

The paper source and internal development notes are intentionally kept off `main`; they are preserved on the `dev` branch.

## Repository Layout

```text
.
├── README.md
└── open-unlearning/
    ├── src/trainer/unlearn/kforge.py          # K-FORGE trainer implementation
    ├── configs/trainer/KFORGE.yaml            # K-FORGE trainer config
    ├── configs/experiment/unlearn/tofu/kforge.yaml
    ├── docs/kforge.md                         # method/config documentation
    ├── scripts/kforge_make_corrected_figures.py
    ├── scripts/kforge_week2_init_experiment.sh
    ├── scripts/make_matched_init_checkpoint.py
    └── saves/figures/kforge_corrected/        # curated figures and plotted CSVs
```

Generated checkpoints, raw evals, logs, and local caches are ignored by git. The only files retained under `open-unlearning/saves/` are curated paper figures and the CSVs used to draw them.

## Environment

The code is based on OpenUnlearning. From `open-unlearning/`:

```bash
conda create -n unlearning python=3.11
conda activate unlearning
pip install ".[lm-eval]"
pip install --no-build-isolation flash-attn==2.6.3
python setup_data.py --eval
```

Experiments use the OpenUnlearning TOFU model releases, including Llama-3.2-1B-Instruct and Llama-3.2-3B-Instruct checkpoints.

## K-FORGE Entry Points

Core implementation:

```text
open-unlearning/src/trainer/unlearn/kforge.py
```

Main configs:

```text
open-unlearning/configs/trainer/KFORGE.yaml
open-unlearning/configs/experiment/unlearn/tofu/kforge.yaml
```

Method documentation:

```text
open-unlearning/docs/kforge.md
```

## Reproducing Curated Figures

From `open-unlearning/`:

```bash
python scripts/kforge_make_corrected_figures.py
```

Curated outputs are written to:

```text
open-unlearning/saves/figures/kforge_corrected/
```

Key retained artifacts include:

```text
fig2_improved.{png,pdf}
fig_matched_init_arrows.{png,pdf}
fig_one_shot_frontier.{png,pdf}
fig_pareto_frontier.{png,pdf}
corrected_aggregate_used.csv
corrected_runs_used.csv
init_controls_aggregate_used.csv
init_controls_runs_used.csv
kforge_compute_overhead_data.csv
```

## Running Initialization Experiments

The main initialization harness is:

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

Matched-norm initialization controls are generated with:

```text
open-unlearning/scripts/make_matched_init_checkpoint.py
open-unlearning/scripts/run_init_control_random_npo.sh
open-unlearning/scripts/run_init_control_random_simnpo.sh
open-unlearning/scripts/run_init_control_weight_svd_npo.sh
open-unlearning/scripts/run_init_control_weight_svd_simnpo.sh
```

## Git Hygiene

Before committing or publishing, check:

```bash
git status --short
git status --ignored --short
```

The root `.gitignore` excludes local/editor state, Python caches, build products, OpenUnlearning raw outputs, model checkpoints, logs, and LaTeX build products.

## License

The OpenUnlearning code under `open-unlearning/` retains its upstream MIT license. See `open-unlearning/LICENSE`.
