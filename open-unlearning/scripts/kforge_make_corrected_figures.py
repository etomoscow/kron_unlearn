
#!/usr/bin/env python3
"""Regenerate K-FORGE figures from corrected Wiener-v2 experiment outputs."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap


SUMMARY = "TOFU_SUMMARY.json"
METRICS = ("model_utility", "forget_Q_A_Prob", "forget_Q_A_ROUGE", "extraction_strength")
RUN_RE = re.compile(
    r"tofu_Llama-3\.2-1B-Instruct_"
    r"(?P<forget>forget(?:10|05|01))_"
    r"(?P<algo>NPO|SimNPO)_"
    r"(?P<init>scratch|kforge_s045|kforge_s06)_"
    r"S(?P<steps>\d+)_seed(?P<seed>\d+)_v2lam0p01_corr_EVAL_FP32"
)
CTRL_RE = re.compile(
    r"tofu_Llama-3\.2-1B-Instruct_"
    r"(?P<forget>forget(?:10|05|01))_"
    r"(?P<algo>NPO|SimNPO)_"
    r"(?P<init>random_rank2|weight_svd|diagonal|forget_only)_"
    r"S(?P<steps>\d+)_seed(?P<seed>\d+)_initctrl_EVAL_FP32"
)
ONESHOT_RE = re.compile(
    r"KFORGE_TOFU_forget10_R2_M1_B32_"
    r"S(?P<tag>[0-9]+p[0-9]+)_kron_retain_cfix_retune_v2_lam0p01_EVAL_FP32"
)

PREFERENCE_CMAP = LinearSegmentedColormap.from_list(
    "forget_utility_preference",
    ["#BBD7EC", "#F6EAD2", "#E98632"],
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_summary(path: Path) -> dict[str, float] | None:
    summary = path / SUMMARY if path.is_dir() else path
    if not summary.exists():
        return None
    with summary.open() as handle:
        data = json.load(handle)
    return {metric: float(data[metric]) for metric in METRICS if metric in data}


def strength_from_tag(tag: str) -> float:
    return float(tag.replace("p", "."))


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def collect_runs(eval_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    runs: list[dict[str, object]] = []
    controls: list[dict[str, object]] = []
    oneshot: list[dict[str, object]] = []
    for path in sorted(eval_root.iterdir()):
        if not path.is_dir():
            continue
        summary = read_summary(path)
        if summary is None:
            continue
        if match := RUN_RE.fullmatch(path.name):
            rows = runs
        elif match := CTRL_RE.fullmatch(path.name):
            rows = controls
        elif match := ONESHOT_RE.fullmatch(path.name):
            row = {
                "method": "K-FORGE v2",
                "strength": strength_from_tag(match.group("tag")),
                "source": path.name,
                **summary,
            }
            oneshot.append(row)
            continue
        else:
            continue
        row = match.groupdict()
        row.update(summary)
        row["steps"] = int(row["steps"])
        row["seed"] = int(row["seed"])
        row["source"] = path.name
        rows.append(row)
    return pd.DataFrame(runs), pd.DataFrame(controls), pd.DataFrame(oneshot)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    group_cols = ["forget", "algo", "init", "steps"]
    out = (
        df.groupby(group_cols, as_index=False)
        .agg(
            model_utility_mean=("model_utility", "mean"),
            model_utility_std=("model_utility", "std"),
            forget_Q_A_Prob_mean=("forget_Q_A_Prob", "mean"),
            forget_Q_A_Prob_std=("forget_Q_A_Prob", "std"),
            forget_Q_A_ROUGE_mean=("forget_Q_A_ROUGE", "mean"),
            extraction_strength_mean=("extraction_strength", "mean"),
            seeds=("seed", "nunique"),
        )
        .sort_values(group_cols)
    )
    return out.fillna(0.0)


def add_reference_rows(eval_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    refs = {
        "Base model": eval_root / "tofu_Llama-3.2-1B-Instruct_full" / "evals_forget10",
        "Gold retrain": eval_root / "tofu_Llama-3.2-1B-Instruct_retain90",
    }
    for label, path in refs.items():
        summary = read_summary(path)
        if summary:
            rows.append({"method": label, "source": str(path.relative_to(eval_root)), **summary})
    return pd.DataFrame(rows)


def plot_figure1(oneshot: pd.DataFrame, refs: pd.DataFrame, out_dir: Path) -> None:
    data = oneshot.sort_values("strength").copy()
    data.to_csv(out_dir / "fig1_wiener_v2_strength_sweep_data.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.9), sharex=True)
    axes[0].plot(data["strength"], data["model_utility"], marker="o", color="#1f77b4", label="K-FORGE v2")
    axes[1].plot(data["strength"], data["forget_Q_A_Prob"], marker="o", color="#d62728", label="K-FORGE v2")
    for _, ref in refs.iterrows():
        color = "#555555" if ref["method"] == "Base model" else "#2ca02c"
        axes[0].axhline(ref["model_utility"], color=color, linewidth=1.0, linestyle="--", alpha=0.65)
        axes[1].axhline(ref["forget_Q_A_Prob"], color=color, linewidth=1.0, linestyle="--", alpha=0.65)
        axes[0].text(data["strength"].max() * 1.04, ref["model_utility"], ref["method"], va="center", fontsize=6.2)
    axes[0].set_ylabel("Model Utility")
    axes[1].set_ylabel("Forget Probability")
    axes[1].set_xlabel(r"Strength $\alpha$")
    axes[1].set_xscale("log")
    axes[1].set_ylim(0.0, 0.92)
    for ax in axes:
        ax.grid(True, alpha=0.24, linewidth=0.6)
    axes[0].set_title("Corrected Wiener-v2 one-shot strength sweep")
    save_figure(fig, out_dir, "fig1_wiener_v2_strength_sweep")


def interp_reach_step(curve: pd.DataFrame, target: float) -> float | None:
    """Return first step where decreasing forget probability reaches target."""
    c = curve.sort_values("steps")
    steps = c["steps"].to_numpy(dtype=float)
    vals = c["forget_Q_A_Prob_mean"].to_numpy(dtype=float)
    for step, val in zip(steps, vals):
        if val <= target:
            return float(step)
    for i in range(1, len(steps)):
        y0, y1 = vals[i - 1], vals[i]
        if (y0 - target) * (y1 - target) <= 0 and y0 != y1:
            t = (target - y0) / (y1 - y0)
            return float(steps[i - 1] + t * (steps[i] - steps[i - 1]))
    return None




def plot_figure2(agg: pd.DataFrame, out_dir: Path) -> None:
    data = agg[agg["init"].isin(["scratch", "kforge_s045"])].copy()

    COLORS   = {"scratch": "#666666", "kforge_s045": "#CC79A7"}
    FORGETS  = ["forget10", "forget05", "forget01"]
    # Two-line title: dataset name (top, farther) then readable label (bottom, closer to axes)
    COL_TITLES = {
        "forget10": r"$\mathtt{forget10}$",
        "forget05": r"$\mathtt{forget05}$",
        "forget01": r"$\mathtt{forget01}$",
    }
    ALGOS    = ["NPO", "SimNPO"]

    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.5), sharex=True)

    legend_ax = None   # will place legend in panel with most vertical room

    for r, algo in enumerate(ALGOS):
        for c, forget in enumerate(FORGETS):
            ax = axes[r, c]
            panel = data[(data["algo"] == algo) & (data["forget"] == forget)]
            kf = panel[panel["init"] == "kforge_s045"].sort_values("steps")
            sc = panel[panel["init"] == "scratch"].sort_values("steps")

            # ── shaded advantage region ───────────────────────────────────
            if len(kf) and len(sc):
                common = np.intersect1d(kf["steps"].values, sc["steps"].values)
                y_kf = kf.set_index("steps").loc[common]["forget_Q_A_Prob_mean"].values
                y_sc = sc.set_index("steps").loc[common]["forget_Q_A_Prob_mean"].values
                ax.fill_between(
                    common, y_kf, y_sc,
                    where=(y_kf <= y_sc),
                    color=COLORS["kforge_s045"], alpha=0.13, zorder=1,
                )

            # ── scratch: dashed, open markers ────────────────────────────
            if len(sc):
                ax.errorbar(
                    sc["steps"], sc["forget_Q_A_Prob_mean"],
                    yerr=sc["forget_Q_A_Prob_std"],
                    marker="o", ms=3.8, lw=1.2, ls="--",
                    capsize=2.0, elinewidth=0.8,
                    color=COLORS["scratch"],
                    mfc="none", mew=1.2,
                    label="Scratch", zorder=3,
                )

            # ── K-FORGE: solid, filled markers ───────────────────────────
            if len(kf):
                ax.errorbar(
                    kf["steps"], kf["forget_Q_A_Prob_mean"],
                    yerr=kf["forget_Q_A_Prob_std"],
                    marker="o", ms=3.8, lw=1.5, ls="-",
                    capsize=2.0, elinewidth=0.8,
                    color=COLORS["kforge_s045"],
                    label="K-FORGE init", zorder=4,
                )

            # ── y limits ─────────────────────────────────────────────────
            y_lo = (panel["forget_Q_A_Prob_mean"] - panel["forget_Q_A_Prob_std"]).min()
            y_hi = (panel["forget_Q_A_Prob_mean"] + panel["forget_Q_A_Prob_std"]).max()
            if np.isfinite(y_lo) and np.isfinite(y_hi):
                pad = max((y_hi - y_lo) * 0.18, y_hi * 0.05)
                ax.set_ylim(max(0.0, y_lo - pad), y_hi + pad)

            # ── 1-D vertical gradient (y-axis only: cool=high forget, warm=low) ──
            # x-axis is just a parameter (steps), not an objective, so no x-gradient.
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            grad = np.linspace(1, 0, 256).reshape(-1, 1)   # top→bottom: bad→good
            ax.imshow(
                grad,
                extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                aspect="auto", origin="lower",
                cmap=PREFERENCE_CMAP,
                alpha=0.20,          # subtle: fill_between carries the main story
                zorder=-10,
            )

            # ── column title: dataset name farther, readable label closer ──
            if r == 0:
                ax.set_title(COL_TITLES[forget], pad=2, fontsize=8.0,
                             linespacing=1.)

            # ── gradient description (leftmost column only) ───────────────
            if c == 0:
                ax.text(0.03, 0.98, "less desired", transform=ax.transAxes,
                        ha="left", va="top", fontsize=6, color="#456C87",
                        alpha=0.82, style="normal", zorder=5)
                ax.text(0.03, 0.03, "more desired", transform=ax.transAxes,
                        ha="left", va="bottom", fontsize=6, color="#9A4E17",
                        alpha=0.88, style="normal", zorder=5)
                ax.text(
                    -0.22, 0.5, algo,
                    transform=ax.transAxes,
                    rotation=90, ha="center", va="center",
                    fontsize=8.5, fontweight="semibold",
                )

            ax.set_xticks([50, 100, 250])
            ax.grid(True, axis="y", color="#D0D0D0", linewidth=0.5, alpha=0.8)
            ax.set_axisbelow(True)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            for spine in ["left", "bottom"]:
                ax.spines[spine].set_linewidth(0.75)
                ax.spines[spine].set_color("#333333")

            if r == 1:
                ax.set_xlabel("Training steps", fontsize=7.5)

            # pick legend panel: NPO × forget10 has most vertical space
            if r == 0 and c == 0:
                legend_ax = ax

    # ── shared y-axis label ───────────────────────────────────────────────
    fig.text(
        0.042, 0.5, "Forget Q/A Probability ↓",
        rotation=90, va="center", fontsize=8,
    )

    # ── legend inside upper-right of NPO/forget10 panel ──────────────────
    if legend_ax is not None:
        handles, labels = legend_ax.get_legend_handles_labels()
        legend_ax.legend(
            handles, labels,
            frameon=True, framealpha=0.88, edgecolor="#cccccc",
            fontsize=7.0, loc="upper right",
            handlelength=1.6, borderpad=0.5,
        )

    fig.tight_layout(rect=(0.055, 0.0, 1.0, 1.0), w_pad=0.4, h_pad=0.4)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig2_improved.pdf")
    fig.savefig(out_dir / "fig2_improved.png", dpi=300)
    plt.close(fig)

def pareto_frontier(points: pd.DataFrame) -> pd.DataFrame:
    cur = points.sort_values(["model_utility", "forget_Q_A_Prob"], ascending=[False, True])
    frontier = []
    best_forget = math.inf
    for _, row in cur.iterrows():
        if row["forget_Q_A_Prob"] < best_forget:
            frontier.append(row)
            best_forget = row["forget_Q_A_Prob"]
    return pd.DataFrame(frontier).sort_values("model_utility")


def plot_figure3(
    agg: pd.DataFrame,
    controls: pd.DataFrame,
    oneshot: pd.DataFrame,
    refs: pd.DataFrame,
    out_dir: Path,
) -> None:
    rows: list[dict[str, object]] = []
    for _, row in refs.iterrows():
        rows.append({"method": row["method"], "family": "reference", **{m: row[m] for m in METRICS}})
    for _, row in oneshot.iterrows():
        rows.append({"method": f"K-FORGE one-shot α={row['strength']:g}", "family": "one-shot", **{m: row[m] for m in METRICS}})
    main = agg[(agg["forget"] == "forget10") & (agg["steps"].isin([50, 100, 250]))]
    for _, row in main.iterrows():
        method = f"{row['algo']} {row['init']} S{int(row['steps'])}"
        rows.append(
            {
                "method": method,
                "family": row["init"],
                "model_utility": row["model_utility_mean"],
                "forget_Q_A_Prob": row["forget_Q_A_Prob_mean"],
                "forget_Q_A_ROUGE": row["forget_Q_A_ROUGE_mean"],
                "extraction_strength": row["extraction_strength_mean"],
            }
        )
    ctrl = controls[(controls["forget"] == "forget10") & (controls["steps"].isin([50, 100]))]
    for _, row in ctrl.iterrows():
        method = f"{row['algo']} {row['init']} S{int(row['steps'])}"
        rows.append(
            {
                "method": method,
                "family": row["init"],
                "model_utility": row["model_utility_mean"],
                "forget_Q_A_Prob": row["forget_Q_A_Prob_mean"],
                "forget_Q_A_ROUGE": row["forget_Q_A_ROUGE_mean"],
                "extraction_strength": row["extraction_strength_mean"],
            }
        )
    data = pd.DataFrame(rows)
    data.to_csv(out_dir / "fig3_corrected_pareto_forget10_data.csv", index=False)
    plot_data = data[(data["family"] != "one-shot") | (data["model_utility"] >= 0.45)].copy()
    excluded = data.loc[~data.index.isin(plot_data.index)].copy()
    plot_data.to_csv(out_dir / "fig3_corrected_pareto_forget10_plot_data.csv", index=False)
    if not excluded.empty:
        excluded.to_csv(out_dir / "fig3_corrected_pareto_forget10_excluded_low_utility_data.csv", index=False)

    colors = {
        "reference": "#222222",
        "one-shot": "#cc79a7",
        "scratch": "#777777",
        "kforge_s045": "#0072b2",
        "kforge_s06": "#56b4e9",
        "random_rank2": "#e69f00",
        "weight_svd": "#009e73",
        "diagonal": "#d55e00",
        "forget_only": "#9467bd",
    }
    markers = {
        "reference": "*",
        "one-shot": "D",
        "scratch": "o",
        "kforge_s045": "o",
        "kforge_s06": "o",
        "random_rank2": "s",
        "weight_svd": "^",
        "diagonal": "v",
        "forget_only": "P",
    }
    fig, ax = plt.subplots(figsize=(4.2, 3.35))
    for family, cur in plot_data.groupby("family"):
        ax.scatter(
            cur["model_utility"],
            cur["forget_Q_A_Prob"],
            s=38 if family != "reference" else 70,
            color=colors.get(family, "#333333"),
            marker=markers.get(family, "o"),
            alpha=0.9,
            label=family.replace("_", " "),
        )
    frontier = pareto_frontier(plot_data)
    ax.plot(frontier["model_utility"], frontier["forget_Q_A_Prob"], color="#111111", linewidth=1.0, alpha=0.45)
    if not excluded.empty:
        ax.annotate(
            f"{len(excluded)} one-shot failures excluded\nfrom visible operating region",
            xy=(0.03, 0.05),
            xycoords="axes fraction",
            fontsize=6.2,
            color="#555555",
        )
    ax.set_xlabel("Model Utility")
    ax.set_ylabel("Forget Probability")
    ax.set_title("TOFU forget10 corrected Pareto frontier")
    ax.set_xlim(0.49, 0.607)
    ax.set_ylim(0.0, 0.92)
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.legend(frameon=False, ncol=2, fontsize=6.1)
    save_figure(fig, out_dir, "fig3_corrected_pareto_forget10")


def plot_controls(controls: pd.DataFrame, agg: pd.DataFrame, out_dir: Path) -> None:
    data = pd.concat(
        [
            controls[controls["forget"] == "forget10"],
            agg[(agg["forget"] == "forget10") & (agg["init"].isin(["scratch", "kforge_s045"]))],
        ],
        ignore_index=True,
    )
    data = data[data["steps"].isin([50, 100])].copy()
    data.to_csv(out_dir / "figA2_init_controls_forget10_data.csv", index=False)
    order = ["scratch", "random_rank2", "weight_svd", "diagonal", "forget_only", "kforge_s045"]
    labels = ["scratch", "random", "weight-SVD", "diagonal", "forget-only", "K-FORGE"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, algo in zip(axes, ["NPO", "SimNPO"]):
        panel = data[data["algo"] == algo]
        for step, marker in [(50, "o"), (100, "s")]:
            cur = panel[panel["steps"] == step].set_index("init").reindex(order)
            ax.errorbar(
                np.arange(len(order)) + (-0.08 if step == 50 else 0.08),
                cur["forget_Q_A_Prob_mean"],
                yerr=cur["forget_Q_A_Prob_std"],
                fmt=marker,
                capsize=2,
                label=f"S{step}",
            )
        ax.set_title(algo)
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(True, axis="y", alpha=0.22, linewidth=0.6)
        ax.set_ylabel("Forget Probability")
        ax.legend(frameon=False)
    save_figure(fig, out_dir, "figA2_init_controls_forget10")


def module_short_name(module: str) -> str:
    if module.endswith("q_proj"):
        return "q"
    if module.endswith("k_proj"):
        return "k"
    if module.endswith("v_proj"):
        return "v"
    if module.endswith("o_proj"):
        return "o"
    if module.endswith("gate_proj"):
        return "gate"
    if module.endswith("up_proj"):
        return "up"
    if module.endswith("down_proj"):
        return "down"
    return module.rsplit(".", 1)[-1]


def plot_spectrum(root: Path, out_dir: Path) -> None:
    path = root / "saves/spectrum/kforge_spectrum_summary.csv"
    if not path.exists():
        return
    spec = pd.read_csv(path)
    spec["module_short"] = spec["module"].map(module_short_name)
    order = ["q", "k", "v", "o", "gate", "up", "down"]
    pivot = (
        spec.pivot_table(index="layer", columns="module_short", values="top1", aggfunc="max")
        .sort_index()
        .reindex(columns=order)
    )
    pivot.to_csv(out_dir / "figA1_spectrum_heatmap_data.csv")
    values = np.log10(np.clip(pivot.to_numpy(dtype=float), 1e-12, None))
    fig, ax = plt.subplots(figsize=(4.1, 4.7))
    im = ax.imshow(values, aspect="auto", cmap="magma")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(int(x)) for x in pivot.index])
    ax.set_xlabel("Module")
    ax.set_ylabel("Layer")
    ax.set_title(r"Per-layer $\sigma_f/\sigma_r$ spectrum")
    if 15 in pivot.index:
        ax.add_patch(Rectangle((order.index("down") - 0.5, list(pivot.index).index(15) - 0.5), 1, 1, fill=False, edgecolor="#00d5ff", linewidth=1.8))
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label(r"$\log_{10}$ top $\sigma_f/\sigma_r$")
    save_figure(fig, out_dir, "figA1_spectrum_heatmap")


def write_manifest(out_dir: Path, runs: pd.DataFrame, controls: pd.DataFrame, oneshot: pd.DataFrame) -> None:
    manifest = [
        "# Corrected K-FORGE Figures",
        "",
        "Generated from corrected `wiener_v2` one-shot runs, `v2lam0p01_corr` downstream runs, and `initctrl` initializer controls.",
        "",
        f"- Corrected downstream seed runs parsed: {len(runs)}",
        f"- Init-control seed runs parsed: {len(controls)}",
        f"- Corrected one-shot strength points parsed: {len(oneshot)}",
        "",
        "Figures:",
        "- `fig1_wiener_v2_strength_sweep.{png,pdf}`",
        "- `fig2_corrected_steps_to_target.{png,pdf}`",
        "- `fig3_corrected_pareto_forget10.{png,pdf}`",
        "- `figA1_spectrum_heatmap.{png,pdf}`",
        "- `figA2_init_controls_forget10.{png,pdf}`",
    ]
    (out_dir / "MANIFEST.md").write_text("\n".join(manifest) + "\n")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    eval_root = root / "saves/eval"
    out_dir = (args.out_dir or root / "saves/figures/kforge_corrected").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    runs, controls_raw, oneshot = collect_runs(eval_root)
    agg = aggregate(runs)
    controls = aggregate(controls_raw)
    refs = add_reference_rows(eval_root)
    runs.to_csv(out_dir / "corrected_runs_used.csv", index=False)
    agg.to_csv(out_dir / "corrected_aggregate_used.csv", index=False)
    controls_raw.to_csv(out_dir / "init_controls_runs_used.csv", index=False)
    controls.to_csv(out_dir / "init_controls_aggregate_used.csv", index=False)
    refs.to_csv(out_dir / "reference_points_used.csv", index=False)

    plot_figure1(oneshot, refs, out_dir)
    plot_figure2(agg, out_dir)
    plot_figure3(agg, controls, oneshot, refs, out_dir)
    plot_controls(controls, agg, out_dir)
    plot_spectrum(root, out_dir)
    write_manifest(out_dir, runs, controls_raw, oneshot)
    print(f"Wrote corrected figures to {out_dir}")
    print(f"Parsed {len(runs)} corrected downstream runs, {len(controls_raw)} controls, {len(oneshot)} one-shot points.")


if __name__ == "__main__":
    main()
