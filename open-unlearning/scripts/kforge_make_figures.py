#!/usr/bin/env python3
"""Build K-FORGE paper figures from saved experiment summaries."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


WEEK2_PATTERN = re.compile(
    r"tofu_(?P<model>.+)_(?P<forget>forget\d+)_(?P<trainer>NPO|SimNPO|RMU)"
    r"_(?P<init>scratch|kforge)_S(?P<steps>\d+)_seed(?P<seed>\d+)_week2_EVAL_FP32"
)

METRICS = [
    "model_utility",
    "forget_Q_A_Prob",
    "forget_Q_A_ROUGE",
    "extraction_strength",
    "privleak",
]

SCRATCH_COLOR = "#2f6fbb"
KFORGE_COLOR = "#d66b00"
GRAY = "#666666"


@dataclass(frozen=True)
class Paths:
    root: Path
    eval_root: Path
    kforge_csv: Path
    spectrum_csv: Path
    out_dir: Path


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def read_summary(path: Path) -> dict[str, float | str | None]:
    data = json.loads(path.read_text())
    return {metric: data.get(metric) for metric in METRICS}


def load_week2_runs(eval_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(eval_root.glob("*week2_EVAL_FP32/TOFU_SUMMARY.json")):
        match = WEEK2_PATTERN.fullmatch(path.parent.name)
        if not match:
            continue
        row: dict[str, object] = match.groupdict()
        row["steps"] = int(row["steps"])
        row["seed"] = int(row["seed"])
        row.update(read_summary(path))
        row["summary_path"] = str(path)
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["forget", "trainer", "steps", "init", "seed"] + METRICS)
    return pd.DataFrame(rows)


def aggregate_runs(runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame()
    return (
        runs.groupby(["forget", "trainer", "steps", "init"], as_index=False)
        .agg(
            n=("seed", "count"),
            **{
                f"{metric}_{stat}": (metric, stat)
                for metric in METRICS
                for stat in ["mean", "std"]
            },
        )
        .sort_values(["forget", "trainer", "steps", "init"])
    )


def line_from_aggregate(
    aggregate: pd.DataFrame, forget: str, trainer: str, init: str
) -> pd.DataFrame:
    sub = aggregate[
        (aggregate["forget"] == forget)
        & (aggregate["trainer"] == trainer)
        & (aggregate["init"] == init)
    ].copy()
    return sub.sort_values("steps")


def acceleration_label(panel: pd.DataFrame) -> str:
    scratch = panel[panel["init"] == "scratch"].sort_values("steps")
    kforge = panel[panel["init"] == "kforge"].sort_values("steps")
    if scratch.empty or kforge.empty:
        return "k n/a"

    best_ratio = -math.inf
    best_label = "k n/a"
    for _, k_row in kforge.iterrows():
        k_step = int(k_row["steps"])
        k_prob = float(k_row["forget_Q_A_Prob_mean"])
        hits = scratch[scratch["forget_Q_A_Prob_mean"] <= k_prob]
        if hits.empty:
            final_step = int(scratch["steps"].max())
            final_prob = float(
                scratch.loc[scratch["steps"] == final_step, "forget_Q_A_Prob_mean"].iloc[0]
            )
            if k_prob >= final_prob:
                continue
            ratio = final_step / k_step
            prefix = ">"
            scratch_step_label = f">S{final_step}"
        else:
            scratch_step = int(hits.sort_values("steps")["steps"].iloc[0])
            ratio = scratch_step / k_step
            prefix = "="
            scratch_step_label = f"S{scratch_step}"
        if ratio > best_ratio:
            best_ratio = ratio
            best_label = f"k{prefix}{ratio:.1f}x\n{scratch_step_label}->S{k_step}"
    return best_label


def finite_positive(values: pd.Series) -> pd.Series:
    return values[np.isfinite(values) & (values > 0)]


def plot_figure1(kforge: pd.DataFrame, paths: Paths) -> None:
    wanted_batches = [2, 4, 8, 16, 32, 64]
    cal = kforge[
        (kforge["forget"] == "forget10")
        & (kforge["rank"] == 2)
        & (kforge["modules"] == 1)
        & (kforge["mode"] == "kron")
        & (kforge["retain"] == "retain")
        & (kforge["suffix"] == "_bcal")
        & (kforge["batches"].isin(wanted_batches))
    ].copy()
    cal = cal.sort_values(["batches", "strength_float"])
    cal.to_csv(paths.out_dir / "fig1_strength_cliff_calibration_data.csv", index=False)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(3.45, 3.85),
        sharex=True,
        gridspec_kw={"hspace": 0.08, "height_ratios": [1, 1]},
    )
    cmap = plt.get_cmap("viridis")
    for idx, batches in enumerate(wanted_batches):
        cur = cal[cal["batches"] == batches]
        if cur.empty:
            continue
        x = cur["strength_float"] * 1000.0
        color = cmap(idx / max(len(wanted_batches) - 1, 1))
        label = f"B={batches}"
        axes[0].plot(
            x,
            cur["model_utility"],
            marker="o",
            markersize=3.2,
            linewidth=1.5,
            color=color,
            label=label,
        )
        axes[1].plot(
            x,
            cur["forget_Q_A_Prob"],
            marker="o",
            markersize=3.2,
            linewidth=1.5,
            color=color,
        )

    axes[0].set_ylabel("Model utility")
    axes[1].set_ylabel("Forget probability")
    axes[1].set_xlabel(r"K-FORGE strength $\alpha$ ($\times 10^{-3}$)")
    axes[0].set_title("Calibration scale controls the one-shot cliff")
    xticks = sorted(cal["strength_float"].mul(1000.0).unique())
    axes[1].set_xticks(xticks)
    axes[1].set_xticklabels([f"{x:g}" for x in xticks])
    for ax in axes:
        ax.grid(True, alpha=0.22, linewidth=0.6)
    axes[0].legend(title=r"$B_{cal}$", ncol=3, frameon=False, loc="lower left")
    save_figure(fig, paths.out_dir, "fig1_strength_cliff_calibration")


def plot_figure2(aggregate: pd.DataFrame, paths: Paths) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.65), sharey=True)
    trainers = ["NPO", "SimNPO"]
    forgets = ["forget10", "forget05", "forget01"]
    panel_data = aggregate[
        aggregate["trainer"].isin(trainers) & aggregate["forget"].isin(forgets)
    ].copy()
    panel_data.to_csv(paths.out_dir / "fig2_steps_to_target_data.csv", index=False)

    positives = finite_positive(panel_data["forget_Q_A_Prob_mean"])
    ymin = max(float(positives.min()) * 0.65, 1e-4) if not positives.empty else 1e-3

    for row, trainer in enumerate(trainers):
        for col, forget in enumerate(forgets):
            ax = axes[row, col]
            panel = panel_data[
                (panel_data["trainer"] == trainer) & (panel_data["forget"] == forget)
            ].copy()
            if panel.empty:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue

            curves: dict[str, pd.DataFrame] = {}
            for init, color, label in [
                ("scratch", SCRATCH_COLOR, "scratch"),
                ("kforge", KFORGE_COLOR, "K-FORGE init"),
            ]:
                cur = panel[panel["init"] == init].sort_values("steps")
                curves[init] = cur
                if cur.empty:
                    continue
                ax.errorbar(
                    cur["steps"],
                    cur["forget_Q_A_Prob_mean"],
                    yerr=cur["forget_Q_A_Prob_std"].fillna(0),
                    marker="o",
                    markersize=3.2,
                    linewidth=1.5,
                    capsize=2,
                    color=color,
                    label=label,
                    zorder=3,
                )

            if not curves.get("scratch", pd.DataFrame()).empty and not curves.get(
                "kforge", pd.DataFrame()
            ).empty:
                scratch = curves["scratch"].set_index("steps")
                kforge = curves["kforge"].set_index("steps")
                common = sorted(set(scratch.index).intersection(kforge.index))
                if common:
                    ax.fill_between(
                        common,
                        scratch.loc[common, "forget_Q_A_Prob_mean"].to_numpy(dtype=float),
                        kforge.loc[common, "forget_Q_A_Prob_mean"].to_numpy(dtype=float),
                        color=GRAY,
                        alpha=0.14,
                        linewidth=0,
                        zorder=1,
                    )

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(ymin, 1.0)
            ax.grid(True, alpha=0.22, linewidth=0.6, which="both")
            ax.set_title(f"{trainer} / {forget}")
            ax.text(
                0.04,
                0.08,
                acceleration_label(panel),
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=7,
                bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
            )

            if panel["n"].min() < 3:
                ax.text(
                    0.98,
                    0.08,
                    "partial n",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=6.5,
                    color="#8a4b00",
                )
            if row == len(trainers) - 1:
                ax.set_xlabel("Training steps")
            else:
                ax.tick_params(labelbottom=False)
            if col == 0:
                ax.set_ylabel("Forget probability")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.95))
    fig.suptitle("K-FORGE initialization accelerates forgetting", y=1.0, fontsize=10)
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.10, top=0.82, hspace=0.36, wspace=0.25)
    save_figure(fig, paths.out_dir, "fig2_steps_to_target_acceleration")


def add_json_point(
    rows: list[dict[str, object]],
    path: Path,
    method: str,
    init: str,
    label: str,
    steps: int | None = None,
) -> None:
    if not path.exists():
        return
    data = read_summary(path)
    rows.append(
        {
            "method": method,
            "init": init,
            "label": label,
            "steps": steps,
            "model_utility": data["model_utility"],
            "forget_Q_A_Prob": data["forget_Q_A_Prob"],
            "source": str(path),
        }
    )


def pareto_frontier(points: pd.DataFrame) -> pd.DataFrame:
    sub = points[["model_utility", "forget_Q_A_Prob"]].dropna().copy()
    rows = []
    for idx, row in sub.iterrows():
        utility = float(row["model_utility"])
        prob = float(row["forget_Q_A_Prob"])
        dominated = (
            (sub["model_utility"] >= utility)
            & (sub["forget_Q_A_Prob"] <= prob)
            & ((sub["model_utility"] > utility) | (sub["forget_Q_A_Prob"] < prob))
        ).any()
        if not dominated:
            rows.append(idx)
    return sub.loc[rows].sort_values("model_utility")


def plot_figure3(kforge: pd.DataFrame, aggregate: pd.DataFrame, paths: Paths) -> None:
    rows: list[dict[str, object]] = []
    add_json_point(
        rows,
        paths.eval_root / "tofu_Llama-3.2-1B-Instruct_full/evals_forget10/TOFU_SUMMARY.json",
        "Base",
        "base",
        "base",
    )
    add_json_point(
        rows,
        paths.eval_root / "tofu_Llama-3.2-1B-Instruct_retain90/TOFU_SUMMARY.json",
        "Gold retrain",
        "retrain",
        "gold retrain",
    )
    for method in ["NPO", "SimNPO", "RMU", "GradDiff"]:
        add_json_point(
            rows,
            paths.eval_root
            / f"tofu_Llama-3.2-1B-Instruct_forget10_{method}_week1_EVAL_FP32/TOFU_SUMMARY.json",
            method,
            "week1",
            method,
        )

    week2 = aggregate[aggregate["forget"] == "forget10"].copy()
    for _, row in week2.iterrows():
        init = str(row["init"])
        method = str(row["trainer"])
        if init == "kforge":
            method = f"K-FORGE+{method}"
        rows.append(
            {
                "method": method,
                "init": init,
                "label": f"{method} S{int(row['steps'])}",
                "steps": int(row["steps"]),
                "model_utility": row["model_utility_mean"],
                "forget_Q_A_Prob": row["forget_Q_A_Prob_mean"],
                "source": "week2 aggregate",
            }
        )

    one_shot = kforge[
        (kforge["forget"] == "forget10")
        & (kforge["rank"] == 2)
        & (kforge["modules"] == 1)
        & (kforge["batches"] == 2)
        & (kforge["mode"] == "kron")
        & (kforge["retain"] == "retain")
        & (kforge["suffix"].isin(["", "_bcal", "_stage3down", "_down"]))
    ].copy()
    one_shot = one_shot.drop_duplicates(
        subset=["strength_float", "model_utility", "forget_Q_A_Prob"]
    )
    for _, row in one_shot.iterrows():
        rows.append(
            {
                "method": "K-FORGE one-shot",
                "init": "closed-form",
                "label": f"one-shot a={row['strength_float']:.4g}",
                "steps": None,
                "model_utility": row["model_utility"],
                "forget_Q_A_Prob": row["forget_Q_A_Prob"],
                "source": row["path"],
            }
        )

    points = pd.DataFrame(rows)
    points.to_csv(paths.out_dir / "fig3_pareto_frontier_data.csv", index=False)

    fig, ax = plt.subplots(figsize=(3.75, 3.35))
    styles = {
        "Base": ("*", "#111111", 92),
        "Gold retrain": ("D", "#2f8f46", 48),
        "GradDiff": ("X", "#707070", 46),
        "RMU": ("^", "#8a4fbf", 40),
        "NPO": ("o", SCRATCH_COLOR, 34),
        "SimNPO": ("s", SCRATCH_COLOR, 34),
        "K-FORGE+NPO": ("o", KFORGE_COLOR, 38),
        "K-FORGE+SimNPO": ("s", KFORGE_COLOR, 38),
        "K-FORGE+RMU": ("^", KFORGE_COLOR, 38),
        "K-FORGE one-shot": ("x", "#a85d00", 28),
    }
    order = [
        "Base",
        "Gold retrain",
        "GradDiff",
        "RMU",
        "NPO",
        "SimNPO",
        "K-FORGE one-shot",
        "K-FORGE+NPO",
        "K-FORGE+SimNPO",
        "K-FORGE+RMU",
    ]
    for method in order:
        cur = points[points["method"] == method]
        if cur.empty:
            continue
        marker, color, size = styles[method]
        ax.scatter(
            cur["model_utility"],
            cur["forget_Q_A_Prob"],
            marker=marker,
            s=size,
            color=color,
            alpha=0.78 if method != "K-FORGE one-shot" else 0.55,
            linewidths=0.8,
            label=method,
            zorder=3,
        )

    frontier = pareto_frontier(points)
    if len(frontier) >= 2:
        ax.plot(
            frontier["model_utility"],
            frontier["forget_Q_A_Prob"],
            color="#111111",
            linestyle="--",
            linewidth=1.0,
            alpha=0.32,
            label="Pareto envelope",
            zorder=2,
        )

    ax.set_xlabel("Model utility (higher is better)")
    ax.set_ylabel("Forget probability (lower is better)")
    ax.set_title("TOFU forget10 Pareto frontier")
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.set_xlim(0.30, 0.61)
    ax.set_ylim(-0.02, 0.92)
    ax.legend(frameon=False, fontsize=6.3, loc="upper left", ncol=1)
    save_figure(fig, paths.out_dir, "fig3_pareto_frontier_forget10")


def module_short_name(module: str) -> str:
    return {
        "self_attn_q_proj": "q",
        "self_attn_k_proj": "k",
        "self_attn_v_proj": "v",
        "self_attn_o_proj": "o",
        "mlp_gate_proj": "gate",
        "mlp_up_proj": "up",
        "mlp_down_proj": "down",
    }.get(module, module)


def plot_figure_a1(spectrum: pd.DataFrame, paths: Paths) -> None:
    order = ["q", "k", "v", "o", "gate", "up", "down"]
    spec = spectrum.copy()
    spec["module_short"] = spec["module"].map(module_short_name)
    pivot = (
        spec.pivot_table(index="layer", columns="module_short", values="top1", aggfunc="max")
        .sort_index()
        .reindex(columns=order)
    )
    pivot.to_csv(paths.out_dir / "figA1_spectrum_heatmap_data.csv")
    values = pivot.to_numpy(dtype=float)
    log_values = np.log10(np.clip(values, 1e-12, None))

    fig, ax = plt.subplots(figsize=(4.1, 4.7))
    im = ax.imshow(log_values, aspect="auto", cmap="magma")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(int(x)) for x in pivot.index])
    ax.set_xlabel("Module")
    ax.set_ylabel("Layer")
    ax.set_title(r"Per-layer $\sigma_f/\sigma_r$ spectrum")
    if 15 in pivot.index and "down" in order:
        y = list(pivot.index).index(15)
        x = order.index("down")
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, edgecolor="#00d5ff", linewidth=1.8))
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label(r"$\log_{10}$ top $\sigma_f/\sigma_r$")
    save_figure(fig, paths.out_dir, "figA1_spectrum_heatmap")


def choose_one_per_strength(df: pd.DataFrame, suffix_priority: list[str]) -> pd.DataFrame:
    priority = {suffix: idx for idx, suffix in enumerate(suffix_priority)}
    cur = df.copy()
    cur["_priority"] = cur["suffix"].map(lambda x: priority.get(str(x), len(priority)))
    cur = cur.sort_values(["strength_float", "_priority"])
    return cur.drop_duplicates(subset=["strength_float"], keep="first").drop(columns=["_priority"])


def ablation_sweep(
    kforge: pd.DataFrame,
    *,
    mode: str,
    retain: str,
    suffix_priority: list[str],
) -> pd.DataFrame:
    cur = kforge[
        (kforge["forget"] == "forget10")
        & (kforge["rank"] == 2)
        & (kforge["modules"] == 1)
        & (kforge["batches"] == 2)
        & (kforge["mode"] == mode)
        & (kforge["retain"] == retain)
    ].copy()
    cur = cur[cur["suffix"].isin(suffix_priority)]
    return choose_one_per_strength(cur, suffix_priority).sort_values("strength_float")


def plot_two_panel_ablation(
    rows: list[tuple[str, pd.DataFrame, str]],
    out_dir: Path,
    stem: str,
    title: str,
    data_name: str,
) -> None:
    combined = []
    for label, df, _ in rows:
        cur = df.copy()
        cur["series"] = label
        combined.append(cur)
    if combined:
        pd.concat(combined, ignore_index=True).to_csv(out_dir / data_name, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(6.25, 2.65))
    for label, df, color in rows:
        if df.empty:
            continue
        axes[0].plot(
            df["strength_float"],
            df["model_utility"],
            marker="o",
            markersize=3.4,
            linewidth=1.5,
            color=color,
            label=label,
        )
        axes[1].plot(
            df["model_utility"],
            df["forget_Q_A_Prob"],
            marker="o",
            markersize=3.4,
            linewidth=1.5,
            color=color,
            label=label,
        )
        for _, point in df.iterrows():
            axes[1].annotate(
                f"{point['strength_float']:.4g}",
                (point["model_utility"], point["forget_Q_A_Prob"]),
                textcoords="offset points",
                xytext=(3, 3),
                fontsize=5.8,
                color=color,
            )

    axes[0].set_xlabel(r"Strength $\alpha$")
    axes[0].set_ylabel("Model utility")
    axes[1].set_xlabel("Model utility")
    axes[1].set_ylabel("Forget probability")
    axes[1].set_ylim(0.24, 0.91)
    for ax in axes:
        ax.grid(True, alpha=0.22, linewidth=0.6)
        ax.legend(frameon=False)
    fig.suptitle(title, y=1.03, fontsize=9.5)
    save_figure(fig, out_dir, stem)


def plot_figure_a2(kforge: pd.DataFrame, paths: Paths) -> None:
    priority = ["_stage3down", "_bcal", ""]
    kron = ablation_sweep(kforge, mode="kron", retain="retain", suffix_priority=priority)
    diagonal = ablation_sweep(kforge, mode="diagonal", retain="retain", suffix_priority=priority)
    common = sorted(set(kron["strength_float"]).intersection(diagonal["strength_float"]))
    kron = kron[kron["strength_float"].isin(common)]
    diagonal = diagonal[diagonal["strength_float"].isin(common)]
    plot_two_panel_ablation(
        [("Kronecker", kron, KFORGE_COLOR), ("Diagonal", diagonal, SCRATCH_COLOR)],
        paths.out_dir,
        "figA2_diagonal_vs_kronecker",
        "Diagonal Fisher loses the useful tradeoff",
        "figA2_diagonal_vs_kronecker_data.csv",
    )


def plot_figure_a3(kforge: pd.DataFrame, paths: Paths) -> None:
    priority = ["_stage3down", "_bcal", ""]
    retain = ablation_sweep(kforge, mode="kron", retain="retain", suffix_priority=priority)
    forget_only = ablation_sweep(kforge, mode="kron", retain="forgetonly", suffix_priority=priority)
    common = sorted(set(retain["strength_float"]).intersection(forget_only["strength_float"]))
    retain = retain[retain["strength_float"].isin(common)]
    forget_only = forget_only[forget_only["strength_float"].isin(common)]
    plot_two_panel_ablation(
        [("Retain-whitened", retain, KFORGE_COLOR), ("Forget-only", forget_only, SCRATCH_COLOR)],
        paths.out_dir,
        "figA3_forget_only_vs_retain_whitened",
        "Retain whitening is load-bearing",
        "figA3_forget_only_vs_retain_whitened_data.csv",
    )


def write_manifest(paths: Paths, runs: pd.DataFrame, aggregate: pd.DataFrame) -> None:
    figure_files = sorted(p.name for p in paths.out_dir.glob("fig*.png"))
    figure_pdf_files = sorted(p.name for p in paths.out_dir.glob("fig*.pdf"))
    notes = [
        "# K-FORGE Figure Manifest",
        "",
        f"Output directory: `{paths.out_dir}`",
        "",
        "Generated figures:",
        "",
    ]
    for name in figure_files:
        notes.append(f"- `{name}`")
    notes += ["", "PDF companions:", ""]
    for name in figure_pdf_files:
        notes.append(f"- `{name}`")
    notes += [
        "",
        "Data inventory:",
        "",
        f"- Parsed Week 2 init/eval runs: `{len(runs)}`.",
        f"- Aggregated init rows: `{len(aggregate)}`.",
        f"- Figure 2 uses all available `*_week2_EVAL_FP32` summaries at generation time.",
        "- Figure A4 was not generated because no layer-inclusion acceleration sweep is present yet.",
        "",
        "Figure 2 acceleration annotation:",
        "",
        "For each panel, the reported factor is the strongest observed `scratch steps to match / K-FORGE-init steps used` ratio.",
        "A `k>` label means K-FORGE-init reached a forget probability lower than the final available scratch point, so the plotted ratio is a lower bound.",
    ]
    (paths.out_dir / "MANIFEST.md").write_text("\n".join(notes) + "\n")


def parse_args() -> argparse.Namespace:
    script_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=script_root)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else root / "saves/figures/kforge"
    paths = Paths(
        root=root,
        eval_root=root / "saves/eval",
        kforge_csv=root / "saves/eval/kforge_all_summary.csv",
        spectrum_csv=root / "saves/spectrum/kforge_spectrum_summary.csv",
        out_dir=out_dir,
    )
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    kforge = pd.read_csv(paths.kforge_csv)
    kforge["suffix"] = kforge["suffix"].fillna("")
    spectrum = pd.read_csv(paths.spectrum_csv)
    runs = load_week2_runs(paths.eval_root)
    aggregate = aggregate_runs(runs)
    runs.to_csv(paths.out_dir / "week2_runs_used.csv", index=False)
    aggregate.to_csv(paths.out_dir / "week2_aggregate_used.csv", index=False)

    plot_figure1(kforge, paths)
    plot_figure2(aggregate, paths)
    plot_figure3(kforge, aggregate, paths)
    plot_figure_a1(spectrum, paths)
    plot_figure_a2(kforge, paths)
    plot_figure_a3(kforge, paths)
    write_manifest(paths, runs, aggregate)

    print(f"Wrote figures to {paths.out_dir}")
    print(f"Parsed {len(runs)} Week 2 runs into {len(aggregate)} aggregate rows")
    print("Skipped Figure A4: layer-inclusion acceleration data is not present")


if __name__ == "__main__":
    main()
