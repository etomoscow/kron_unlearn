#!/usr/bin/env python3
"""Create the matched-budget K-FORGE dumbbell plot for the paper."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "open-unlearning" / "saves" / "figures" / "kforge_corrected"
INPUT = FIG_DIR / "corrected_aggregate_used.csv"
OUT_STEM = FIG_DIR / "fig_matched_budget_dumbbell"


COLORS = {
    "scratch": "#5A5A5A",
    "kforge": "#0072B2",
    "line": "#9A9A9A",
    "grid": "#DEDEDE",
    "text": "#242424",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.6,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def load_rows() -> pd.DataFrame:
    df = pd.read_csv(INPUT)
    df = df[
        (df["forget"].eq("forget10"))
        & (df["algo"].isin(["NPO", "SimNPO"]))
        & (df["init"].isin(["scratch", "kforge_s045"]))
    ].copy()

    rows = []
    for algo in ["NPO", "SimNPO"]:
        sub = df[df["algo"].eq(algo)]
        for steps in [50, 100, 250]:
            scratch = sub[(sub["steps"].eq(steps)) & (sub["init"].eq("scratch"))].iloc[0]
            kforge = sub[
                (sub["steps"].eq(steps)) & (sub["init"].eq("kforge_s045"))
            ].iloc[0]
            rows.append(
                {
                    "label": f"{algo}-{steps}",
                    "algo": algo,
                    "steps": steps,
                    "scratch_forget_prob": scratch["forget_Q_A_Prob_mean"],
                    "kforge_forget_prob": kforge["forget_Q_A_Prob_mean"],
                    "forget_relative_reduction_pct": 100.0
                    * (
                        scratch["forget_Q_A_Prob_mean"]
                        - kforge["forget_Q_A_Prob_mean"]
                    )
                    / scratch["forget_Q_A_Prob_mean"],
                    "scratch_utility": scratch["model_utility_mean"],
                    "kforge_utility": kforge["model_utility_mean"],
                    "utility_delta": kforge["model_utility_mean"]
                    - scratch["model_utility_mean"],
                }
            )
    return pd.DataFrame(rows)


def style_axis(ax, title: str, xlabel: str) -> None:
    ax.set_title(title, pad=6)
    ax.set_xlabel(xlabel)
    ax.grid(True, axis="x", color=COLORS["grid"], linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#444444")
        ax.spines[side].set_linewidth(0.75)


def draw_panel(ax, rows: pd.DataFrame, y_pos: np.ndarray, metric: str) -> None:
    scratch_col = f"scratch_{metric}"
    kforge_col = f"kforge_{metric}"
    for _, row in rows.iterrows():
        y = y_pos[row.name]
        ax.plot(
            [row[scratch_col], row[kforge_col]],
            [y, y],
            color=COLORS["line"],
            linewidth=0.9,
            zorder=1,
        )
        ax.scatter(
            row[scratch_col],
            y,
            s=24,
            facecolors="white",
            edgecolors=COLORS["scratch"],
            linewidths=1.2,
            zorder=3,
        )
        ax.scatter(
            row[kforge_col],
            y,
            s=28,
            color=COLORS["kforge"],
            edgecolors="white",
            linewidths=0.5,
            zorder=4,
        )


def make_plot() -> None:
    rows = load_rows()
    rows.to_csv(f"{OUT_STEM}_data.csv", index=False)

    y_pos = np.arange(len(rows))[::-1]
    rows = rows.copy()
    rows.index = np.arange(len(rows))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.45, 2.9),
        sharey=True,
        gridspec_kw={"width_ratios": [1.08, 1.0]},
    )

    draw_panel(axes[0], rows, y_pos, "forget_prob")
    draw_panel(axes[1], rows, y_pos, "utility")

    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(rows["label"])
    axes[1].tick_params(axis="y", left=False, labelleft=False)

    style_axis(axes[0], "A. Forget Q/A Probability", "lower is better")
    style_axis(axes[1], "B. Model Utility", "higher is better")

    axes[0].invert_xaxis()
    axes[0].set_xlim(0.70, 0.00)
    axes[0].set_xticks([0.7, 0.5, 0.3, 0.1])
    axes[1].set_xlim(0.50, 0.61)
    axes[1].set_xticks([0.50, 0.55, 0.60])

    for _, row in rows.iterrows():
        y = y_pos[row.name]
        axes[0].annotate(
            f"-{row['forget_relative_reduction_pct']:.0f}%",
            xy=(row["kforge_forget_prob"], y),
            xytext=(-14, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=6.4,
            color=COLORS["text"],
        )
        delta = row["utility_delta"]
        axes[1].annotate(
            f"{delta:+.3f}",
            xy=(row["kforge_utility"], y),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=6.4,
            color=COLORS["text"],
        )

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=COLORS["scratch"],
            markeredgewidth=1.2,
            markersize=5,
            label="scratch",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=COLORS["kforge"],
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=5,
            label=r"K-FORGE ($\alpha=.45$)",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.56, 1.04),
    )
    fig.tight_layout(w_pad=1.8)
    fig.savefig(f"{OUT_STEM}.pdf")
    fig.savefig(f"{OUT_STEM}.png", dpi=300)
    plt.close(fig)


def main() -> None:
    configure_matplotlib()
    make_plot()
    print(f"Wrote {OUT_STEM}.pdf")


if __name__ == "__main__":
    main()
