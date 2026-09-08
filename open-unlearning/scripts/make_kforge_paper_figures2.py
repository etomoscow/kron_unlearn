#!/usr/bin/env python3
"""Create paper-ready K-FORGE figures from saved CSV artifacts (improved)."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from kforge_matched_fixed import write_matched_figure

ROOT = Path(__file__).resolve().parents[2]
IN_DIR = ROOT / "open-unlearning" / "saves" / "figures" / "kforge_corrected"
OUT_DIR = IN_DIR


COLORS = {
    "kforge": "#CC79A7",  # teal-green: contrasts with blue gradient bg and orange arrows
    "scratch": "#5A5A5A",
    "arrow": "#C2541A",
    "grid": "#D0D0D0",
    "text": "#202020",
}

PREFERENCE_CMAP = LinearSegmentedColormap.from_list(
    "forget_utility_preference",
    ["#BBD7EC", "#F6EAD2", "#E98632"],
)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.5,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.015,
        }
    )


def style_axes(ax) -> None:
    ax.grid(True, color=COLORS["grid"], linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.8)


def add_preference_gradient(ax, xlim, ylim) -> None:
    """Shade low-utility/high-forget as cool and high-utility/low-forget as warm."""
    x = np.linspace(0.0, 1.0, 240)
    y = np.linspace(0.0, 1.0, 240)
    xx, yy = np.meshgrid(x, y)
    desirability = 0.5 * xx + 0.5 * (1.0 - yy)
    ax.imshow(
        desirability,
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        origin="lower",
        cmap=PREFERENCE_CMAP,
        alpha=0.45,          # FIX: slightly more subtle
        aspect="auto",
        interpolation="bilinear",
        zorder=-10,
    )
    # FIX: move corner labels inside the actual corners, slightly larger
    ax.text(
        0.03, 0.97,
        "less desired",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=6.4,
        color="#3B6480",
        alpha=0.85,
        zorder=-4,
        style="italic",
    )
    ax.text(
        0.97, 0.03,
        "more desired",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=6.4,
        color="#9A4E17",
        alpha=0.88,
        zorder=-4,
        style="italic",
    )


def save(fig, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{name}.pdf")
    fig.savefig(OUT_DIR / f"{name}.png", dpi=300)
    plt.close(fig)


def make_one_shot_frontier() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sweep = pd.read_csv(IN_DIR / "fig1_wiener_v2_strength_sweep_data.csv")
    refs  = pd.read_csv(IN_DIR / "reference_points_used.csv")

    sweep = sweep[sweep["strength"] <= 0.8].copy()
    sweep = sweep.sort_values("strength")

    source = pd.concat(
        [
            refs[refs["method"].eq("Base model")].assign(strength=np.nan),
            sweep.assign(method=lambda d: "K-FORGE alpha=" + d["strength"].map("{:.2g}".format)),
        ],
        ignore_index=True, sort=False,
    )
    source[["method","strength","model_utility","forget_Q_A_Prob",
            "forget_Q_A_ROUGE","extraction_strength"]].to_csv(
        OUT_DIR / "fig_one_shot_frontier_data.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(3.25, 2.65))  # FIX: slightly taller

    # FIX: tighter xlim; trim dead space on left; a touch of room on right
    ax.set_xlim(0.508, 0.606)
    # FIX: trim ylim top; α=0.1 ≈ 0.878 so 0.91 gives clean headroom
    ax.set_ylim(0.365, 0.910)
    add_preference_gradient(ax, ax.get_xlim(), ax.get_ylim())
    style_axes(ax)

    # --- K-FORGE frontier line ---
    ax.plot(
        sweep["model_utility"],
        sweep["forget_Q_A_Prob"],
        color=COLORS["kforge"],
        marker="o",
        markersize=4.5,   # FIX: slightly larger
        linewidth=1.7,
    )

    # --- Base model marker ---
    base = refs[refs["method"].eq("Base model")].iloc[0]
    ax.scatter(
        [base["model_utility"]],
        [base["forget_Q_A_Prob"]],
        color=COLORS["scratch"],
        marker="D",
        s=30,
        zorder=4,
    )
    # FIX: nudge "base" label up so it doesn't collide with α=0.1 annotation
    ax.annotate(
        "base",
        xy=(base["model_utility"], base["forget_Q_A_Prob"]),
        xytext=(4, 5),     # right & above the diamond
        textcoords="offset points",
        color=COLORS["text"],
        fontsize=7.0,
    )

    # --- α annotations ---
    # α=0.1 is ~0.001 from base in both dims — label it with a leader line
    # α=0.35 is ~0.001 from α=0.3 — drop it to avoid clutter
    # α=0.3 and α=0.45 are close but distinct — use leader lines to spread them
    leader_annotations = [
        # (strength, label_xy_offset_pts, ha)
        (0.10,  (-18, -22), "right"),   # down-left; clears "base" above
        (0.30,  (-50,  10), "right"),   # pull left with line to point
        (0.45,  (  6, -14), "left"),    # pull down-right
        (0.60,  (  6,  -1), "left"),    # plenty of room on the right
        (0.80,  (  6,  -2), "left"),
    ]
    for strength, (dx, dy), ha in leader_annotations:
        row = sweep[np.isclose(sweep["strength"], strength)].iloc[0]
        label = rf"$\alpha={strength:.2g}$"
        ax.annotate(
            label,
            xy=(row["model_utility"], row["forget_Q_A_Prob"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=6.8,
            color=COLORS["text"],
            ha=ha,
            arrowprops=dict(
                arrowstyle="-",
                color="#888888",
                linewidth=0.55,
                shrinkA=0,
                shrinkB=3,
            ) if abs(dx) > 12 or abs(dy) > 10 else None,
        )

    # --- "K-FORGE one-shot" diagonal label ---
    ax.text(
        0.529, 0.565,
        "K-FORGE one-shot",
        color=COLORS["kforge"],
        fontsize=7.2,
        rotation=37,
        ha="center", va="center",
    )

    # --- direction hint: single arrow labeled "better", no separate "more desired" text ---
    # ("more desired" corner label from add_preference_gradient is sufficient)
    ax.annotate(
        "better →",
        xy=(0.597, 0.398),
        xytext=(0.558, 0.490),
        arrowprops=dict(arrowstyle="->", color="#666666", linewidth=0.9),
        color="#666666",
        fontsize=7.0,
        ha="center",
    )

    ax.set_xlabel("Model Utility ↑", labelpad=1)
    ax.set_ylabel("Forget Q/A Probability ↓", labelpad=1)
    save(fig, "fig_one_shot_frontier")


def make_matched_init_arrows() -> None:
    write_matched_figure(
        IN_DIR / "corrected_aggregate_used.csv",
        OUT_DIR,
        stem="fig_matched_init_arrows",
    )


def main() -> None:
    configure_matplotlib()
    make_one_shot_frontier()
    make_matched_init_arrows()
    print(f"Wrote paper figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
