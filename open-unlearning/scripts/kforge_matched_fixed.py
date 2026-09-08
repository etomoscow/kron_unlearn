#!/usr/bin/env python3
"""Matched-step K-FORGE plot: real endpoints, stable labels, no shifted arrows.

Only replaces the two-panel matched-initialization figure. The one-shot
frontier code can remain unchanged. This module contains NO experimental
or demonstration data; all plotted values must come from the input CSV.

Example:
    python kforge_matched_fixed.py \
        --input path/to/corrected_aggregate_used.csv \
        --out-dir path/to/figures

Dependencies: matplotlib, numpy, pandas.
"""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd


# Preserve the supplied palette. #CC79A7 is pink/mauve, not teal-green.
COLORS = {
    "kforge": "#CC79A7",
    "scratch": "#5A5A5A",
    "arrow": "#C2541A",
    "grid": "#D0D0D0",
    "text": "#202020",
}
PREFERENCE_CMAP = LinearSegmentedColormap.from_list(
    "forget_utility_preference", ["#BBD7EC", "#F6EAD2", "#E98632"]
)
ALGOS = ("NPO", "SimNPO")
INITS = ("scratch", "kforge_s045")
STEPS = (50, 100, 250)
X = "model_utility_mean"
Y = "forget_Q_A_Prob_mean"
SD = "forget_Q_A_Prob_std"

# Preserve the axis ranges from the supplied figure. They are expanded only
# when necessary to avoid silently clipping input means or error bars.
PANEL_LIMITS = {
    "NPO": ((0.497, 0.610), (0.012, 0.105)),
    "SimNPO": ((0.550, 0.610), (0.220, 0.710)),
}

# Six editorial placements for this figure, not a generic collision solver.
# Each label is anchored to the TRUE midpoint of its straight connector.
# (dx in points, dy in points, horizontal alignment, vertical alignment)
STEP_LABELS = {
    ("NPO", 50): (0, 10, "center", "bottom"),
    ("NPO", 100): (0, 10, "center", "bottom"),
    ("NPO", 250): (-2, 12, "center", "bottom"),
    ("SimNPO", 50): (-10, 0, "right", "center"),
    ("SimNPO", 100): (-10, 0, "right", "center"),
    ("SimNPO", 250): (-10, 0, "right", "center"),
}

RC = {
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


def select_matched_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Require exactly one aggregate for every requested algo/init/step."""
    required = {"forget", "init", "algo", "steps", X, Y, SD}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")
    df = raw.loc[
        raw["forget"].eq("forget10")
        & raw["init"].isin(INITS)
        & raw["algo"].isin(ALGOS)
    ].copy()
    df["steps"] = pd.to_numeric(df["steps"], errors="raise")
    df = df.loc[df["steps"].isin(STEPS)].copy()
    keys = ["algo", "init", "steps"]
    duplicates = df.duplicated(keys, keep=False)
    if duplicates.any():
        bad = df.loc[duplicates, keys].drop_duplicates().to_dict("records")
        raise ValueError(
            f"Duplicate aggregate rows: {bad}. "
            "Filter model/configuration columns or aggregate runs explicitly."
        )
    expected = set(product(ALGOS, INITS, STEPS))
    actual = set(df[keys].itertuples(index=False, name=None))
    if expected - actual:
        raise ValueError(f"Missing matched rows: {sorted(expected - actual)}")
    for column in (X, Y, SD):
        df[column] = pd.to_numeric(df[column], errors="raise")
        if not np.isfinite(df[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{column} contains NaN or infinite values.")
    if df[SD].lt(0).any():
        raise ValueError(f"{SD} must be nonnegative.")
    return df.sort_values(keys).reset_index(drop=True)


def draw_pair_arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
) -> FancyArrowPatch | None:
    """Connect actual data points with point-sized endpoint gaps."""
    if np.array_equal(start, end):
        return None
    arrow = FancyArrowPatch(
        posA=start,
        posB=end,
        transform=ax.transData,
        connectionstyle="arc3,rad=0",
        arrowstyle="-|>",
        shrinkA=3.2,
        shrinkB=3.2,
        mutation_scale=7.5,
        linewidth=1.05,
        color=COLORS["arrow"],
        alpha=0.95,
        zorder=2,
    )
    ax.add_patch(arrow)
    return arrow


def _include_values(limits, values):
    """Retain preset limits unless doing so would hide data."""
    lo, hi = limits
    padding = 0.04 * (hi - lo)
    data_lo, data_hi = float(np.min(values)), float(np.max(values))
    return (
        min(lo, data_lo - padding) if data_lo <= lo else lo,
        max(hi, data_hi + padding) if data_hi >= hi else hi,
    )


def _style_panel(ax: Axes, gradient: bool) -> None:
    if gradient:
        x, y = np.meshgrid(np.linspace(0, 1, 240), np.linspace(0, 1, 240))
        ax.imshow(
            0.5 * x + 0.5 * (1 - y),
            extent=(*ax.get_xlim(), *ax.get_ylim()),
            origin="lower", cmap=PREFERENCE_CMAP, alpha=0.45, aspect="auto",
            interpolation="bilinear", zorder=-10,
        )
        ax.text(
            0.03, 0.97, "less desired", transform=ax.transAxes,
            ha="left", va="top", fontsize=6.4, color="#3B6480",
            alpha=0.85, zorder=-4, style="italic",
        )
        ax.text(
            0.97, 0.03, "more desired", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.4, color="#9A4E17",
            alpha=0.88, zorder=-4, style="italic",
        )
    ax.set_axisbelow(True)
    ax.grid(True, color=COLORS["grid"], linewidth=0.5, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.8)


def build_matched_figure(raw: pd.DataFrame, *, gradient: bool = True) -> Figure:
    """Create the two-panel figure without changing any input measurements."""
    df = select_matched_data(raw)
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75))
        for ax, algo in zip(axes, ALGOS):
            sub = df.loc[df["algo"].eq(algo)]
            xlim, ylim = PANEL_LIMITS[algo]
            ax.set_xlim(_include_values(xlim, sub[X].to_numpy()))
            ax.set_ylim(
                _include_values(
                    ylim,
                    np.r_[
                        sub[Y].to_numpy() - sub[SD].to_numpy(),
                        sub[Y].to_numpy() + sub[SD].to_numpy(),
                    ],
                )
            )
            _style_panel(ax, gradient)
            ax.set_title(algo, pad=2, fontweight="semibold")

            indexed = sub.set_index(["init", "steps"])
            for step in STEPS:
                s = indexed.loc[("scratch", step)]
                k = indexed.loc[("kforge_s045", step)]
                start = (float(s[X]), float(s[Y]))
                end = (float(k[X]), float(k[Y]))
                arrow = draw_pair_arrow(ax, start, end)
                if arrow is not None:
                    arrow.set_gid(f"pair:{algo}:{step}")
                midpoint = tuple((np.asarray(start) + np.asarray(end)) / 2)
                dx, dy, ha, va = STEP_LABELS[(algo, step)]
                label = ax.annotate(
                    str(step),
                    xy=midpoint,
                    xycoords="data",
                    xytext=(dx, dy),
                    textcoords="offset points",
                    ha=ha,
                    va=va,
                    fontsize=7.0,
                    fontweight="semibold",
                    color="#2A2A2A",
                    zorder=5,
                )
                label.set_gid(f"step:{algo}:{step}")

            for init, color, label, face in (
                ("scratch", COLORS["scratch"], "scratch", "none"),
                ("kforge_s045", COLORS["kforge"], r"K-FORGE ($\alpha=.45$)", COLORS["kforge"]),
            ):
                cur = sub.loc[sub["init"].eq(init)].sort_values("steps")
                ax.errorbar(
                    cur[X],
                    cur[Y],
                    yerr=cur[SD],
                    fmt="o",
                    markersize=5.0,
                    markerfacecolor=face,
                    markeredgecolor=color,
                    markeredgewidth=1.4,
                    ecolor=color,
                    elinewidth=0.9,
                    capsize=2.5,
                    linestyle="none",
                    label=label,
                    zorder=3,
                )
            ax.set_xlabel("Model Utility ↑")
            if algo == "NPO":
                ax.set_ylabel("Forget Q/A Probability ↓")
            ax.text(
                0.97, 0.97, "step counts: 50, 100, 250",
                transform=ax.transAxes, ha="right", va="top", fontsize=5.8,
                color="#444444", style="italic",
            )

        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend(
            handles, labels, loc="lower left", ncol=1, frameon=True,
            framealpha=0.82, edgecolor="#cccccc", fontsize=7.2,
            handlelength=1.4, handletextpad=0.5, borderpad=0.5,
        )
        fig.tight_layout(w_pad=2.0)
    return fig


def write_matched_figure(
    input_csv: str | Path,
    out_dir: str | Path,
    *,
    stem: str = "fig_matched_init_arrows_fixed",
    gradient: bool = True,
) -> tuple[Path, Path, Path]:
    """Read actual aggregates and save PDF, PNG, and the exact selected data."""
    selected = select_matched_data(pd.read_csv(input_csv))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pdf = out / f"{stem}.pdf"
    png = out / f"{stem}.png"
    csv = out / f"{stem}_data.csv"
    with plt.rc_context(RC):
        fig = build_matched_figure(selected, gradient=gradient)
        try:
            fig.savefig(pdf)
            fig.savefig(png, dpi=300)
        finally:
            plt.close(fig)
    selected.to_csv(csv, index=False)
    return pdf, png, csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to corrected_aggregate_used.csv",
    )
    parser.add_argument(
        "--out-dir", type=Path, help="Default: the input CSV directory"
    )
    parser.add_argument("--stem", default="fig_matched_init_arrows_fixed")
    parser.add_argument(
        "--white-background",
        action="store_true",
        help="Remove the qualitative background gradient",
    )
    args = parser.parse_args()
    paths = write_matched_figure(
        args.input,
        args.out_dir or args.input.parent,
        stem=args.stem,
        gradient=not args.white_background,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
