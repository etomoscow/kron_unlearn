#!/usr/bin/env python3
"""Create paper-ready K-FORGE figures from saved CSV artifacts."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
IN_DIR = ROOT / "open-unlearning" / "saves" / "figures" / "kforge_corrected"
OUT_DIR = IN_DIR


COLORS = {
    "kforge": "#0072B2",
    "frontier": "#009E73",
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
    """Lightly shade low-utility/high-forget as cool and high-utility/low-forget as warm."""
    x = np.linspace(0.0, 1.0, 240)
    y = np.linspace(0.0, 1.0, 240)
    xx, yy = np.meshgrid(x, y)
    desirability = 0.5 * xx + 0.5 * (1.0 - yy)
    ax.imshow(
        desirability,
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        origin="lower",
        cmap=PREFERENCE_CMAP,
        alpha=0.55,
        aspect="auto",
        interpolation="bilinear",
        zorder=-10,
    )
    ax.text(
        0.02,
        0.96,
        "less desired",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.8,
        color="#456C87",
        alpha=0.78,
        zorder=-4,
    )
    ax.text(
        0.86,
        0.06,
        "more desired",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.8,
        color="#9A4E17",
        alpha=0.82,
        zorder=-4,
    )


def draw_shift_arrow(ax, start, end, rad: float) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>",
        mutation_scale=13.5,
        linewidth=2.0,
        color=COLORS["arrow"],
        alpha=0.86,
        shrinkA=8,
        shrinkB=9,
        zorder=2,
    )
    ax.add_patch(arrow)


def style_delta_axes(ax) -> None:
    ax.grid(True, axis="y", color="#D8D8D8", linewidth=0.55, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#444444")
        ax.spines[side].set_linewidth(0.7)


def save(fig, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{name}.pdf")
    fig.savefig(OUT_DIR / f"{name}.png", dpi=300)
    plt.close(fig)


def load_forget10_init_data() -> pd.DataFrame:
    df = pd.read_csv(IN_DIR / "corrected_aggregate_used.csv")
    return df[
        (df["forget"].eq("forget10"))
        & (df["init"].isin(["scratch", "kforge_s045"]))
        & (df["algo"].isin(["NPO", "SimNPO"]))
    ].copy()


def make_delta_over_scratch() -> None:
    df = load_forget10_init_data()
    rows = []
    for algo in ["NPO", "SimNPO"]:
        sub = df[df["algo"].eq(algo)]
        for step in [50, 100, 250]:
            scratch = sub[(sub["steps"].eq(step)) & (sub["init"].eq("scratch"))].iloc[0]
            kforge = sub[(sub["steps"].eq(step)) & (sub["init"].eq("kforge_s045"))].iloc[0]
            rows.append(
                {
                    "algo": algo,
                    "steps": int(step),
                    "delta_forget_prob": scratch["forget_Q_A_Prob_mean"]
                    - kforge["forget_Q_A_Prob_mean"],
                    "relative_forget_reduction_pct": 100.0
                    * (
                        scratch["forget_Q_A_Prob_mean"]
                        - kforge["forget_Q_A_Prob_mean"]
                    )
                    / scratch["forget_Q_A_Prob_mean"],
                    "delta_utility": kforge["model_utility_mean"]
                    - scratch["model_utility_mean"],
                    "scratch_forget_prob": scratch["forget_Q_A_Prob_mean"],
                    "kforge_forget_prob": kforge["forget_Q_A_Prob_mean"],
                    "scratch_utility": scratch["model_utility_mean"],
                    "kforge_utility": kforge["model_utility_mean"],
                }
            )
    delta = pd.DataFrame(rows)
    delta.to_csv(OUT_DIR / "fig_delta_over_scratch_data.csv", index=False)

    fig, ax = plt.subplots(figsize=(4.05, 2.75))
    styles = {
        "NPO": dict(color="#0072B2", marker="o", label="NPO"),
        "SimNPO": dict(color="#C2541A", marker="s", label="SimNPO"),
    }
    label_offsets = {
        ("NPO", 50): (-18, 1),
        ("NPO", 100): (8, 0),
        ("NPO", 250): (8, -2),
        ("SimNPO", 50): (-30, -11),
        ("SimNPO", 100): (8, -3),
        ("SimNPO", 250): (-30, 6),
    }

    style_delta_axes(ax)
    ax.axhspan(0.0, 0.062, color="#0072B2", alpha=0.055, zorder=-5)
    ax.axhline(0.0, color="#333333", linewidth=0.85, linestyle="--", alpha=0.85)

    for algo in ["NPO", "SimNPO"]:
        cur = delta[delta["algo"].eq(algo)].sort_values("steps")
        style = styles[algo]
        ax.plot(
            cur["relative_forget_reduction_pct"],
            cur["delta_utility"],
            color=style["color"],
            linestyle="--",
            linewidth=0.75,
            alpha=0.45,
            zorder=2,
        )
        ax.scatter(
            cur["relative_forget_reduction_pct"],
            cur["delta_utility"],
            s=34,
            color=style["color"],
            marker=style["marker"],
            edgecolor="white",
            linewidth=0.6,
            label=style["label"],
            zorder=3,
        )
        for _, row in cur.iterrows():
            dx, dy = label_offsets[(algo, int(row["steps"]))]
            ax.annotate(
                f"{int(row['steps'])}",
                xy=(row["relative_forget_reduction_pct"], row["delta_utility"]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=6.4,
                color="#242424",
            )

    ax.text(
        2.0,
        0.058,
        "utility gain",
        ha="left",
        va="top",
        fontsize=6.0,
        color="#345F86",
        alpha=0.78,
    )
    ax.set_xlim(0.0, 55.0)
    ax.set_ylim(-0.01, 0.062)
    ax.set_xlabel("Relative forget-probability reduction (%)")
    ax.set_ylabel(r"$\Delta$ Utility")
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=2,
        fontsize=7.0,
    )
    fig.tight_layout(pad=0.35)
    save(fig, "fig_delta_over_scratch")


def make_one_shot_frontier() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sweep = pd.read_csv(IN_DIR / "fig1_wiener_v2_strength_sweep_data.csv")
    refs = pd.read_csv(IN_DIR / "reference_points_used.csv")

    # Keep the non-collapsed operating region shown in the main text.
    sweep = sweep[sweep["strength"] <= 0.8].copy()
    sweep = sweep.sort_values("strength")

    source = pd.concat(
        [
            refs[refs["method"].eq("Base model")].assign(strength=np.nan),
            sweep.assign(method=lambda d: "K-FORGE alpha=" + d["strength"].map("{:.2g}".format)),
        ],
        ignore_index=True,
        sort=False,
    )
    source[
        [
            "method",
            "strength",
            "model_utility",
            "forget_Q_A_Prob",
            "forget_Q_A_ROUGE",
            "extraction_strength",
        ]
    ].to_csv(OUT_DIR / "fig_one_shot_frontier_data.csv", index=False)

    fig, ax = plt.subplots(figsize=(3.25, 2.55))
    ax.set_xlim(0.508, 0.602)
    ax.set_ylim(0.38, 0.895)
    add_preference_gradient(ax, ax.get_xlim(), ax.get_ylim())
    style_axes(ax)

    ax.plot(
        sweep["model_utility"],
        sweep["forget_Q_A_Prob"],
        color=COLORS["frontier"],
        marker="o",
        markersize=4.0,
        markeredgecolor="white",
        markeredgewidth=0.45,
        linewidth=1.7,
        zorder=3,
    )

    base = refs[refs["method"].eq("Base model")].iloc[0]
    ax.scatter(
        [base["model_utility"]],
        [base["forget_Q_A_Prob"]],
        color=COLORS["scratch"],
        marker="D",
        s=26,
        zorder=4,
    )
    ax.annotate(
        "base",
        xy=(base["model_utility"], base["forget_Q_A_Prob"]),
        xytext=(-42, -2),
        textcoords="offset points",
        color=COLORS["text"],
        fontsize=7.0,
        va="top",
    )

    for _, row in sweep.iterrows():
        strength = float(row["strength"])
        if strength not in {0.3, 0.45, 0.6, 0.8}:
            continue
        label = rf"$\alpha={strength:.2g}$"
        dx, dy, leader = {
            0.3: (-64, -4, True),
            0.45: (-64, -17, True),
            0.6: (5, -1, False),
            0.8: (5, -2, False),
        }[strength]
        kwargs = {
            "text": label,
            "xy": (row["model_utility"], row["forget_Q_A_Prob"]),
            "xytext": (dx, dy),
            "textcoords": "offset points",
            "fontsize": 6.8,
            "color": COLORS["text"],
        }
        if leader:
            kwargs["arrowprops"] = dict(
                arrowstyle="-",
                color="#555555",
                linewidth=0.55,
                shrinkA=1,
                shrinkB=3,
            )
        ax.annotate(**kwargs)

    ax.text(
        0.529,
        0.565,
        "K-FORGE one-shot",
        color=COLORS["frontier"],
        fontsize=7.2,
        rotation=35,
        ha="center",
        va="center",
    )
    ax.set_xlabel("Model Utility ↑")
    ax.set_ylabel("Forget Q/A Probability ↓")
    save(fig, "fig_one_shot_frontier")


def make_matched_init_arrows() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_forget10_init_data()
    df.to_csv(OUT_DIR / "fig_matched_init_arrows_data.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65))
    for ax, algo in zip(axes, ["NPO", "SimNPO"]):
        sub = df[df["algo"].eq(algo)]
        if algo == "NPO":
            ax.set_xlim(0.50, 0.608)
            ax.set_ylim(0.015, 0.100)
        else:
            ax.set_xlim(0.568, 0.599)
            ax.set_ylim(0.24, 0.69)
        add_preference_gradient(ax, ax.get_xlim(), ax.get_ylim())
        style_axes(ax)

        label_offsets = {
            ("NPO", 50): (-7, 3),
            ("NPO", 100): (-2, 5),
            ("NPO", 250): (10, 2),
            ("SimNPO", 50): (4, 5),
            ("SimNPO", 100): (9, 0),
            ("SimNPO", 250): (-15, 4),
        }
        arrow_rads = {
            ("NPO", 50): -0.03,
            ("NPO", 100): 0.06,
            ("NPO", 250): 0.16,
            ("SimNPO", 50): -0.06,
            ("SimNPO", 100): 0.07,
            ("SimNPO", 250): -0.18,
        }

        for step in [50, 100, 250]:
            s = sub[(sub["steps"].eq(step)) & (sub["init"].eq("scratch"))].iloc[0]
            k = sub[(sub["steps"].eq(step)) & (sub["init"].eq("kforge_s045"))].iloc[0]
            draw_shift_arrow(
                ax,
                (s["model_utility_mean"], s["forget_Q_A_Prob_mean"]),
                (k["model_utility_mean"], k["forget_Q_A_Prob_mean"]),
                arrow_rads[(algo, step)],
            )
            mid_x = 0.5 * (s["model_utility_mean"] + k["model_utility_mean"])
            mid_y = 0.5 * (s["forget_Q_A_Prob_mean"] + k["forget_Q_A_Prob_mean"])
            dx, dy = label_offsets[(algo, step)]
            ax.annotate(
                f"{step}",
                xy=(mid_x, mid_y),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=7.0,
                fontweight="semibold",
                color="#2A2A2A",
                zorder=5,
            )

        for init, color, marker, label, fill in [
            ("scratch", COLORS["scratch"], "o", "scratch", "none"),
            ("kforge_s045", COLORS["kforge"], "o", r"K-FORGE ($\alpha=.45$)", COLORS["kforge"]),
        ]:
            cur = sub[sub["init"].eq(init)].sort_values("steps")
            ax.errorbar(
                cur["model_utility_mean"],
                cur["forget_Q_A_Prob_mean"],
                yerr=cur["forget_Q_A_Prob_std"],
                fmt=marker,
                markersize=4.4,
                markerfacecolor=fill,
                markeredgecolor=color,
                markeredgewidth=1.2,
                ecolor=color,
                elinewidth=0.75,
                capsize=2.0,
                linestyle="none",
                label=label,
                zorder=3,
            )

        ax.set_title(algo)
        ax.set_xlabel("Model Utility ↑")
        if algo == "NPO":
            ax.set_ylabel("Forget Q/A Probability ↓")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
    )
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig_matched_init_arrows")


def main() -> None:
    configure_matplotlib()
    make_one_shot_frontier()
    make_matched_init_arrows()
    make_delta_over_scratch()
    print(f"Wrote paper figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
