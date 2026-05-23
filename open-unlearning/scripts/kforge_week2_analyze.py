#!/usr/bin/env python3
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PATTERN = re.compile(
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


def load_rows(eval_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(eval_root.glob("*week2_EVAL_FP32/TOFU_SUMMARY.json")):
        match = PATTERN.fullmatch(path.parent.name)
        if not match:
            continue
        data = json.loads(path.read_text())
        row = match.groupdict()
        row["steps"] = int(row["steps"])
        row["seed"] = int(row["seed"])
        for metric in METRICS:
            row[metric] = data.get(metric)
        row["path"] = str(path)
        rows.append(row)
    return pd.DataFrame(rows)


def fmt_mean_std(mean: float, std: float, digits: int = 4) -> str:
    if pd.isna(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def write_markdown_tables(df: pd.DataFrame, aggregate: pd.DataFrame, out_dir: Path) -> None:
    lines = [
        "# K-FORGE Week 2 Analysis",
        "",
        "Generated from `saves/eval/*week2_EVAL_FP32/TOFU_SUMMARY.json`.",
        "",
        f"Total parsed runs: {len(df)}.",
        "",
        "Lower forget probability, ROUGE, and extraction strength indicate stronger forgetting; higher utility is better.",
        "",
    ]

    for forget in sorted(aggregate["forget"].unique()):
        lines += [f"## {forget}", ""]
        for trainer in ["NPO", "SimNPO", "RMU"]:
            sub = aggregate[(aggregate["forget"] == forget) & (aggregate["trainer"] == trainer)]
            if sub.empty:
                continue
            lines += [f"### {trainer}", ""]
            lines += [
                "| Steps | Init | n | Utility | Forget Prob | Forget ROUGE | Extraction Strength |",
                "|---:|---|---:|---:|---:|---:|---:|",
            ]
            for _, row in sub.sort_values(["steps", "init"]).iterrows():
                lines.append(
                    "| {steps} | {init} | {n} | {utility} | {prob} | {rouge} | {es} |".format(
                        steps=int(row["steps"]),
                        init=row["init"],
                        n=int(row["n"]),
                        utility=fmt_mean_std(row["model_utility_mean"], row["model_utility_std"]),
                        prob=fmt_mean_std(row["forget_Q_A_Prob_mean"], row["forget_Q_A_Prob_std"]),
                        rouge=fmt_mean_std(row["forget_Q_A_ROUGE_mean"], row["forget_Q_A_ROUGE_std"]),
                        es=fmt_mean_std(row["extraction_strength_mean"], row["extraction_strength_std"]),
                    )
                )
            lines.append("")

    (out_dir / "week2_summary.md").write_text("\n".join(lines))


def plot_forget10_steps(aggregate: pd.DataFrame, out_dir: Path) -> None:
    for trainer in ["NPO", "SimNPO", "RMU"]:
        sub = aggregate[(aggregate["forget"] == "forget10") & (aggregate["trainer"] == trainer)]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for init, color in [("scratch", "#4c78a8"), ("kforge", "#f58518")]:
            cur = sub[sub["init"] == init].sort_values("steps")
            if cur.empty:
                continue
            ax.errorbar(
                cur["steps"],
                cur["forget_Q_A_Prob_mean"],
                yerr=cur["forget_Q_A_Prob_std"].fillna(0),
                marker="o",
                linewidth=2,
                capsize=3,
                color=color,
                label=init,
            )
        ax.set_xscale("log")
        ax.set_xlabel("Training steps")
        ax.set_ylabel("Forget Q/A probability")
        ax.set_title(f"forget10 {trainer}: forgetting vs steps")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"forget10_{trainer.lower()}_forget_prob.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for init, color in [("scratch", "#4c78a8"), ("kforge", "#f58518")]:
            cur = sub[sub["init"] == init].sort_values("steps")
            if cur.empty:
                continue
            ax.errorbar(
                cur["steps"],
                cur["model_utility_mean"],
                yerr=cur["model_utility_std"].fillna(0),
                marker="o",
                linewidth=2,
                capsize=3,
                color=color,
                label=init,
            )
        ax.set_xscale("log")
        ax.set_xlabel("Training steps")
        ax.set_ylabel("Model utility")
        ax.set_title(f"forget10 {trainer}: utility vs steps")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"forget10_{trainer.lower()}_utility.png", dpi=180)
        plt.close(fig)


def plot_pareto(df: pd.DataFrame, out_dir: Path) -> None:
    sub = df[(df["forget"] == "forget10") & (df["trainer"].isin(["NPO", "SimNPO", "RMU"]))]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    markers = {"NPO": "o", "SimNPO": "s", "RMU": "^"}
    colors = {"scratch": "#4c78a8", "kforge": "#f58518"}
    for (trainer, init), cur in sub.groupby(["trainer", "init"]):
        ax.scatter(
            cur["forget_Q_A_Prob"],
            cur["model_utility"],
            s=46,
            marker=markers[trainer],
            color=colors[init],
            alpha=0.78,
            label=f"{trainer} {init}",
        )
    ax.set_xlabel("Forget Q/A probability (lower is better)")
    ax.set_ylabel("Model utility (higher is better)")
    ax.set_title("forget10 Pareto scatter")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "forget10_pareto_scatter.png", dpi=180)
    plt.close(fig)


def main() -> None:
    eval_root = Path("saves/eval")
    out_dir = Path("saves/analysis/week2")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_rows(eval_root)
    if df.empty:
        raise SystemExit("No week2 summaries found")
    df.to_csv(out_dir / "week2_runs.csv", index=False)

    aggregate = (
        df.groupby(["forget", "trainer", "steps", "init"], as_index=False)
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
    aggregate.to_csv(out_dir / "week2_aggregate.csv", index=False)
    write_markdown_tables(df, aggregate, out_dir)
    plot_forget10_steps(aggregate, out_dir)
    plot_pareto(df, out_dir)

    print(f"Wrote analysis artifacts to {out_dir}")
    print(f"Parsed {len(df)} runs")


if __name__ == "__main__":
    main()
