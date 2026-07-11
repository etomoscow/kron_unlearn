#!/usr/bin/env python3
import argparse
import json
import statistics
from pathlib import Path

from scipy.stats import ttest_rel


ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "saves/eval"
METRICS = (
    "forget_Q_A_Prob",
    "model_utility",
    "extraction_strength",
    "forget_Q_A_ROUGE",
)


def read(path: Path) -> dict:
    with path.open() as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Invalid summary: {path}")
    return data


def paired(label: str, left: list[dict], right: list[dict], metrics=METRICS) -> None:
    print(f"\n### {label}\n")
    print("| Metric | Left | Right | Delta (right-left) | paired p |")
    print("|---|---:|---:|---:|---:|")
    for metric in metrics:
        a = [row[metric] for row in left]
        b = [row[metric] for row in right]
        delta = [y - x for x, y in zip(a, b)]
        p = ttest_rel(b, a).pvalue if len(a) > 1 else float("nan")
        print(
            f"| {metric} | {statistics.mean(a):.6f} | "
            f"{statistics.mean(b):.6f} | {statistics.mean(delta):+.6f} "
            f"$\\pm$ {statistics.stdev(delta) if len(delta) > 1 else 0:.6f} | {p:.3g} |"
        )


def tofu_rows(method: str, init: str, steps: int, tag: str, seeds=range(3), suffix="") -> list[dict]:
    return [
        read(
            EVAL
            / f"tofu_gemma-3-1b-it_forget10_{method}_{init}_S{steps}_seed{seed}_{tag}{suffix}"
            / "TOFU_SUMMARY.json"
        )
        for seed in seeds
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    expected = 0
    for method in ("NPO", "SimNPO"):
        kforge = tofu_rows(method, "kforge", 100, "rebuttal_gemma3_tuned080_v1_EVAL")
        scratch = tofu_rows(method, "scratch", 100, "rebuttal_gemma3_tuned080_v1_EVAL")
        expected += len(kforge) + len(scratch)
        paired(f"Gemma {method} S100: scratch vs K-FORGE", scratch, kforge)

        for control in ("random_rank2", "weight_svd", "diagonal", "forget_only"):
            rows = tofu_rows(
                method,
                control,
                100,
                "rebuttal_gemma3_controls_v2_EVAL_FP32",
            )
            expected += len(rows)
            paired(f"Gemma {method} S100: {control} vs K-FORGE", rows, kforge)

        compute_steps = 103 if method == "NPO" else 105
        compute = tofu_rows(
            method,
            "scratch",
            compute_steps,
            "rebuttal_gemma3_compute_matched_v2_EVAL_FP32",
        )
        expected += len(compute)
        paired(f"Gemma {method}: compute-matched scratch S{compute_steps} vs K-FORGE S100", compute, kforge)

    for method in ("NPO", "SimNPO"):
        old_s = tofu_rows(method, "scratch", 100, "rebuttal_gemma3_tuned080_v1_EVAL")
        old_k = tofu_rows(method, "kforge", 100, "rebuttal_gemma3_tuned080_v1_EVAL")
        held_s = tofu_rows(method, "scratch", 100, "rebuttal_gemma3_confirm_seed3_v1_EVAL", seeds=(3,))
        held_k = tofu_rows(method, "kforge", 100, "rebuttal_gemma3_confirm_seed3_v1_EVAL", seeds=(3,))
        expected += 2
        paired(f"Gemma {method} S100: four-seed aggregate", old_s + held_s, old_k + held_k)

    muse_scratch = [
        read(
            EVAL
            / f"muse_Llama-2-7b-hf_News_SimNPO_scratch_S100_seed{seed}_rebuttal_muse_clean_v3"
            / "MUSE_SUMMARY.json"
        )
        for seed in range(3)
    ]
    muse_kforge = [
        read(
            EVAL
            / f"muse_Llama-2-7b-hf_News_SimNPO_kforge_grid_1p00_S100_seed{seed}_rebuttal_muse_followup_v2"
            / "MUSE_SUMMARY.json"
        )
        for seed in range(3)
    ]
    expected += 6
    paired(
        "MUSE-News SimNPO S100: scratch vs selected alpha=1.0",
        muse_scratch,
        muse_kforge,
        (
            "extraction_strength",
            "forget_knowmem_ROUGE",
            "forget_verbmem_ROUGE",
            "retain_knowmem_ROUGE",
        ),
    )

    for bits in (8, 4):
        left, right = [], []
        for seed in range(3):
            left.append(
                read(
                    EVAL
                    / f"tofu_Llama-3.2-1B-Instruct_forget10_NPO_scratch_S50_seed{seed}_quant{bits}_rebuttal_v1"
                    / "TOFU_SUMMARY.json"
                )
            )
            right.append(
                read(
                    EVAL
                    / f"tofu_Llama-3.2-1B-Instruct_forget10_NPO_kforge_S50_seed{seed}_quant{bits}_rebuttal_v1"
                    / "TOFU_SUMMARY.json"
                )
            )
        expected += 6
        paired(f"Llama NPO S50 after {bits}-bit loading: scratch vs K-FORGE", left, right)

    if args.check:
        print(f"\nPASS: parsed {expected} expected summaries")


if __name__ == "__main__":
    main()
