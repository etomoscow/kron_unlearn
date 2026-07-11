#!/usr/bin/env python3
import argparse
import json
import math
import numbers
import statistics
from pathlib import Path

from scipy.stats import ttest_rel


ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "saves/eval"
_SNAPSHOT_UNSET = object()
SNAPSHOT_INPUT = None
READ_RECORDS = {}
METRICS = (
    "forget_Q_A_Prob",
    "model_utility",
    "extraction_strength",
    "forget_Q_A_ROUGE",
)
MUSE_METRICS = (
    "extraction_strength",
    "forget_knowmem_ROUGE",
    "forget_verbmem_ROUGE",
    "retain_knowmem_ROUGE",
)


def tofu_eval_paths(
    model: str,
    method: str,
    init: str,
    steps: int,
    tag: str,
    seeds=range(3),
) -> list[Path]:
    return [
        EVAL / f"tofu_{model}_forget10_{method}_{init}_S{steps}_seed{seed}_{tag}"
        for seed in seeds
    ]


def muse_eval_paths(domain: str, init: str, steps: int, seeds=range(3)) -> list[Path]:
    tag = {"News": "rebuttal_muse_clean_v3", "Books": "rebuttal_muse_books_v2"}[domain]
    return [
        EVAL
        / f"muse_Llama-2-7b-hf_{domain}_SimNPO_{init}_S{steps}_seed{seed}_{tag}"
        for seed in seeds
    ]


def read(path: Path, snapshot=_SNAPSHOT_UNSET) -> dict:
    key = path.relative_to(ROOT).as_posix()
    if snapshot is _SNAPSHOT_UNSET:
        snapshot = SNAPSHOT_INPUT
    if snapshot is None:
        with path.open() as handle:
            data = json.load(handle)
    else:
        data = snapshot[key]
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Invalid summary: {path}")
    READ_RECORDS[key] = data
    return data


def paired(label: str, left: list[dict], right: list[dict], metrics=METRICS) -> None:
    if len(left) != len(right) or not left:
        raise ValueError(f"Invalid pair lengths for {label}: {len(left)} vs {len(right)}")
    print(f"\n### {label}\n")
    print("| Metric | Left | Right | Delta (right-left) | paired p |")
    print("|---|---:|---:|---:|---:|")
    for metric in metrics:
        a = [row[metric] for row in left]
        b = [row[metric] for row in right]
        if any(not isinstance(value, numbers.Real) or not math.isfinite(value) for value in a + b):
            raise ValueError(f"Non-finite {metric} in {label}")
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


def gemma_s100_rows(method: str, init: str) -> list[dict]:
    return tofu_rows(method, init, 100, "rebuttal_gemma3_tuned080_v1_EVAL") + tofu_rows(
        method,
        init,
        100,
        "rebuttal_gemma3_confirm_seed3_v1_EVAL",
        seeds=(3,),
    )


def relearn_rows(method: str, arm: str, epochs: int, step: int) -> list[dict]:
    return [
        read(
            ROOT
            / "saves/finetune"
            / f"tofu_Llama-3.2-1B-Instruct_forget10_{method}_{arm}_matched_seed{seed}_relearn_e{epochs}_rebuttal_matched_relearn_v1"
            / f"checkpoint-{step}/evals/TOFU_SUMMARY.json"
        )
        for seed in range(3)
    ]


def gemma_relearn_rows(method: str, arm: str, epochs: int, step: int) -> list[dict]:
    return [
        read(
            ROOT
            / "saves/finetune"
            / f"tofu_gemma-3-1b-it_forget10_{method}_{arm}_matched_S50_seed{seed}_relearn_e{epochs}_rebuttal_gemma_relearn_v1"
            / f"checkpoint-{step}/evals/TOFU_SUMMARY.json"
        )
        for seed in range(3)
    ]


def llama_unlearn_rows(method: str, arm: str, steps: int) -> list[dict]:
    init = "scratch" if arm == "scratch" else "kforge_s06"
    return [
        read(
            ROOT
            / f"saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget10_{method}_{init}_S{steps}_seed{seed}_v2lam0p01_corr"
            / f"checkpoint-{steps}/evals/TOFU_SUMMARY.json"
        )
        for seed in range(3)
    ]


def llama_fp32_rows(method: str, init: str, steps: int, tag: str) -> list[dict]:
    return [
        read(
            EVAL
            / f"tofu_Llama-3.2-1B-Instruct_forget10_{method}_{init}_S{steps}_seed{seed}_{tag}"
            / "TOFU_SUMMARY.json"
        )
        for seed in range(3)
    ]


def matched_quant_rows(arm: str, bits: int) -> list[dict]:
    return [
        read(
            EVAL
            / f"tofu_Llama-3.2-1B-Instruct_forget10_NPO_{arm}_matched_seed{seed}_quant{bits}_rebuttal_matched_quant_v1"
            / "TOFU_SUMMARY.json"
        )
        for seed in range(3)
    ]


def simnpo_quant_rows(arm: str, bits: int) -> list[dict]:
    return [
        read(
            EVAL
            / f"tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_{arm}_S50_seed{seed}_quant{bits}_rebuttal_simnpo_quant_v1"
            / "TOFU_SUMMARY.json"
        )
        for seed in range(3)
    ]


def gemma_quant_rows(method: str, arm: str, bits: int) -> list[dict]:
    return [
        read(
            EVAL
            / f"tofu_gemma-3-1b-it_forget10_{method}_{arm}_S100_seed{seed}_quant{bits}_rebuttal_gemma_quant_v1"
            / "TOFU_SUMMARY.json"
        )
        for seed in range(4)
    ]


def changes(before: list[dict], after: list[dict]) -> list[dict]:
    return [
        {metric: post[metric] - pre[metric] for metric in METRICS}
        for pre, post in zip(before, after)
    ]


def main() -> None:
    global SNAPSHOT_INPUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--snapshot-in", type=Path)
    parser.add_argument("--snapshot-out", type=Path)
    args = parser.parse_args()

    if args.snapshot_in:
        with args.snapshot_in.open() as handle:
            SNAPSHOT_INPUT = json.load(handle)
        if not isinstance(SNAPSHOT_INPUT, dict) or not SNAPSHOT_INPUT:
            raise ValueError(f"Invalid snapshot: {args.snapshot_in}")

    expected = 0
    for method in ("NPO", "SimNPO"):
        kforge = tofu_rows(method, "kforge", 100, "rebuttal_gemma3_tuned080_v1_EVAL")
        scratch = tofu_rows(method, "scratch", 100, "rebuttal_gemma3_tuned080_v1_EVAL")
        expected += len(kforge) + len(scratch)
        paired(f"Gemma {method} S100: scratch vs K-FORGE", scratch, kforge)

        kforge_four = gemma_s100_rows(method, "kforge")
        for control in ("random_rank2", "weight_svd", "diagonal", "forget_only"):
            rows = tofu_rows(
                method,
                control,
                100,
                "rebuttal_gemma3_controls_v2_EVAL_FP32",
                seeds=range(4),
            )
            expected += len(rows)
            paired(f"Gemma {method} S100: {control} vs K-FORGE", rows, kforge_four)

        compute_steps = 103 if method == "NPO" else 105
        compute = tofu_rows(
            method,
            "scratch",
            compute_steps,
            "rebuttal_gemma3_compute_matched_v2_EVAL_FP32",
            seeds=range(4),
        )
        expected += len(compute)
        paired(
            f"Gemma {method}: compute-matched scratch S{compute_steps} vs K-FORGE S100",
            compute,
            kforge_four,
        )

    for method in ("NPO", "SimNPO"):
        scratch = gemma_s100_rows(method, "scratch")
        kforge = gemma_s100_rows(method, "kforge")
        expected += len(scratch) + len(kforge)
        paired(f"Gemma {method} S100: four-seed aggregate", scratch, kforge)

    for method in ("NPO", "SimNPO"):
        for steps in (50, 100, 250):
            scratch = [
                read(path / "TOFU_SUMMARY.json")
                for path in tofu_eval_paths(
                    "Llama-3.2-3B-Instruct",
                    method,
                    "scratch",
                    steps,
                    "3b_minimal_v2_EVAL_FP32",
                )
            ]
            kforge = [
                read(path / "TOFU_SUMMARY.json")
                for path in tofu_eval_paths(
                    "Llama-3.2-3B-Instruct",
                    method,
                    "kforge_s045",
                    steps,
                    "3b_minimal_v2_EVAL_FP32",
                )
            ]
            expected += len(scratch) + len(kforge)
            paired(f"Llama-3.2-3B {method} S{steps}: scratch vs K-FORGE", scratch, kforge)

    for steps in (50, 100):
        scratch = [
            read(path / "TOFU_SUMMARY.json")
            for path in tofu_eval_paths(
                "Qwen2.5-1.5B-Instruct",
                "SimNPO",
                "scratch",
                steps,
                "rebuttal_qwen15_v1_EVAL",
            )
        ]
        kforge = [
            read(path / "TOFU_SUMMARY.json")
            for path in tofu_eval_paths(
                "Qwen2.5-1.5B-Instruct",
                "SimNPO",
                "kforge",
                steps,
                "rebuttal_qwen15_v1_EVAL",
            )
        ]
        expected += len(scratch) + len(kforge)
        paired(f"Qwen2.5-1.5B SimNPO S{steps}: scratch vs K-FORGE", scratch, kforge)

    for domain in ("News", "Books"):
        for steps in (50, 100):
            scratch = [read(path / "MUSE_SUMMARY.json") for path in muse_eval_paths(domain, "scratch", steps)]
            kforge = [read(path / "MUSE_SUMMARY.json") for path in muse_eval_paths(domain, "kforge", steps)]
            expected += len(scratch) + len(kforge)
            paired(
                f"MUSE-{domain} SimNPO S{steps}: scratch vs K-FORGE",
                scratch,
                kforge,
                MUSE_METRICS,
            )

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
        MUSE_METRICS,
    )

    for method in ("NPO", "SimNPO"):
        for steps in (50, 100, 250):
            compute_steps = steps + 5
            scratch = llama_fp32_rows(
                method,
                "scratch",
                compute_steps,
                "rebuttal_compute_matched_v3_EVAL_FP32",
            )
            kforge = llama_fp32_rows(
                method,
                "kforge_s06",
                steps,
                "v2lam0p01_corr_EVAL_FP32",
            )
            expected += len(scratch) + len(kforge)
            paired(
                f"Llama {method}: compute-matched scratch S{compute_steps} vs K-FORGE S{steps}",
                scratch,
                kforge,
            )

    for bits in (8, 4):
        left, right = [], []
        pre_scratch, pre_kforge = [], []
        for seed in range(3):
            pre_scratch.append(
                read(
                    ROOT
                    / f"saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget10_NPO_scratch_S50_seed{seed}_week2"
                    / "checkpoint-50/evals/TOFU_SUMMARY.json"
                )
            )
            pre_kforge.append(
                read(
                    ROOT
                    / f"saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget10_NPO_kforge_layer15_down_s0p65_S50_seed{seed}_review_reps"
                    / "checkpoint-50/evals/TOFU_SUMMARY.json"
                )
            )
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
        expected += len(left) + len(right) + len(pre_scratch) + len(pre_kforge)
        paired(f"Llama NPO S50 after {bits}-bit loading: scratch vs K-FORGE", left, right)
        paired(f"Llama NPO scratch S50: pre vs {bits}-bit", pre_scratch, left)
        paired(f"Llama NPO K-FORGE S50: pre vs {bits}-bit", pre_kforge, right)

    pre_scratch = llama_unlearn_rows("NPO", "scratch", 100)
    pre_kforge = llama_unlearn_rows("NPO", "kforge", 50)
    expected += len(pre_scratch) + len(pre_kforge)
    for bits in (8, 4):
        post_scratch = matched_quant_rows("scratch", bits)
        post_kforge = matched_quant_rows("kforge", bits)
        expected += len(post_scratch) + len(post_kforge)
        paired(
            f"Matched NPO after {bits}-bit loading: scratch S100 vs K-FORGE S50",
            post_scratch,
            post_kforge,
        )
        paired(
            f"Matched NPO {bits}-bit recovery: scratch vs K-FORGE",
            changes(pre_scratch, post_scratch),
            changes(pre_kforge, post_kforge),
        )

    pre_scratch = llama_unlearn_rows("SimNPO", "scratch", 50)
    pre_kforge = llama_unlearn_rows("SimNPO", "kforge", 50)
    expected += len(pre_scratch) + len(pre_kforge)
    for bits in (8, 4):
        post_scratch = simnpo_quant_rows("scratch", bits)
        post_kforge = simnpo_quant_rows("kforge", bits)
        expected += len(post_scratch) + len(post_kforge)
        paired(
            f"SimNPO S50 after {bits}-bit loading: scratch vs K-FORGE",
            post_scratch,
            post_kforge,
        )
        paired(
            f"SimNPO S50 {bits}-bit change: scratch vs K-FORGE",
            changes(pre_scratch, post_scratch),
            changes(pre_kforge, post_kforge),
        )

    for method in ("NPO", "SimNPO"):
        for epochs in (1, 3):
            pre_scratch = relearn_rows(method, "scratch", epochs, 0)
            pre_kforge = relearn_rows(method, "kforge", epochs, 0)
            post_scratch = relearn_rows(method, "scratch", epochs, 13 * epochs)
            post_kforge = relearn_rows(method, "kforge", epochs, 13 * epochs)
            expected += len(pre_scratch) + len(pre_kforge) + len(post_scratch) + len(post_kforge)
            paired(
                f"Matched {method} relearning ({epochs} epoch): pre-attack scratch vs K-FORGE",
                pre_scratch,
                pre_kforge,
            )
            paired(
                f"Matched {method} relearning ({epochs} epoch): post-attack scratch vs K-FORGE",
                post_scratch,
                post_kforge,
            )
            paired(
                f"Matched {method} relearning ({epochs} epoch): recovery scratch vs K-FORGE",
                changes(pre_scratch, post_scratch),
                changes(pre_kforge, post_kforge),
            )

    for method in ("NPO", "SimNPO"):
        pre_scratch = gemma_s100_rows(method, "scratch")
        pre_kforge = gemma_s100_rows(method, "kforge")
        expected += len(pre_scratch) + len(pre_kforge)
        for bits in (8, 4):
            post_scratch = gemma_quant_rows(method, "scratch", bits)
            post_kforge = gemma_quant_rows(method, "kforge", bits)
            expected += len(post_scratch) + len(post_kforge)
            paired(
                f"Gemma {method} S100 after {bits}-bit loading: scratch vs K-FORGE",
                post_scratch,
                post_kforge,
            )
            paired(
                f"Gemma {method} S100 {bits}-bit change: scratch vs K-FORGE",
                changes(pre_scratch, post_scratch),
                changes(pre_kforge, post_kforge),
            )

    for method in ("NPO", "SimNPO"):
        for epochs in (1, 3):
            pre_scratch = gemma_relearn_rows(method, "scratch", epochs, 0)
            pre_kforge = gemma_relearn_rows(method, "kforge", epochs, 0)
            post_scratch = gemma_relearn_rows(method, "scratch", epochs, 13 * epochs)
            post_kforge = gemma_relearn_rows(method, "kforge", epochs, 13 * epochs)
            expected += len(pre_scratch) + len(pre_kforge) + len(post_scratch) + len(post_kforge)
            paired(
                f"Gemma matched {method} relearning ({epochs} epoch): pre-attack scratch vs K-FORGE",
                pre_scratch,
                pre_kforge,
            )
            paired(
                f"Gemma matched {method} relearning ({epochs} epoch): post-attack scratch vs K-FORGE",
                post_scratch,
                post_kforge,
            )
            paired(
                f"Gemma matched {method} relearning ({epochs} epoch): recovery scratch vs K-FORGE",
                changes(pre_scratch, post_scratch),
                changes(pre_kforge, post_kforge),
            )

    if args.snapshot_out:
        args.snapshot_out.parent.mkdir(parents=True, exist_ok=True)
        with args.snapshot_out.open("w") as handle:
            json.dump(READ_RECORDS, handle, indent=2, sort_keys=True)
            handle.write("\n")

    if args.check:
        print(f"\nPASS: parsed {expected} expected summaries")


if __name__ == "__main__":
    main()
