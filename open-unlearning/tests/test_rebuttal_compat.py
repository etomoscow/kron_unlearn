import csv
import math
import unittest
from types import SimpleNamespace

import torch

from data.utils import _as_token_id_list
from evals.metrics.utils import evaluate_probability
from scripts.estimate_kforge_compute import estimate_setup_flops, estimate_step_flops
from scripts.summarize_rebuttal_additions import ROOT, muse_eval_paths, read, tofu_eval_paths


class _BFloat16Model:
    device = torch.device("cpu")

    def __call__(self, **_batch):
        torch.manual_seed(0)
        return SimpleNamespace(logits=torch.randn(2, 4, 7, dtype=torch.bfloat16))


class RebuttalCompatibilityTest(unittest.TestCase):
    def test_rebuttal_eval_paths_encode_model_and_seed(self):
        paths = tofu_eval_paths(
            "Qwen2.5-1.5B-Instruct",
            "SimNPO",
            "scratch",
            50,
            "rebuttal_qwen15_v1_EVAL",
            seeds=range(3),
        )

        self.assertEqual(len(paths), 3)
        self.assertEqual(
            paths[-1].name,
            "tofu_Qwen2.5-1.5B-Instruct_forget10_SimNPO_scratch_S50_seed2_rebuttal_qwen15_v1_EVAL",
        )

    def test_snapshot_supplies_summary_without_raw_file(self):
        path = ROOT / "saves/eval/not-on-disk/TOFU_SUMMARY.json"
        key = path.relative_to(ROOT).as_posix()
        expected = {"forget_Q_A_Prob": 0.25}

        self.assertEqual(read(path, snapshot={key: expected}), expected)

    def test_muse_eval_paths_encode_domain_and_seed(self):
        paths = muse_eval_paths("Books", "kforge", 50, seeds=range(3))

        self.assertEqual(len(paths), 3)
        self.assertEqual(
            paths[-1].name,
            "muse_Llama-2-7b-hf_Books_SimNPO_kforge_S50_seed2_rebuttal_muse_books_v2",
        )

    def test_compute_estimate_includes_prompt_and_answer_input_tokens(self):
        estimate = estimate_setup_flops(
            parameters=1_235_814_400,
            calibration_input_tokens=47_727.9,
            factor_token_rows=18_614,
            rows=2_048,
            columns=8_192,
        )

        self.assertAlmostEqual(estimate["calibration_flops"], 3.5389695661056e14)
        self.assertEqual(estimate["factor_accumulation_flops"], 2_654_474_338_304)
        self.assertEqual(estimate["dense_flops"], 5_583_457_484_800)
        self.assertAlmostEqual(estimate["total_flops"], 3.62134888433664e14)
        self.assertAlmostEqual(estimate["calibration_fraction"], 0.9772517587)

        steps = estimate_step_flops(
            parameters=1_235_814_400,
            forget_input_tokens=3_042.32,
            retain_input_tokens=2_923.67,
        )
        self.assertAlmostEqual(steps["simnpo_flops"], 4.4237138113536e13)
        self.assertAlmostEqual(steps["npo_flops"], 5.1756623844352e13)

        padded_setup = estimate_setup_flops(
            parameters=1_235_814_400,
            calibration_input_tokens=58_855.5,
            factor_token_rows=18_614,
            rows=2_048,
            columns=8_192,
        )
        padded_steps = estimate_step_flops(
            parameters=1_235_814_400,
            forget_input_tokens=3_544.1,
            retain_input_tokens=3_392.8,
        )
        self.assertLess(
            padded_setup["total_flops"] / padded_steps["simnpo_flops"], 10
        )
        self.assertLess(padded_setup["total_flops"] / padded_steps["npo_flops"], 10)

    def test_compute_csv_matches_the_released_calculator(self):
        path = ROOT / "saves/figures/kforge_corrected/kforge_compute_overhead_data.csv"
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))

        parameters = {
            "Llama-3.2-1B": 1_235_814_400,
            "Llama-3.2-3B": 3_212_749_824,
        }
        for row in rows:
            n, m = (int(value) for value in row["edited_matrix"].split("x"))
            estimate = estimate_setup_flops(
                parameters=parameters[row["model"]],
                calibration_input_tokens=float(row["calibration_input_tokens_est"]),
                factor_token_rows=float(row["factor_token_rows"]),
                rows=n,
                columns=m,
            )
            self.assertAlmostEqual(
                float(row["estimated_flops"]), estimate["total_flops"]
            )
            end_to_end = float(row["end_to_end_upper_s"])
            self.assertGreater(end_to_end, float(row["estimator_edit_time_s"]))
            for method in ("npo", "simnpo"):
                s50 = row[f"{method}_s50_train_runtime_mean_s"]
                if not s50:
                    continue
                self.assertAlmostEqual(
                    float(row[f"setup_vs_{method}_s50_pct"]),
                    100 * end_to_end / float(s50),
                    places=3,
                )
                self.assertGreaterEqual(
                    float(row[f"{method}_matched_scratch_runtime_mean_s"]),
                    float(s50) + end_to_end,
                )
                self.assertGreaterEqual(
                    float(row[f"{method}_wall_margin_min_s"]), 0
                )

    def test_token_ids_are_normalized_to_a_flat_list(self):
        self.assertEqual(_as_token_id_list({"input_ids": [[1, 2, 3]]}), [1, 2, 3])
        self.assertEqual(_as_token_id_list(torch.tensor([[4, 5, 6]])), [4, 5, 6])

    def test_bfloat16_probability_results_are_python_floats(self):
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]]),
            "labels": torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]]),
        }

        results = evaluate_probability(_BFloat16Model(), batch)

        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIsInstance(result["prob"], float)
            self.assertIsInstance(result["avg_loss"], float)
            self.assertTrue(math.isfinite(result["prob"]))
            self.assertTrue(math.isfinite(result["avg_loss"]))


if __name__ == "__main__":
    unittest.main()
