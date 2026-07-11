import math
import unittest
from types import SimpleNamespace

import torch

from data.utils import _as_token_id_list
from evals.metrics.utils import evaluate_probability
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
