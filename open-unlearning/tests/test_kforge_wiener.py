import unittest

import torch

from trainer.unlearn.kforge import KFORGE


class KFORGEWienerTest(unittest.TestCase):
    def test_full_rank_wiener_matches_direct_quadratic_solve(self):
        torch.manual_seed(0)
        n, m = 3, 4
        weight = torch.randn(n, m, dtype=torch.float64)

        def chol(size):
            matrix = torch.randn(size, size, dtype=torch.float64)
            return torch.linalg.cholesky(
                matrix @ matrix.T + 0.5 * torch.eye(size, dtype=torch.float64)
            )

        l_af, l_bf, l_ar, l_br = chol(m), chol(n), chol(m), chol(n)
        solver = object.__new__(KFORGE)
        solver.rank = min(n, m)
        solver.lambda_tradeoff = 0.7
        actual, _ = solver._compute_module_delta_wiener_v2(
            weight, l_af, l_bf, l_ar, l_br
        )

        def objective(flat_delta):
            delta = flat_delta.reshape(n, m)
            forget = l_bf.T @ (weight + delta) @ l_af
            retain = l_br.T @ delta @ l_ar
            return (
                forget.square().sum()
                + solver.lambda_tradeoff * retain.square().sum()
            )

        zero = torch.zeros(n * m, dtype=torch.float64, requires_grad=True)
        gradient = torch.autograd.grad(objective(zero), zero, create_graph=True)[0]
        hessian = torch.autograd.functional.hessian(objective, zero)
        expected = torch.linalg.solve(hessian, -gradient).reshape(n, m)

        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
