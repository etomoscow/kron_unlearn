import unittest

import torch

from trainer.unlearn.kforge import KFORGE


def _random_cholesky(size):
    matrix = torch.randn(size, size, dtype=torch.float64)
    return torch.linalg.cholesky(
        matrix @ matrix.T + 0.5 * torch.eye(size, dtype=torch.float64)
    )


class KFORGEWienerTest(unittest.TestCase):
    def test_full_rank_wiener_matches_direct_quadratic_solve(self):
        torch.manual_seed(0)
        n, m = 3, 4
        weight = torch.randn(n, m, dtype=torch.float64)

        l_af, l_bf = _random_cholesky(m), _random_cholesky(n)
        l_ar, l_br = _random_cholesky(m), _random_cholesky(n)
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

    def test_zero_penalty_rank_r_matches_eckart_young_solution(self):
        torch.manual_seed(17)
        n, m, rank = 4, 5, 2
        weight = torch.randn(n, m, dtype=torch.float64)
        l_af, l_bf = _random_cholesky(m), _random_cholesky(n)
        l_ar, l_br = _random_cholesky(m), _random_cholesky(n)

        solver = object.__new__(KFORGE)
        solver.rank = rank
        solver.lambda_tradeoff = 0.0
        actual, _ = solver._compute_module_delta_wiener_v2(
            weight, l_af, l_bf, l_ar, l_br
        )

        k_a = torch.linalg.solve_triangular(l_ar, l_af, upper=False)
        k_b = torch.linalg.solve_triangular(l_br, l_bf, upper=False)
        u_a, s_a, vh_a = torch.linalg.svd(k_a, full_matrices=False)
        u_b, s_b, vh_b = torch.linalg.svd(k_b, full_matrices=False)
        r_matrix = vh_b @ (l_bf.T @ weight @ l_af) @ vh_a.T
        u_r, s_r, vh_r = torch.linalg.svd(r_matrix, full_matrices=False)
        y_r = -((u_r[:, :rank] * s_r[:rank]) @ vh_r[:rank])
        h_r = (y_r / s_b[:, None]) / s_a[None, :]
        expected = torch.linalg.solve_triangular(
            l_br.T, u_b @ h_r, upper=True
        )
        expected = torch.linalg.solve_triangular(
            l_ar.T, (expected @ u_a.T).T, upper=True
        ).T

        forget_residual = l_bf.T @ (weight + actual) @ l_af
        expected_objective = s_r[rank:].square().sum()
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(
            forget_residual.square().sum(),
            expected_objective,
            rtol=1e-10,
            atol=1e-10,
        )
        self.assertLessEqual(
            int(torch.linalg.matrix_rank(actual, rtol=1e-10, atol=1e-10)),
            rank,
        )


if __name__ == "__main__":
    unittest.main()
