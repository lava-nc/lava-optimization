# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import unittest
from lava.lib.optimization.problems.problems import QP
from lava.lib.optimization.solvers.qp.solver import QPSolver
import cProfile as profile


class TestProcessesFloatingPoint(unittest.TestCase):
    def test_3x3_qp(self):
        Q = np.array([[100, 0, 0], [0, 15, 0], [0, 0, 5]])
        p = np.array([[1, 2, 1]]).T
        A = -np.array([[1, 2, 2], [2, 100, 3]])
        k = -np.array([[-50, 50]]).T

        alpha, beta = 0.001, 1
        alpha_d, beta_g = 10000, 10000
        iterations = 400
        problem = QP(Q=Q, p=p, A=A, k=k)
        solver = QPSolver(
            alpha=alpha,
            beta=beta,
            alpha_decay_schedule=alpha_d,
            beta_growth_schedule=beta_g,
        )
        profile.runctx(
            "solver.solve(problem, iterations=iterations)",
            globals(),
            locals(),
            "3x3QPprofiling.file",
        )
        # solver.solve(problem, iterations=iterations)

    def test_4x4_qp(self):
        Q = np.array(
            [[100, 0, 0, 1], [0, 15, 0, 3], [0, 0, 5, 2], [0, 0, 5, 2]]
        )
        p = np.array([[1, 2, 1, 3]]).T
        A = -np.array([[1, 2, 2, 2], [2, 100, 3, 1], [2, 100, 3, 1]])
        k = -np.array([[-50, 50, 100]]).T

        alpha, beta = 0.001, 1
        alpha_d, beta_g = 10000, 10000
        iterations = 400
        problem = QP(Q=Q, p=p, A=A, k=k)
        solver = QPSolver(
            alpha=alpha,
            beta=beta,
            alpha_decay_schedule=alpha_d,
            beta_growth_schedule=beta_g,
        )
        profile.runctx(
            "solver.solve(problem, iterations=iterations)",
            globals(),
            locals(),
            "4x4QPprofiling.file",
        )

    def test_5x5_qp(self):
        Q = np.array(
            [
                [100, 0, 0, 1, 3],
                [0, 15, 0, 3, 1],
                [0, 0, 5, 2, 2],
                [0, 0, 5, 2, 4],
                [0, 0, 5, 2, 2],
            ]
        )
        p = np.array([[1, 2, 1, 3, 3]]).T
        A = -np.array([[1, 2, 2, 2, 1], [2, 100, 3, 1, 3], [2, 100, 3, 1, 5]])
        k = -np.array([[-50, 50, 100]]).T

        alpha, beta = 0.001, 1
        alpha_d, beta_g = 10000, 10000
        iterations = 400
        problem = QP(Q=Q, p=p, A=A, k=k)
        solver = QPSolver(
            alpha=alpha,
            beta=beta,
            alpha_decay_schedule=alpha_d,
            beta_growth_schedule=beta_g,
        )
        profile.runctx(
            "solver.solve(problem, iterations=iterations)",
            globals(),
            locals(),
            "5x5QPprofiling.file",
        )


if __name__ == "__main__":
    unittest.main()
