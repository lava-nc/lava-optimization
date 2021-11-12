# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
from lava.lib.optimization.problems.problems import QP
from lava.lib.optimization.solvers.qp.solver import QPSolver


def main():
    """
    The QP cost function is 1/2*x^T*Q*x + p^T*x
    The inequality constraints are of the form A*x<=k
    The equality constraints are A_eq*x=k_eq.
    """
    Q = np.array([[100, 0, 0], [0, 15, 0], [0, 0, 5]])
    p = np.array([[1, 2, 1]]).T
    A = -np.array([[1, 2, 2], [2, 100, 3]])
    k = -np.array([[-50, 50]]).T

    alpha, beta = 0.001, 1
    alpha_d, beta_g = 10000, 10000
    iterations = 400
    problem = QP(Q, p, A, k)
    solver = QPSolver(
        alpha=alpha,
        beta=beta,
        alpha_decay_schedule=alpha_d,
        beta_growth_schedule=beta_g,
    )
    solver.solve(problem, iterations=iterations)


if __name__ == "__main__":
    main()
