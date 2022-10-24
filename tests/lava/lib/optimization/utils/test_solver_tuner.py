# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

import numpy as np

from lava.lib.optimization.problems.problems import QUBO, CSP
from lava.lib.optimization.solvers.generic.solver import OptimizationSolver
from lava.lib.optimization.utils.solver_tuner import SolverTuner


def prepare_problem_and_solver():
    # Define the QUBO matrix
    q = np.asarray([[-5, 2, 4, 0],
                    [2, -3, 1, 0],
                    [4, 1, -8, 5],
                    [0, 0, 5, -6]])

    # Instantiate the QUBO problem
    qubo_problem = QUBO(q=q)

    # Instantiate a constraint optimization solver for this workload
    solver = OptimizationSolver(qubo_problem)
    return solver


class TestSolverTuner(unittest.TestCase):

    def setUp(self) -> None:
        params_grid = {
            "step_size": (1, 2),
            "noise_amplitude": (4, 5),
            "var_comm_rate": (8, 9)
        }
        self.solver_tuner = SolverTuner(params_grid=params_grid)

    def test_create_obj(self):
        self.assertIsInstance(self.solver_tuner, SolverTuner)

    def test_tune_success(self):
        solver = prepare_problem_and_solver()

        params = {"timeout": 1000,
                  "target_cost": -11,
                  "backend": "CPU"}

        stopping_condition = lambda best_cost, best_to_sol: best_cost < -6

        hyperparams, succeeded = self.solver_tuner.tune(solver=solver,
                                                        solver_parameters=params,
                                                        stopping_condition=stopping_condition)

        self.assertIsInstance(solver, OptimizationSolver)
        self.assertTrue(succeeded)


if __name__ == '__main__':
    unittest.main()
