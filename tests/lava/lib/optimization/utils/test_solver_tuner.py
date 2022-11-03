# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

import numpy as np

from lava.lib.optimization.problems.problems import QUBO
from lava.lib.optimization.solvers.generic.solver import OptimizationSolver
from lava.lib.optimization.utils.solver_tuner import SolverTuner


class TestSolverTuner(unittest.TestCase):
    """Unit tests for SolverTuner class."""

    def setUp(self) -> None:
        params_grid = {
            "step_size": (1, 20),
            "noise_amplitude": (0, 10),
        }
        self.solver_tuner = SolverTuner(params_grid=params_grid)

    def test_create_obj(self):
        """Tests correct instantiation of SolverTuner object."""
        self.assertIsInstance(self.solver_tuner, SolverTuner)

    def test_tune_success(self):
        """Tests the correct set of hyper-parameters is found, for a known
        problem."""
        q = np.asarray([[-5, 2, 4, 0],
                        [2, -3, 1, 0],
                        [4, 1, -8, 5],
                        [0, 0, 5, -6]])

        qubo_problem = QUBO(q=q)
        solver = OptimizationSolver(qubo_problem)
        optimal_cost = -11

        params = {"timeout": 1000,
                  "target_cost": optimal_cost,
                  "backend": "CPU"}

        def fitness(cost, step_to_sol):
            return - step_to_sol if cost == optimal_cost else -float('inf')

        f_target = -200
        seed = 2

        hyperparams, success = self.solver_tuner.tune(solver=solver,
                                                      solver_params=params,
                                                      fitness_fn=fitness,
                                                      fitness_target=f_target,
                                                      seed=seed)
        correct_best_hyperparams = {
            "step_size": 20,
            "noise_amplitude": 10
        }

        self.assertEqual(hyperparams, correct_best_hyperparams)
        self.assertTrue(success)


if __name__ == '__main__':
    unittest.main()
