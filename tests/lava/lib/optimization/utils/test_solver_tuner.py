# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import typing as ty

import numpy as np

from lava.lib.optimization.problems.problems import QUBO, CSP
from lava.lib.optimization.solvers.generic.solver import OptimizationSolver
from lava.lib.optimization.utils.solver_tuner import SolverTuner


def prepare_problem():
    # Define the QUBO matrix
    q = np.asarray([[-5, 2, 4, 0],
                    [2, -3, 1, 0],
                    [4, 1, -8, 5],
                    [0, 0, 5, -6]])

    # Instantiate the QUBO problem
    qubo_problem = QUBO(q=q)

    return qubo_problem


class TestSolverTuner(unittest.TestCase):

    def setUp(self) -> None:
        params_grid = {
            "step_size": (1, 2),
            "noise_amplitude": (3, 4, 5, 6),
        }
        self.solver_tuner = SolverTuner(params_grid=params_grid)

    def test_create_obj(self):
        self.assertIsInstance(self.solver_tuner, SolverTuner)

    def test_tune_success(self):
        problem = prepare_problem()

        params = {"timeout": 1000,
                  "target_cost": -11,
                  "backend": "CPU"}

        def stop(best_cost, best_to_sol):
            return best_cost < -6

        hyperparams, success = self.solver_tuner.tune(problem=problem,
                                                      solver_parameters=params,
                                                      stopping_condition=stop)

        self.assertIsInstance(hyperparams, ty.Dict)
        self.assertTrue(success)


if __name__ == '__main__':
    unittest.main()
