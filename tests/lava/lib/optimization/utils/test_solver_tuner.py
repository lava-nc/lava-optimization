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
        self.solver_tuner = SolverTuner(step_range=(1, 2),
                                        noise_range=(4, 5),
                                        steps_to_fire_range=(8, 9))

    def test_create_obj(self):
        self.assertIsInstance(self.solver_tuner, SolverTuner)

    def test_problem_type_check(self):
        constraints = [
            (0, 1, np.logical_not(np.eye(5))),
            (1, 2, np.eye(5, 4)),
        ]
        problem = CSP(domains=[5, 5, 4], constraints=constraints)
        solver = OptimizationSolver(problem)
        args = (solver, {}, 0)
        self.assertRaises(ValueError, self.solver_tuner.tune, *args)

    def test_tune_success(self):
        solver = prepare_problem_and_solver()

        params = {"timeout": 1000,
                  "target_cost": -11,
                  "backend": "CPU"}

        target_cost = -11

        solver, succeeded = self.solver_tuner.tune(solver=solver,
                                                   solver_parameters=params,
                                                   target_cost=target_cost)

        self.assertIsInstance(solver, OptimizationSolver)
        self.assertIsInstance(succeeded, bool)
        self.assertTrue(succeeded)


if __name__ == '__main__':
    unittest.main()
