# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

from lava.lib.optimization.solvers.generic.solver import \
    OptimizationSolver


class TestOptimizationSolver(unittest.TestCase):
    def setUp(self) -> None:
        self.solver = OptimizationSolver()
        self.mock_problem = None  # todo mock problem
        self.mock_solution = None  # todo mock solution

    def test_create_obj(self):
        self.assertIsInstance(self.solver, OptimizationSolver)

    @unittest.skip('WIP')
    def test_solve_method(self):
        solution = self.solver.solve(self.mock_problem, timeout=3000)
        self.assertEqual(solution, self.mock_solution)

    @unittest.skip('WIP')
    def test_init_ceates_optimization_solver_process(self):
        pass

    @unittest.skip('WIP')
    def test_solve_method_creates_solvernet_process(self):
        pass

    @unittest.skip('WIP')
    def test_run_network_is_called_with_timeout(self):
        pass

    @unittest.skip('WIP')
    def test_timeout_in_solve_method(self):
        pass

    @unittest.skip('WIP')
    def test_profiling_true_on_solve_call(self):
        pass

    @unittest.skip('WIP')
    def test_profiling_false_on_solve_call(self):
        pass


if __name__ == '__main__':
    unittest.main()
