# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import unittest

import numpy as np
from lava.lib.optimization.problems.problems import QUBO
from lava.lib.optimization.solvers.generic.solver import (
    OptimizationSolver,
    SolverConfig,
)


@unittest.skip("CPU backend of QUBO solver temporarily disabled.")
class TestParallelOptimizationSolver(unittest.TestCase):
    def test_parallel_run(self):
        q = np.array(
            [[-5, 2, 4, 0], [2, -3, 1, 0], [4, 1, -8, 5], [0, 0, 5, -6]]
        )
        problem = QUBO(q)
        solution = np.asarray([1, 0, 0, 1]).astype(int)
        solver = OptimizationSolver(problem=problem)
        solution_cost = solution @ q @ solution

        np.random.seed(2)

        config = SolverConfig(
            timeout=50,
            target_cost=-11,
            backend="CPU",
            hyperparameters=[
                {
                    "neuron_model": "scif",
                    "noise_amplitude": i,
                    "noise_precision": 5,
                    "sustained_on_tau": -3,
                }
                for i in range(1, 6)
            ]
            + [{"neuron_model": "nebm"}],
        )
        report = solver.solve(config=config)
        self.assertEqual(report.best_cost, solution_cost)


def solve_workload(
    q, reference_solution, noise_precision=5, noise_amplitude=1, on_tau=-3
):
    expected_cost = reference_solution @ q @ reference_solution
    problem = QUBO(q)
    np.random.seed(2)
    solver = OptimizationSolver(problem)
    report = solver.solve(
        config=SolverConfig(
            timeout=20000,
            target_cost=expected_cost,
            hyperparameters=[
                {
                    "neuron_model": "scif",
                    "noise_amplitude": noise_amplitude,
                    "noise_precision": noise_precision,
                    "sustained_on_tau": -on_tau_i,
                }
                for on_tau_i in range(1, 8)
            ],
        )
    )
    cost = report.best_state @ q @ report.best_state
    return report.best_state, cost, expected_cost


@unittest.skip("CPU backend of QUBO solver temporarily disabled.")
class TestWorkloads(unittest.TestCase):
    def test_solve_polynomial_minimization(self):
        """Polynomial minimization with y=-5x_1 -3x_2 -8x_3 -6x_4 +
        4x_1x_2+8x_1x_3+2x_2x_3+10x_3x_4
        """
        q = np.array(
            [[-5, 2, 4, 0], [2, -3, 1, 0], [4, 1, -8, 5], [0, 0, 5, -6]]
        )
        reference_solution = np.asarray([1, 0, 0, 1]).astype(int)
        solution, cost, expected_cost = solve_workload(
            q, reference_solution, noise_precision=5
        )
        self.assertEqual(cost, expected_cost)

    def test_solve_set_packing(self):
        q = -np.array(
            [[1, -3, -3, -3], [-3, 1, 0, 0], [-3, 0, 1, -3], [-3, 0, -3, 1]]
        )

        reference_solution = np.zeros(4)
        np.put(reference_solution, [1, 2], 1)
        solution, cost, expected_cost = solve_workload(
            q, reference_solution, noise_precision=5
        )
        self.assertEqual(cost, expected_cost)

    def test_solve_max_cut_problem(self):
        """Max-Cut Problem"""
        q = -np.array(
            [
                [2, -1, -1, 0, 0],
                [-1, 2, 0, -1, 0],
                [-1, 0, 3, -1, -1],
                [0, -1, -1, 3, -1],
                [0, 0, -1, -1, 2],
            ]
        )
        reference_solution = np.zeros(5)
        np.put(reference_solution, [1, 2], 1)
        solution, cost, expected_cost = solve_workload(
            q, reference_solution, noise_precision=5
        )
        self.assertEqual(cost, expected_cost)

    def test_solve_set_partitioning(self):
        q = np.array(
            [
                [-17, 10, 10, 10, 0, 20],
                [10, -18, 10, 10, 10, 20],
                [10, 10, -29, 10, 20, 20],
                [10, 10, 10, -19, 10, 10],
                [0, 10, 20, 10, -17, 10],
                [20, 20, 20, 10, 10, -28],
            ]
        )
        reference_solution = np.zeros(6)
        np.put(reference_solution, [0, 4], 1)
        solution, cost, expected_cost = solve_workload(
            q, reference_solution, noise_precision=6, noise_amplitude=1
        )
        self.assertEqual(cost, expected_cost)

    def test_solve_map_coloring(self):
        q = np.array(
            [
                [-4, 4, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
                [4, -4, 4, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
                [4, 4, -4, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2],
                [2, 0, 0, -4, 4, 4, 2, 0, 0, 2, 0, 0, 2, 0, 0],
                [0, 2, 0, 4, -4, 4, 0, 2, 0, 0, 2, 0, 0, 2, 0],
                [0, 0, 2, 4, 4, -4, 0, 0, 2, 0, 0, 2, 0, 0, 2],
                [0, 0, 0, 2, 0, 0, -4, 4, 4, 2, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 2, 0, 4, -4, 4, 0, 2, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 2, 4, 4, -4, 0, 0, 2, 0, 0, 0],
                [0, 0, 0, 2, 0, 0, 2, 0, 0, -4, 4, 4, 2, 0, 0],
                [0, 0, 0, 0, 2, 0, 0, 2, 0, 4, -4, 4, 0, 2, 0],
                [0, 0, 0, 0, 0, 2, 0, 0, 2, 4, 4, -4, 0, 0, 2],
                [2, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, -4, 4, 4],
                [0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 4, -4, 4],
                [0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 2, 4, 4, -4],
            ]
        )
        reference_solution = np.zeros(15)
        np.put(reference_solution, [1, 3, 8, 10, 14], 1)
        solution, cost, expected_cost = solve_workload(
            q,
            reference_solution,
            noise_precision=5,
            noise_amplitude=1,
            on_tau=-1,
        )
        self.assertEqual(cost, expected_cost)


if __name__ == "__main__":
    unittest.main()
