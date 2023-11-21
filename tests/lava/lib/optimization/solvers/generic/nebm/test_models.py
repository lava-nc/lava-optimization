# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import os
import unittest
import numpy as np

from lava.lib.optimization.problems.problems import QUBO
from lava.lib.optimization.solvers.generic.solver import OptimizationSolver
from lava.lib.optimization.solvers.generic.solver import SolverConfig


def solve_nebm(q, time=100, target=-10, refract=1, counter=0, temp=0):
    qubo_problem = QUBO(q=q)
    solver = OptimizationSolver(qubo_problem)
    config = SolverConfig(
        timeout=time,
        target_cost=target,
        backend="CPU",
        probe_cost=True,
        probe_state=True,
        hyperparameters={
            "refract": refract,
            "refract_counter": counter,
            "neuron_model": "nebm",
            "temperature": temp,
        }
    )
    return solver.solve(config=config)


@unittest.skip("CPU backend of QUBO solver is temporarily deactivated")
class TestPyNEBM(unittest.TestCase):
    def test_boltzmann(self):
        # temp = 0 and delta_e >= 0, p(switch) = 0
        q = np.zeros((10, 10), dtype=int)
        report = solve_nebm(q)
        self.assertEqual(report.state_timeseries.sum(), 0)
        # temp = 0 and delta_e < 0, p(switch) = 1
        q[np.diag_indices(10)] = -1
        report = solve_nebm(q)
        self.assertEqual(report.best_state.sum(), 10)
        # With temp > 0 and delta_e = 0, p(switch) = 0.5
        report = solve_nebm(q, temp=10, target=-100)
        self.assertAlmostEqual(report.state_timeseries.mean(), 0.5, delta=0.1)
        # With delta_e != 0, p(switch) increases as temp increases
        q[np.diag_indices(10)] = 1
        report = solve_nebm(q, temp=1)
        prob_switch_at_t1 = report.state_timeseries.mean()
        report = solve_nebm(q, temp=10)
        prob_switch_at_t10 = report.state_timeseries.mean()
        self.assertLess(prob_switch_at_t1, prob_switch_at_t10)
        # With temp > 0, p(switch) decreases as delta_e increases
        q[np.diag_indices(10)] = np.arange(-5, 5, 1)
        report = solve_nebm(q, temp=1, target=-100, time=1000)
        prob_on = report.state_timeseries.mean(axis=0)
        diff_prob_on = np.diff(prob_on)
        self.assertTrue(np.all(diff_prob_on <= 0))

    def test_refract_oscillatory(self):
        # Check that the model oscillates with no tie-breaking
        q = -2 * np.eye(10, dtype=int)
        q[1:, 1:] += 1
        report = solve_nebm(q)
        solution = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        self.assertEqual(-2 * solution.sum(), report.best_cost)
        np.testing.assert_array_equal(report.best_state, solution)

        actual_cost = report.cost_timeseries.flatten()[:10]
        expected_cost = np.array([0, 0, 61, 61, -2, -2, 61, 61, -2, -2])
        np.testing.assert_array_equal(actual_cost, expected_cost)

    def test_refract_descend(self):
        # Check that the model descends cost with varied refract
        q = -2 * np.eye(10, dtype=int) + 1
        report = solve_nebm(q, refract=np.array(range(1, 11)))
        solution = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

        self.assertEqual(-1 * solution.sum(), report.best_cost)
        np.testing.assert_array_equal(report.best_state, solution)

        actual_cost = report.cost_timeseries.flatten()
        actual_cost = actual_cost[:report.best_timestep + 1]
        expected_cost = np.array([0, 0, 80, 80, 63, 48, 35, 24, 15,
                                  8, 3, 0, -1])
        np.testing.assert_array_equal(actual_cost, expected_cost)

    def test_refract_counter_descend(self):
        # Check that the model descends cost with varied refract_counter
        q = -1 * np.eye(10, dtype=int)
        report = solve_nebm(q, counter=np.array(range(10)))
        solution = np.ones(10, dtype=int)

        self.assertEqual(-1 * solution.sum(), report.best_cost)
        np.testing.assert_array_equal(report.best_state, solution)

        actual_cost = report.cost_timeseries.flatten()
        actual_cost = actual_cost[:report.best_timestep + 1]
        expected_cost = np.array([0, 0, -1, -2, -3, -4, -5, -6, -7,
                                  -8, -9, -10])
        np.testing.assert_array_equal(actual_cost, expected_cost)

    def test_robust_descend(self):
        # Check that the model descends cost with varied refract_counter
        np.random.seed(42)

        q = 1 * (np.random.rand(10, 10) < 0.1)
        q -= np.tril(q, 0)
        q += np.triu(q, 0).T
        q *= (1 - np.eye(10, dtype=int))
        q -= 1 * np.eye(10, dtype=int)

        report = solve_nebm(q, time=10, refract=10,
                            counter=np.array(range(0, 10)),
                            temp=0, target=-10)

        diff_cost = np.diff(report.cost_timeseries)
        self.assertTrue(np.all(diff_cost <= 0))


if __name__ == "__main__":
    unittest.main()
