# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
from lava.lib.optimization.problems.problems import QUBO
import numpy as np
from lava.lib.optimization.solvers.generic.solver import SolverReport
from lava.lib.optimization.utils.state_analysis import StateAnalysis
import os


def prepare_problem_and_report():
    np.random.seed(0)
    size = 5
    timeout = 50
    problem = QUBO(
        q=np.random.randint(0, 20, size=(size, size), dtype=np.int32)
    )
    states = np.random.randint(0, 2, size=(timeout, size), dtype=np.int32)
    costs = list(map(problem.evaluate_cost, states))
    report = SolverReport(cost_timeseries=costs, state_timeseries=states)
    return problem, report


class TestStateAnalysis(unittest.TestCase):
    def setUp(self) -> None:
        self.problem, self.report = prepare_problem_and_report()
        self.analysis = StateAnalysis(problem=self.problem)

    def test_create_obj(self) -> None:
        self.assertIsInstance(self.analysis, StateAnalysis)

    def test_plot_cost_timeseries(self) -> None:
        filename = "plot_cost_timeseries.png"
        self.analysis.plot_cost_timeseries(
            report=self.report, filename=filename
        )
        self.assertTrue(os.path.exists(filename))
        os.remove(filename)

    def test_plot_min_cost_timeseries(self) -> None:
        filename = "plot_min_cost_timseries.png"
        self.analysis.plot_min_cost_timeseries(
            report=self.report, filename=filename
        )
        self.assertTrue(os.path.exists(filename))
        os.remove(filename)

    def test_plot_cost_distribution(self) -> None:
        filename = "plot_cost_distribution.png"
        self.analysis.plot_cost_distribution(
            report=self.report, filename=filename
        )
        self.assertTrue(os.path.exists(filename))
        os.remove(filename)

    def test_plot_delta_cost_distribution(self) -> None:
        filename = "plot_delta_cost_distribution.png"
        self.analysis.plot_delta_cost_distribution(
            report=self.report, filename=filename
        )
        self.assertTrue(os.path.exists(filename))
        os.remove(filename)

    def test_plot_unique_state_visits(self) -> None:
        filename = "plot_unique_state_visits.png"
        self.analysis.plot_unique_state_visits(
            report=self.report, filename=filename
        )
        self.assertTrue(os.path.exists(filename))
        os.remove(filename)

        self.assertTrue(os.path.exists(filename))
        os.remove(filename)


if __name__ == "__main__":
    unittest.main()
