# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import os
import unittest

import numpy as np
from lava.lib.optimization.problems.problems import QUBO
from lava.lib.optimization.solvers.generic.solver import SolverReport
from lava.lib.optimization.utils.report_analyzer import ReportAnalyzer


def prepare_problem_and_report():
    np.random.seed(0)
    size = 5
    timeout = 100
    q = np.random.randint(0, 20, size=(size, size), dtype=np.int32)
    q_symm = ((q + q.T) / 2).astype(int)
    problem = QUBO(q=q_symm)
    states = np.random.randint(0, 2, size=(timeout, size), dtype=np.int32)
    costs = list(map(problem.evaluate_cost, states))
    report = SolverReport(
        problem=problem, cost_timeseries=costs, state_timeseries=states
    )
    return report


class TestReportAnalyzer(unittest.TestCase):
    def setUp(self) -> None:
        self.report = prepare_problem_and_report()
        self.analysis = ReportAnalyzer(report=self.report)

    def test_create_obj(self) -> None:
        self.assertIsInstance(self.analysis, ReportAnalyzer)

    def test_plot_cost_timeseries(self) -> None:
        filename = "plot_cost_timeseries.png"
        self.analysis.plot_cost_timeseries(filename=filename)
        self.assertTrue(os.path.exists(filename))
        os.remove(filename)

    def test_plot_min_cost_timeseries(self) -> None:
        filename = "plot_min_cost_timseries.png"
        self.analysis.plot_min_cost_timeseries(filename=filename)
        self.assertTrue(os.path.exists(filename))
        os.remove(filename)

    def test_plot_cost_distribution(self) -> None:
        filename = "plot_cost_distribution.png"
        self.analysis.plot_cost_distribution(filename=filename)
        self.assertTrue(os.path.exists(filename))
        os.remove(filename)

    def test_plot_delta_cost_distribution(self) -> None:
        filename = "plot_delta_cost_distribution.png"
        self.analysis.plot_delta_cost_distribution(filename=filename)
        self.assertTrue(os.path.exists(filename))
        os.remove(filename)

    def test_plot_num_visited_states(self) -> None:
        filename = "plot_num_visited_states.png"
        self.analysis.plot_num_visited_states(filename=filename)
        self.assertTrue(os.path.exists(filename))
        os.remove(filename)

    def test_plot_successive_states_distance(self) -> None:
        filename = "plot_successive_states_distance.png"
        self.analysis.plot_successive_states_distance(filename=filename)
        self.assertTrue(os.path.exists(filename))
        os.remove(filename)

    def test_plot_state_timeseries(self) -> None:
        filename = "plot_state_timeseries.png"
        self.analysis.plot_state_timeseries(filename=filename)
        self.assertTrue(os.path.exists(filename))
        os.remove(filename)

    def test_plot_state_analysis_summary(self) -> None:
        filename = "plot_state_analysis_summary.png"
        self.analysis.plot_state_analysis_summary(filename=filename)
        self.assertTrue(os.path.exists(filename))
        os.remove(filename)


if __name__ == "__main__":
    unittest.main()
