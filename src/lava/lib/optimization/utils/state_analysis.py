# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/


from lava.lib.optimization.problems.problems import QUBO
from lava.lib.optimization.solvers.generic.solver import SolverReport

from matplotlib import pyplot as plt
import warnings
import numpy as np


class StateAnalysis:
    def __init__(self, problem: QUBO) -> None:
        self.problem = problem

    def _line_plot(
        self,
        x: np.ndarray = None,
        y: np.ndarray = None,
        title: str = None,
    ) -> plt.Axes:
        ax = plt.figure(figsize=(10, 10)).add_subplot(1, 1, 1)
        ax.plot(x, y)
        ax.set_title(title)
        return ax

    def _hist_plot(
        self,
        y: np.ndarray = None,
        title: str = None,
    ) -> plt.Axes:
        ax = plt.figure(figsize=(10, 10)).add_subplot(1, 1, 1)
        ax.hist(y)
        ax.set_title(title)
        return ax

    def _extract_costs_or_warn(self, report: SolverReport) -> np.ndarray:
        if report.cost_timeseries is not None:
            return np.array(report.cost_timeseries)
        elif self.problem is not None and report.state_timeseries is not None:
            return np.array(
                map(self.problem.evaluate_cost, report.state_timeseries)
            )
        else:
            warnings.warn("Cost timeseries is not available.")

    def _extract_states_or_warn(self, report: SolverReport) -> np.ndarray:
        if report.state_timeseries is not None:
            return np.array(report.state_timeseries)
        else:
            warnings.warn("State timeseries is not available.")

    def _show_or_save(self, ax: plt.Axes, filename: str = None) -> None:
        if filename is None:
            ax.get_figure().show()
        else:
            ax.get_figure().savefig(filename)

    def plot_cost_timeseries(
        self, report: SolverReport, filename: str = None
    ) -> None:
        cost = self._extract_costs_or_warn(report)
        if cost is None:
            return
        ax = self._line_plot(x=list(range(len(cost))), y=cost, title="Cost")
        self._show_or_save(ax, filename)

    def plot_min_cost_timeseries(
        self, report: SolverReport, filename: str = None
    ) -> None:
        cost = self._extract_costs_or_warn(report)
        if cost is None:
            return
        min_cost = np.minimum.accumulate(cost, axis=0)
        ax = self._line_plot(
            x=list(range(len(min_cost))), y=min_cost, title="Minimum Cost"
        )
        self._show_or_save(ax, filename)

    def plot_cost_distribution(
        self, report: SolverReport, filename: str = None
    ) -> None:
        cost = self._extract_costs_or_warn(report)
        if cost is None:
            return
        ax = self._hist_plot(y=cost, title="Cost Distribution")
        self._show_or_save(ax, filename)

    def plot_delta_cost_distribution(
        self, report: SolverReport, filename: str = None
    ) -> None:
        cost = self._extract_costs_or_warn(report)
        if cost is None:
            return
        delta_cost = cost[1:] - cost[:-1]
        ax = self._hist_plot(y=delta_cost, title="Delta-Cost Distribution")
        self._show_or_save(ax, filename)

    def plot_unique_state_visits(
        self, report: SolverReport, filename: str = None
    ) -> None:
        states = self._extract_states_or_warn(report)
        if states is None:
            return
        num_unique_states = list(
            map(
                lambda i: np.unique(states[:i], axis=0).shape[0],
                range(len(states)),
            )
        )
        ax = self._line_plot(
            x=list(range(len(states))),
            y=num_unique_states,
            title="Cumulative Number of Unique States Visited",
        )
        self._show_or_save(ax, filename)

    def plot_successive_states_distance(self, filename: str = None) -> None:
        return NotImplementedError
