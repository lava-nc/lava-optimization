# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/


import typing as ty
import warnings

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from lava.lib.optimization.problems.problems import QUBO
from lava.lib.optimization.solvers.generic.solver import SolverReport

sns.set_style("whitegrid")


class StateAnalysis:
    def __init__(self, problem: QUBO) -> None:
        self.problem = problem

    def _line_plot(
        self,
        x: np.ndarray = None,
        y: np.ndarray = None,
        xlabel: str = "",
        ylabel: str = "",
        xlim: ty.Tuple[float, float] = None,
        ylim: ty.Tuple[float, float] = None,
        ax: plt.Axes = None,
    ) -> Figure:
        if ax is None:
            ax = plt.figure(figsize=(4, 2)).add_subplot(1, 1, 1)
        ax.plot(x, y)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        return ax.get_figure()

    def _hist_plot(
        self,
        y: np.ndarray = None,
        xlabel: str = "",
        ylabel: str = "% of Samples",
        ylim: ty.Tuple[float, float] = None,
        bins: int = 12,
        ax: plt.Axes = None,
    ) -> Figure:
        if ax is None:
            ax = plt.figure(figsize=(4, 2)).add_subplot(1, 1, 1)
        sns.histplot(y, bins=bins, stat="percent", ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(ylim)
        return ax.get_figure()

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

    def _show_or_save(self, fig: Figure, filename: str = None) -> None:
        if filename is None:
            fig.show()
        else:
            fig.savefig(filename, bbox_inches="tight", dpi=600)

    def plot_cost_timeseries(
        self, report: SolverReport, filename: str = None
    ) -> None:
        cost = self._extract_costs_or_warn(report)
        if cost is None:
            return
        fig = self._line_plot(
            x=list(range(len(cost))), y=cost, xlabel="Timestep", ylabel="Cost"
        )
        self._show_or_save(fig, filename)

    def plot_min_cost_timeseries(
        self, report: SolverReport, filename: str = None
    ) -> None:
        cost = self._extract_costs_or_warn(report)
        if cost is None:
            return
        min_cost = np.minimum.accumulate(cost, axis=0)
        fig = self._line_plot(
            x=list(range(len(min_cost))),
            y=min_cost,
            xlabel="Timestep",
            ylabel="Cumulative Best Cost",
        )
        self._show_or_save(fig, filename)

    def plot_cost_distribution(
        self, report: SolverReport, filename: str = None
    ) -> None:
        cost = self._extract_costs_or_warn(report)
        if cost is None:
            return
        fig = self._hist_plot(y=cost, xlabel="Cost", ylim=(-4, 104))
        self._show_or_save(fig, filename)

    def plot_delta_cost_distribution(
        self, report: SolverReport, filename: str = None
    ) -> None:
        cost = self._extract_costs_or_warn(report)
        if cost is None:
            return
        delta_cost = cost[1:] - cost[:-1]
        fig = self._hist_plot(
            y=delta_cost, xlabel="Energy Transition", ylim=(-4, 104)
        )
        self._show_or_save(fig, filename)

    def plot_unique_state_visits(
        self, report: SolverReport, filename: str = None
    ) -> None:
        states = self._extract_states_or_warn(report)
        if states is None:
            return
        num_unique_states = list(
            map(
                lambda i: np.unique(states[:i], axis=0).shape[0]
                / (2**self.problem.num_variables)
                * 100,
                range(len(states)),
            )
        )
        fig = self._line_plot(
            x=list(range(len(states))),
            y=num_unique_states,
            xlabel="Timestep",
            ylabel="% of State Space Visited",
        )
        self._show_or_save(fig, filename)

    def plot_successive_states_distance(
        self, report: SolverReport, filename: str = None
    ) -> None:
        states = self._extract_states_or_warn(report)
        if states is None:
            return
        distance = (
            np.sum(np.abs(states[1:] - states[:-1]), axis=1)
            / self.problem.num_variables
            * 100
        )
        fig = self._line_plot(
            x=list(range(len(states) - 1)),
            y=distance,
            ylabel="% of Bit Flips",
            xlabel="Timestep",
        )
        self._show_or_save(fig, filename)

