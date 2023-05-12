# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/


import typing as ty
import warnings

import numpy as np
import seaborn as sns
from lava.lib.optimization.problems.problems import QUBO
from lava.lib.optimization.solvers.generic.solver import SolverReport
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

sns.set_style("whitegrid")


class StateAnalyzer:
    """
    Utility class to analyze and plots useful metrics on the execution of
    OptimizationSolver.
    """

    def __init__(self, problem: QUBO) -> None:
        """
        Constructor method for StateAnalyzer class.

        Parameters
        ----------
        problem: QUBO
            QUBO problem to be analyzed.
        """
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
    ) -> plt.Axes:
        if ax is None:
            ax = plt.figure(figsize=(4, 2)).add_subplot()
        ax.plot(x, y)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        plt.tight_layout()
        return ax

    def _hist_plot(
        self,
        y: np.ndarray = None,
        xlabel: str = "",
        ylabel: str = "% of Samples",
        ylim: ty.Tuple[float, float] = None,
        bins: int = 12,
        ax: plt.Axes = None,
    ) -> plt.Axes:
        if ax is None:
            ax = plt.figure(figsize=(4, 2)).add_subplot()
        sns.histplot(y, bins=bins, stat="percent", ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(ylim)
        plt.tight_layout()
        return ax

    def _extract_costs_or_warn(self, report: SolverReport) -> np.ndarray:
        if report.cost_timeseries is not None:
            return np.array(report.cost_timeseries)
        else:
            warnings.warn("Cost timeseries is not available.")

    def _extract_states_or_warn(self, report: SolverReport) -> np.ndarray:
        if report.state_timeseries is not None:
            return (
                np.array(report.state_timeseries)
                .reshape((self.problem.num_variables, -1))
                .T
            )
        else:
            warnings.warn("State timeseries is not available.")

    def _show_or_save(
        self, ax: plt.Axes, filename: str = None, dpi: int = 600
    ) -> None:
        if filename is None:
            ax.get_figure().show()
        else:
            ax.get_figure().savefig(filename, bbox_inches="tight", dpi=dpi)

    def plot_cost_timeseries(
        self, report: SolverReport, filename: str = None, ax: plt.Axes = None
    ) -> None:
        """Plot cost through time steps."""
        cost = self._extract_costs_or_warn(report)
        if cost is None:
            return
        ax = self._line_plot(
            x=list(range(len(cost))),
            y=cost,
            xlabel="Time Step",
            ylabel="Cost",
            ax=ax,
        )
        self._show_or_save(ax=ax, filename=filename)

    def plot_min_cost_timeseries(
        self, report: SolverReport, filename: str = None, ax: plt.Axes = None
    ) -> None:
        """Plot cumulative min of cost through time steps."""
        cost = self._extract_costs_or_warn(report)
        if cost is None:
            return
        min_cost = np.minimum.accumulate(cost, axis=0)
        ax = self._line_plot(
            x=list(range(len(min_cost))),
            y=min_cost,
            xlabel="Time Step",
            ylabel="Best Cost",
            ax=ax,
        )
        self._show_or_save(ax=ax, filename=filename)

    def plot_cost_distribution(
        self, report: SolverReport, filename: str = None, ax: plt.Axes = None
    ) -> None:
        """Plot distribution of cost."""
        cost = self._extract_costs_or_warn(report)
        if cost is None:
            return
        ax = self._hist_plot(y=cost, xlabel="Cost", ylim=(-4, 104), ax=ax)
        self._show_or_save(ax=ax, filename=filename)

    def plot_delta_cost_distribution(
        self, report: SolverReport, filename: str = None, ax: plt.Axes = None
    ) -> None:
        """Plot disribution of cost transitions."""
        cost = self._extract_costs_or_warn(report)
        if cost is None:
            return
        delta_cost = cost[1:] - cost[:-1]
        ax = self._hist_plot(
            y=delta_cost, xlabel="Energy Transition", ylim=(-4, 104), ax=ax
        )
        self._show_or_save(ax=ax, filename=filename)

    def plot_num_visited_states(
        self, report: SolverReport, filename: str = None, ax: plt.Axes = None
    ) -> None:
        """Plot the total number of unique states visited through time steps"""
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
            xlabel="Time Step",
            ylabel="# of Visited States",
            ax=ax,
        )
        self._show_or_save(ax=ax, filename=filename)

    def plot_successive_states_distance(
        self, report: SolverReport, filename: str = None, ax: plt.Axes = None
    ) -> None:
        states = self._extract_states_or_warn(report)
        if states is None:
            return
        distance = (
            np.sum(np.abs(states[1:] - states[:-1]), axis=1)
            / self.problem.num_variables
            * 100
        )
        ax = self._line_plot(
            x=list(range(len(states) - 1)),
            y=distance,
            ylabel="% of Bit Flips",
            xlabel="Time Step",
            ax=ax,
        )
        self._show_or_save(ax=ax, filename=filename)

    def plot_state_timeseries(
        self, report: SolverReport, filename: str = None, ax: plt.Axes = None
    ) -> None:
        states = self._extract_states_or_warn(report)
        if states is None:
            return
        if ax is None:
            ax = plt.figure(figsize=(10, 3)).add_subplot(1, 1, 1)
        palette = ["black", "white"]
        sns.heatmap(states.T, vmin=0, vmax=1, ax=ax, cmap=palette, cbar=False)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Neuron Component")
        plt.tight_layout()
        self._show_or_save(ax=ax, filename=filename)

    def plot_state_analysis_summary(
        self, report: SolverReport, filename: str = None
    ) -> None:
        axs = plt.figure(figsize=(10, 8)).subplots(
            4, 1, sharex=True, gridspec_kw={"height_ratios": [1, 1, 2, 1]}
        )
        self.plot_min_cost_timeseries(report=report, ax=axs[0])
        axs[0].set_xlabel("")
        self.plot_cost_timeseries(report=report, ax=axs[1])
        axs[1].set_xlabel("")
        self.plot_state_timeseries(report=report, ax=axs[2])
        axs[2].set_xlabel("")
        self.plot_num_visited_states(report=report, ax=axs[3])
        axs[3].set_xticklabels(axs[3].get_xticklabels(), rotation=90)
        self._show_or_save(ax=axs[0], filename=filename, dpi=800)
