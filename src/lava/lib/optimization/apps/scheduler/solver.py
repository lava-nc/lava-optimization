# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/


import networkx as ntx
import numpy as np

from networkx.algorithms.approximation import maximum_independent_set
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path

from lava.utils import loihi
from lava.lib.optimization.apps.scheduler.problems import \
    (SchedulingProblem, SatelliteScheduleProblem)
from lava.lib.optimization.problems.problems import QUBO
from lava.lib.optimization.solvers.generic.solver import (OptimizationSolver,
                                                          SolverConfig)
from lava.lib.optimization.utils.generators.mis import MISProblem


class Scheduler:

    def __init__(self,
                 sp: SchedulingProblem,
                 qubo_weights: Tuple[int, int] = (1, 8),
                 probe_cost: bool = False):
        """Solver for Scheduling Problems.

        Parameters
        ----------
        sp (SchedulingProblem) : Scheduling problem object as defined in
        lava.lib.optimization.apps.scheduler.problems
        qubo_weights (tuple(int, int)) : The QUBO weight matrix parameters
        for diagonal and off-diagonal weights. Default is (1, 8).
        probe_cost (bool) : Toggle whether to probe cost during the solver
        run. Default is False.
        """
        self._problem = sp
        self._graph = sp.graph
        self._qubo_hyperparams = {
            "temperature": int(8),
            "refract": np.random.randint(64, 127,
                                         self._graph.number_of_nodes()),
            "refract_counter": np.random.randint(0, 64,
                                                 self._graph.number_of_nodes()),
        }
        self._qubo_weights = qubo_weights
        self._probe_cost = probe_cost
        self._netx_solution = None
        self._qubo_problem = None
        self._lava_backend = 'Loihi2' if loihi.host else 'CPU'
        self._lava_solver_report = None
        self._lava_solution = None

        sol_criterion = self._problem.sat_cutoff
        if type(sol_criterion) is float and 0.0 < sol_criterion <= 1.0:
            self._qubo_target_cost = (
                int(sol_criterion * self._problem.num_tasks * qubo_weights[0]))
        elif type(sol_criterion) is int and sol_criterion < 0:
            self._qubo_target_cost = sol_criterion

    @property
    def problem(self):
        return self._problem

    @property
    def graph(self):
        return self._graph

    @property
    def qubo_hyperparams(self):
        return self._qubo_hyperparams

    @qubo_hyperparams.setter
    def qubo_hyperparams(self, hp_update: Tuple[Dict, bool]):
        """
        Set hyperparameters for QUBO solver
        Parameters
        ----------
        hp_update (tuple of dict and bool) :
        The bool part toggles whether to update the existing hyperparameters
            or to set new ones from scratch.

        The dict part of the tuple has the following keys:
            - temperature: annealing parameter. Value of this key should be
            an integer, typically between 1 and 10. Default is 8.
            - refract: refractory period for each neuron for which it stays
            "on" or "off". Value of this key should be a NumPy integer array of
            length equal to the number of nodes in the problem graph. Default
            value is comprised by random integers between 64 and 127.
            - refract_counter: Counter for the *initial* number of time-steps
            for which neurons should stay "off". Value of this key should be
            a NumPy integer array of length equal to the number of nodes in
            the problem graph. Default value is comprised by random integers
            between 0 and 64.
        """
        update = hp_update[1]
        if not update:
            self._qubo_hyperparams = hp_update[0]
        else:
            self._qubo_hyperparams.update(hp_update[0])

    @property
    def qubo_weights(self):
        return self._qubo_weights

    @qubo_weights.setter
    def qubo_weights(self, qw: Tuple[int, int]):
        self._qubo_weights = qw

    @property
    def qubo_target_cost(self):
        return self._qubo_target_cost

    @property
    def probe_cost(self):
        return self._probe_cost

    @probe_cost.setter
    def probe_cost(self, val: bool):
        """Toggle whether to probe cost during the solver run.
        Parameters
        ----------
        val (bool) : Default is False.
        """
        self._probe_cost = val

    @property
    def netx_solution(self):
        return self._netx_solution

    @property
    def qubo_problem(self):
        return self._qubo_problem

    @property
    def lava_backend(self):
        return self._lava_backend

    @lava_backend.setter
    def lava_backend(self, backend: str):
        self._lava_backend = backend

    @property
    def lava_solver_report(self):
        return self._lava_solver_report

    @property
    def lava_solution(self):
        return self._lava_solution

    def solve_with_netx(self):
        """ Find an approximate maximum independent set using networkx. """
        solution = maximum_independent_set(self.graph)
        self._netx_solution = np.expand_dims(np.array([row for row in
                                                       solution]), 1)
        self._netx_solution = self._netx_solution[
            np.argsort(self._netx_solution[:, 0])]

    def solve_with_lava_qubo(self, timeout=1000):
        """ Find a maximum independent set using QUBO in Lava. """
        adj_mat = self.problem.adjacency
        qubo_matrix = MISProblem._get_qubo_cost_from_adjacency(
            adj_mat, self.qubo_weights[0], self.qubo_weights[1])
        self._qubo_problem = QUBO(qubo_matrix)
        solver = OptimizationSolver(self.qubo_problem)
        self._lava_solver_report = solver.solve(
            config=SolverConfig(
                timeout=timeout,
                hyperparameters=self.qubo_hyperparams,
                target_cost=self.qubo_target_cost,
                backend=self.lava_backend,
                probe_cost=self.probe_cost,
                log_level=20
            )
        )
        qubo_state = self.lava_solver_report.best_state
        self._lava_solution = (
            np.array(self.graph.nodes))[np.where(qubo_state)[0]]


class SatelliteScheduler(Scheduler):
    def __init__(self,
                 ssp: SatelliteScheduleProblem,
                 **kwargs):
        qubo_weights = kwargs.get("qubo_weights", (1, 8))
        probe_cost = kwargs.get("probe_cost", False)
        super(SatelliteScheduler, self).__init__(ssp,
                                                 qubo_weights=qubo_weights,
                                                 probe_cost=probe_cost)
        self.num_satellites = ssp.num_satellites
        self.num_requests = ssp.num_requests

    def plot_solutions(self):
        """ Plot the solutions using pyplot. """
        plt.figure(figsize=(12, 4), dpi=120)
        if self.netx_solution is not None:
            plt.subplot(131)
            plt.scatter(self.problem.requests[:, 0],
                        self.problem.requests[:, 1],
                        s=2, c='C1')
            for i in self.problem.satellites:
                sat_plan = self.netx_solution[:, 1] == i
                plt.plot(self.netx_solution[sat_plan, 2],
                         self.netx_solution[sat_plan, 3],
                         'C0o-', markersize=2, lw=0.75)
            plt.title(
                f'NetworkX schedule satisfies '
                f'{self.netx_solution.shape[0]} requests.')
            plt.xticks([])
            plt.yticks([])
            plt.subplot(132)
        else:
            plt.subplot(121)
        plt.scatter(self.problem.requests[:, 0],
                    self.problem.requests[:, 1],
                    s=2, c='C1')
        for i in self.problem.satellites:
            sat_plan = self.lava_solution[:, 1] == i
            plt.plot(self.lava_solution[sat_plan, 2],
                     self.lava_solution[sat_plan, 3],
                     'C0o-', markersize=2, lw=0.75)
        plt.title(
            f'Lava schedule satisfies {self.lava_solution.shape[0]} requests.')
        plt.xticks([])
        plt.yticks([])
        if self.lava_solver_report.cost_timeseries is not None:
            plt.subplot(233)
            plt.plot(self.lava_solver_report.cost_timeseries, lw=0.75)
            plt.title(f'QUBO solution cost is '
                      f'{self.lava_solver_report.best_cost}')
            plt.subplot(236)
        else:
            plt.subplot(133)
        longest_plan = 1
        for i in self.problem.satellites:
            sat_plan = self.lava_solution[:, 1] == i
            longest_plan = max(longest_plan, sat_plan.sum() - 1)
            x = self.lava_solution[sat_plan, 2]
            y = self.lava_solution[sat_plan, 3]
            plt.plot(abs(np.diff(y) / np.diff(x)), lw=0.75)
        plt.plot([0, longest_plan],
                 [self.problem.turn_rate, self.problem.turn_rate],
                 '--', lw=0.75)
        plt.title(f'Satellite turn rates')
        plt.tight_layout()
        plt.show()


class SatelliteScheduleProblemm:
    """
    SatelliteScheduleProblem is a synthetic scheduling problem in which a number of
    vehicles must be assigned to view as many requests in a 2-dimensional plane as
    possible. Each vehicle moves horizontally across the plane, has minimum and
    maximum view angle, and has a maximum rotation rate (i.e. the rate at which the
    vehicle can reorient vertically from one target to the next).

    The problem is represented as an infeasibility graph and can be solved by finding
    the Maximum Independent Set.
    """

    def __init__(
            self,
            num_satellites: int = 6,
            view_height: float = 0.25,
            view_coords: Optional[np.ndarray] = None,
            num_requests: int = 48,
            turn_rate: float = 2,
            qubo_weights: Tuple = (1, 8),
            solution_criteria: float = 0.99,
    ):
        """ Create a SatelliteScheduleProblem.
        Parameters
        ----------
        num_satellites : int, default = 6
            The number of satellites to generate schedules for.
        view_height : float, default = 0.25
            The range from minimum to maximum viewable angle for each satellite.
        view_coords : Optional[np.ndarray], default = None
            The view coordinates (i.e. minimum viewable angle) for each satellite
            in a numpy array. If None, view coordinates will be evenly distributed
            across the viewable range.
        num_requests : int, default = 48
            The number of requests to generate.
        turn_rate : float, default = 2
            How quickly each satellite may reorient its view angle.
        qubo_weights : Tuple, default = (1, 8)
            The QUBO weight matrix parameters for diagonal and off-diagonal weights.
        solution_criteria : float, default = 0.99
            The target for a successful solution. The solver will stop looking for
            a better schedule if the specified fraction of requests are satisfied.
        """
        super().__init__()
        self.num_satellites = num_satellites
        self.view_height = view_height
        if view_coords is None:
            self.view_coords = np.linspace(0 - view_height / 2,
                                           1 - view_height / 2,
                                           num_satellites)
        else:
            self.view_coords = view_coords
        self.num_requests = num_requests
        self.turn_rate = turn_rate
        self.qubo_weights = qubo_weights
        if type(solution_criteria) is float and 0.0 < solution_criteria <= 1.0:
            self.target_cost = int(
                -solution_criteria * num_requests * qubo_weights[0])
        elif type(solution_criteria) is int and solution_criteria < 0:
            self.target_cost = solution_criteria
        self.random_seed = None
        self.graph = None
        self.adjacency = None
        self.satellites = None
        self.requests = None
        self.netx_solution = None
        self.lava_backend = None
        self.probe_cost = None
        self.qubo_problem = None
        self.solver_report = None
        self.lava_solution = None

    def generate(self, seed=None):
        """ Generate a new scheduler problem. """
        if seed:
            self.random_seed = seed
            np.random.seed(seed)
        self.graph = ntx.Graph()
        self.satellites = range(self.num_satellites)
        self.generate_requests()
        self.generate_visible_nodes()
        self.generate_infeasibility_graph()
        self.rescale_adjacency()

    def generate_requests(self):
        """ Generate a random set of requests in the 2D plane. """
        self.requests = np.random.random((self.num_requests, 2))
        order = np.argsort(self.requests[:, 0])
        self.requests = self.requests[order, :]

    def generate_visible_nodes(self):
        """
        Add nodes to the graph for every combination of a vehicle and a request visible
        to that vehicle.
        """
        node_id = 0
        for i in self.satellites:
            for j in range(self.num_requests):
                if is_visible(self.view_coords[i], self.requests[j, 1],
                              self.view_height):
                    self.graph.add_node(
                        (node_id, i, self.requests[j, 0], self.requests[j, 1]))
                    node_id += 1
        self.num_nodes = node_id

    def generate_infeasibility_graph(self):
        """
        Create edges between any two nodes of the graph which cannot be traversed and
        add corresponding weights to the adjacency matrix.
        """
        self.adjacency = np.zeros((self.num_nodes, self.num_nodes), dtype=int)
        for n1 in self.graph.nodes:
            for n2 in self.graph.nodes:
                if not is_same_node(n1, n2) and not is_feasible(n1, n2,
                                                                self.turn_rate):
                    self.adjacency[n1[0], n2[0]] = 1
                    self.graph.add_edge(n1, n2)

    def rescale_adjacency(self):
        """ Scale the adjacency matrix weights for QUBO solver. """
        self.adjacency = np.triu(self.adjacency)
        self.adjacency += self.adjacency.T - 2 * np.diag(
            self.adjacency.diagonal())

    def solve_with_netx(self):
        """ Find an approximate maximum independent set using networkx. """
        solution = maximum_independent_set(self.graph)
        self.netx_solution = np.array([row for row in solution])
        self.netx_solution = self.netx_solution[
            np.argsort(self.netx_solution[:, 0])]
        return self.netx_solution

    def set_qubo_hyperparameters(self, t=8, rmin=64, rmax=127):
        """ Set the hyperparameters to use for the QUBO solver. """
        self.hyperparameters = {
            "temperature": int(t),
            "refract": np.random.randint(rmin, rmax,
                                         self.graph.number_of_nodes()),
            "refract_counter": np.random.randint(0, rmin,
                                                 self.graph.number_of_nodes()),
        }

    def solve_with_lava_qubo(self, timeout=1000, probe_cost=False):
        """ Find a maximum independent set using QUBO in Lava. """
        self.lava_backend = 'Loihi2' if loihi.host else 'CPU'
        self.probe_cost = probe_cost
        qubo_matrix = MISProblem._get_qubo_cost_from_adjacency(
            self.adjacency, self.qubo_weights[0], self.qubo_weights[1])
        self.qubo_problem = QUBO(qubo_matrix)
        solver = OptimizationSolver(self.qubo_problem)
        self.solver_report = solver.solve(
            config=SolverConfig(
                timeout=timeout,
                hyperparameters=self.hyperparameters,
                target_cost=self.target_cost,
                backend=self.lava_backend,
                probe_cost=self.probe_cost,
            )
        )
        qubo_state = self.solver_report.best_state
        self.lava_solution = np.array(self.graph.nodes)[np.where(qubo_state)[0]]
        return self.lava_solution

    def plot_problem(self):
        """ Plot the problem state using pyplot. """
        plt.figure(figsize=(12, 4), dpi=120)
        plt.subplot(131)
        plt.scatter(self.requests[:, 0], self.requests[:, 1], s=2)
        for y in self.view_coords:
            codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
            verts = [[-0.05, y + self.view_height / 2],
                     [0.05, y + self.view_height],
                     [0.05, y + 0.0],
                     [-0.05, y + 0.0]]
            plt.gca().add_patch(
                PathPatch(Path(verts, codes), ec='none', alpha=0.3,
                          fc='lightblue'))
            plt.scatter([-0.05], [y + self.view_height / 2], s=10, marker='s',
                        c='gray')
            plt.plot([0, 1],
                     [y + self.view_height / 2, y + self.view_height / 2],
                     'C1--', lw=0.75)
        plt.xticks([])
        plt.yticks([])
        plt.title(
            f'Schedule {self.num_satellites} satellites to observe {self.num_requests} targets.')
        plt.subplot(132)
        ntx.draw_networkx(self.graph, with_labels=False, node_size=2, width=0.5)
        plt.title(
            f'Infeasibility graph with {self.graph.number_of_nodes()} nodes.')
        plt.subplot(133)
        plt.imshow(self.adjacency, aspect='auto')
        plt.title(
            f'Adjacency matrix has {self.adjacency.mean():.2%} connectivity.')
        plt.yticks([])
        plt.tight_layout()
        plt.show()

    def plot_solutions(self):
        """ Plot the solutions using pyplot. """
        plt.figure(figsize=(12, 4), dpi=120)
        if self.netx_solution is not None:
            plt.subplot(131)
            plt.scatter(self.requests[:, 0], self.requests[:, 1], s=2, c='C1')
            for i in self.satellites:
                sat_plan = self.netx_solution[:, 1] == i
                plt.plot(self.netx_solution[sat_plan, 2],
                         self.netx_solution[sat_plan, 3], 'C0o-', markersize=2,
                         lw=0.75)
            plt.title(
                f'NetworkX schedule satisfies {self.netx_solution.shape[0]} requests.')
            plt.xticks([])
            plt.yticks([])
            plt.subplot(132)
        else:
            plt.subplot(121)
        plt.scatter(self.requests[:, 0], self.requests[:, 1], s=2, c='C1')
        for i in self.satellites:
            sat_plan = self.lava_solution[:, 1] == i
            plt.plot(self.lava_solution[sat_plan, 2],
                     self.lava_solution[sat_plan, 3], 'C0o-', markersize=2,
                     lw=0.75)
        plt.title(
            f'Lava schedule satisfies {self.lava_solution.shape[0]} requests.')
        plt.xticks([])
        plt.yticks([])
        if self.solver_report.cost_timeseries is not None:
            plt.subplot(233)
            plt.plot(self.solver_report.cost_timeseries, lw=0.75)
            plt.title(f'QUBO solution cost is {self.solver_report.best_cost}')
            plt.subplot(236)
        else:
            plt.subplot(133)
        longest_plan = 1
        for i in self.satellites:
            sat_plan = self.lava_solution[:, 1] == i
            longest_plan = max(longest_plan, sat_plan.sum() - 1)
            x = self.lava_solution[sat_plan, 2]
            y = self.lava_solution[sat_plan, 3]
            plt.plot(abs(np.diff(y) / np.diff(x)), lw=0.75)
        plt.plot([0, longest_plan], [self.turn_rate, self.turn_rate], '--',
                 lw=0.75)
        plt.title(f'Satellite turn rates')
        plt.tight_layout()
        plt.show()
