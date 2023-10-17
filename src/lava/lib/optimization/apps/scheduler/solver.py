# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/


import numpy as np
import time

from networkx.algorithms.approximation import maximum_independent_set
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt

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
                 probe_cost: bool = False,
                 probe_loihi_exec_time=False,
                 probe_loihi_energy=False):
        """Solver for Scheduling Problems.

        Parameters
        ----------
        sp : SchedulingProblem
            Scheduling problem object as defined in
            lava.lib.optimization.apps.scheduler.problems
        qubo_weights : tuple(int, int)
            The QUBO weight matrix parameters for diagonal and off-diagonal
            weights. Default is (1, 8).
        probe_cost : bool
            Toggle whether to probe cost during the solver run. Default is
            False.
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
        self._probe_loihi_exec_time = probe_loihi_exec_time
        self._probe_loihi_energy = probe_loihi_energy
        self._netx_solution = None
        self._qubo_problem = None
        self._qubo_matrix = None
        self._lava_backend = 'Loihi2' if loihi.host else 'CPU'
        self._lava_solver_report = None
        self._lava_solution = None

        sol_criterion = self._problem.sat_cutoff
        if type(sol_criterion) is float and 0.0 < sol_criterion <= 1.0:
            self._qubo_target_cost = int(
                -sol_criterion * self._problem.num_tasks * qubo_weights[0])
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
        hp_update : tuple(dict, bool)
            The bool part toggles whether to update the existing
            hyperparameters or to set new ones from scratch.

        Notes
        -----
        Refer to the QUBO Solver documentation for the hyperparameters.
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
        val : bool
            Default is False.
        """
        self._probe_cost = val

    @property
    def probe_loihi_exec_time(self):
        return self._probe_loihi_exec_time

    @property
    def probe_loihi_energy(self):
        return self._probe_loihi_energy

    @property
    def netx_solution(self):
        return self._netx_solution

    @property
    def qubo_problem(self):
        return self._qubo_problem

    @property
    def qubo_matrix(self):
        return self._qubo_matrix

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

    def gen_qubo_mat(self):
        adj_mat = self.problem.adjacency
        self._qubo_matrix = MISProblem._get_qubo_cost_from_adjacency(
            adj_mat, self.qubo_weights[0], self.qubo_weights[1])

    def gen_qubo_problem(self):
        self.gen_qubo_mat()
        self._qubo_problem = QUBO(self.qubo_matrix)

    def solve_with_netx(self):
        """ Find an approximate maximum independent set using networkx. """
        start_time = time.time()
        solution = maximum_independent_set(self.graph)
        self.netx_time = time.time() - start_time
        solution = np.array(list(solution))
        self._netx_solution = np.zeros((solution.size, 4))
        nds = self.graph.nodes
        for j, sol_node in enumerate(solution):
            satellite_id = nds[sol_node]["agent_id"]
            request_coords = nds[sol_node]["task_attr"]
            self._netx_solution[j, :] = (
                np.hstack((sol_node, satellite_id, request_coords)))

    def solve_with_lava_qubo(self, timeout=1000):
        """ Find a maximum independent set using QUBO in Lava. """
        self.gen_qubo_problem()
        solver = OptimizationSolver(self.qubo_problem)
        self._lava_solver_report = solver.solve(
            config=SolverConfig(
                timeout=timeout,
                hyperparameters=self.qubo_hyperparams,
                target_cost=self.qubo_target_cost,
                backend=self.lava_backend,
                probe_cost=self.probe_cost,
                probe_time=self.probe_loihi_exec_time,
                probe_energy=self.probe_loihi_energy,
                log_level=40
            )
        )
        qubo_state = self.lava_solver_report.best_state
        solution = (
            np.array(self.graph.nodes))[np.where(qubo_state)[0]]
        self._lava_solution = np.zeros((solution.size, 4))
        nds = self.graph.nodes
        for j, sol_node in enumerate(solution):
            satellite_id = nds[sol_node]["agent_id"]
            request_coords = nds[sol_node]["task_attr"]
            self._lava_solution[j, :] = (
                np.hstack((sol_node, satellite_id, request_coords)))


class SatelliteScheduler(Scheduler):
    def __init__(self,
                 ssp: SatelliteScheduleProblem,
                 **kwargs):
        qubo_weights = kwargs.pop("qubo_weights", (1, 8))
        probe_cost = kwargs.pop("probe_cost", False)
        super(SatelliteScheduler, self).__init__(ssp,
                                                 qubo_weights=qubo_weights,
                                                 probe_cost=probe_cost,
                                                 **kwargs)
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
