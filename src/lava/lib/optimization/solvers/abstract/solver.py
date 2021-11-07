# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from src.lava.lib.optimization.solvers.abstract.processes import Readout, \
    HostMonitor


class OptimizationSolver:
    """Abstract optimization solver from which to derive actual solvers.

    Kwargs:
        problem (OptimizationProblem): specification of the problem to be
        solved. Defaults to None.
    """

    def __init__(self, problem=None, **kwargs):
        self._target_cost = 0
        self._problem = problem
        self.collected_solutions = []

    @property
    def problem(self):
        """Optimization problem to be solved by this solver.
        :return:
        """
        return self._problem

    @problem.setter
    def problem(self, value):
        self._problem = value

    @property
    def target_cost(self):
        """Target value for problem cost function."""
        return self._target_cost

    @target_cost.setter
    def target_cost(self, value):
        self._target_cost = value

    def _build(self, snn, integrator):
        """Create  process network that actually solves the problem.

        :param snn: neural network that solves the problem through its dynamics.
        :param integrator: integrator that notifies solution via a spike.
        """
        # Create postprocessor processes
        self.read_out = Readout(population_size=snn.readout_size,
                                target_cost=self.target_cost)
        self.host_monitor = HostMonitor(population_size=snn.readout_size)

        # Connect processes.
        snn.out_ports.to_integrator.connect(integrator.in_ports.from_snn)
        snn.ref_vars.snn_state(self.read_out.ref_ports.snn_state)
        integrator.out_ports.cost.connect(self.read_out.in_ports.cost)
        self.read_out.out_ports.solving_state.connect(
            self.host_monitor.in_ports.solving_state)
        self.read_out.out_ports.solving_time.connect(
            self.host_monitor.in_ports.solving_time)

    def _run(self, timeout, backend=None):
        pass

    def solve(self, problem=None, problems: list = None, timeout=None,
              target_cost=None, backend=None, seed=1):
        """Tries to solve a given optimization problem.

        :param problem: the optimization problem to be solved
        :param problems list tuple: set of problems to be solved
        :param seed: seed for python's RNG.
        :param timeout: maximum number of timesteps to search for a solution.
        :param target_cost: if an optimal solution is not needed or possible, target_costs set's the number of
        satisfying variables that cause the reporting neuron to spike.
        :return: Solution to the problem if one is found.
        """
        if target_cost is not None:
            self.target_cost = target_cost
        solution = self._run(timeout)
        # Get network state at solving time from readout sequential process.
        self.solution = self.host_monitor.vars.solving_state.get()
        self.solving_time = self.host_monitor.vars.solving_time.get()
        if solution is not None and self._check_solution(solution):
            self.collected_solutions.append(solution)
            return solution
        else:
            return None

    def _check_solution(self, solution):
        """Verify that a solution is consistent, complete and optimal."""
        pass
