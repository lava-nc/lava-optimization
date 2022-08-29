# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

import numpy as np

from lava.lib.optimization.problems.problems import QUBO
from lava.lib.optimization.solvers.generic.processes \
    import ReadGate, SolutionReadout, CostConvergenceChecker
from lava.lib.optimization.solvers.generic.solver import \
    OptimizationSolver


class TestOptimizationSolver(unittest.TestCase):
    def setUp(self) -> None:
        self.solver = OptimizationSolver()
        self.problem = QUBO(np.asarray([[-5, 2, 4, 0],
                                        [2, -3, 1, 0],
                                        [4, 1, -8, 5],
                                        [0, 0, 5, -6]]))
        self.solution = np.asarray([1, 0, 0, 1]).astype(int)

    def test_create_obj(self):
        self.assertIsInstance(self.solver, OptimizationSolver)

    def test_solver_creates_optimization_solver_process(self):
        solver_process = self.solver._create_solver_process(self.problem)
        class_name = type(solver_process).__name__
        self.assertIs(solver_process, self.solver._solver_process)
        self.assertEqual(class_name, 'OptimizationSolverProcess')

    def test_solves_creates_macrostate_reader_processes(self):
        self.assertIsNone(self.solver._solver_process)
        self.solver.solve(self.problem, timeout=1)
        mr = self.solver._solver_process.model_class(
            self.solver._solver_process).macrostate_reader
        self.assertIsInstance(mr.read_gate, ReadGate)
        self.assertIsInstance(mr.solution_readout, SolutionReadout)
        self.assertEqual(mr.solution_readout.solution.shape,
                         (self.problem.variables.num_variables,))
        self.assertIsInstance(mr.cost_convergence_check, CostConvergenceChecker)

    def test_macrostate_reader_processes_connections(self):
        self.assertIsNone(self.solver._solver_process)
        self.solver.solve(self.problem, timeout=1)
        mr = self.solver._solver_process.model_class(
            self.solver._solver_process).macrostate_reader
        self.assertIs(
            mr.cost_convergence_check.s_out.out_connections[0].process,
            mr.read_gate)
        self.assertIs(
            mr.read_gate.out_port.out_connections[0].process,
            mr.solution_readout)
        self.assertIs(
            self.solver._solver_process.variable_assignment.aliased_var,
            mr.solution_readout.solution)
        self.assertIs(
            self.solver._solver_process.variable_assignment.aliased_var.process,
            mr.solution_readout)

    def test_cost_checker_is_connected_to_variables_population(self):
        self.assertIsNone(self.solver._solver_process)
        self.solver.solve(self.problem, timeout=1)
        pm = self.solver._solver_process.model_class(
            self.solver._solver_process)
        mr = pm.macrostate_reader
        self.assertIs(
            mr.cost_convergence_check.s_in.in_connections[0].process,
            pm.variables.discrete)

    def test_qubo_cost_defines_weights(self):
        self.solver.solve(self.problem, timeout=1)
        pm = self.solver._solver_process.model_class(
            self.solver._solver_process)
        q_no_diag = np.copy(self.problem.cost.get_coefficient(2))
        np.fill_diagonal(q_no_diag, 0)
        condition = (pm.cost_minimizer.coefficients_2nd_order.weights.init ==
                     -q_no_diag).all()
        self.assertTrue(condition)


if __name__ == '__main__':
    unittest.main()
