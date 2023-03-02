# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import unittest

import numpy as np

from lava.lib.optimization.problems.problems import QUBO
from lava.lib.optimization.solvers.generic.hierarchical_processes import (
    CostConvergenceChecker,
)
from lava.lib.optimization.solvers.generic.monitoring_processes \
    .solution_readout.process import SolutionReadout
from lava.lib.optimization.solvers.generic.read_gate.process import ReadGate
from lava.lib.optimization.solvers.generic.solver import (
    OptimizationSolver, SolverConfig
)
from lava.lib.optimization.utils.generators.mis import MISProblem


class TestOptimizationSolver(unittest.TestCase):
    def setUp(self) -> None:
        print("SETUP")
        self.problem = QUBO(
            np.array([[-5, 2, 4, 0],
                      [2, -3, 1, 0],
                      [4, 1, -8, 5],
                      [0, 0, 5, -6]]))
        self.solution = np.asarray([1, 0, 0, 1]).astype(int)
        self.solver = OptimizationSolver(problem=self.problem)

    def test_create_obj(self):
        print("test_create_obj")
        self.assertIsInstance(self.solver, OptimizationSolver)

    def test_solution_has_expected_shape(self):
        print("test_solution_has_expected_shape")
        report = self.solver.solve(config=SolverConfig(timeout=3000))
        self.assertEqual(report.best_state.shape, self.solution.shape)

    def test_solve_method_nebm(self):
        print("test_solve_method")
        np.random.seed(2)
        config = SolverConfig(
            timeout=200,
            target_cost=-11,
            backend="CPU",
            hyperparameters={"neuron_model": "nebm"}
        )
        report = self.solver.solve(config=config)
        print(report)
        self.assertTrue((report.best_state == self.solution).all())
        self.assertEqual(report.best_cost, self.problem.evaluate_cost(
            report.best_state))

    def test_solve_method_scif(self):
        print("test_solve_method")
        np.random.seed(2)
        report = self.solver.solve(config=SolverConfig(
            timeout=200,
            target_cost=-11,
            hyperparameters={"neuron_model": "scif",
                             "noise_precision": 5}
        ))
        print(report)
        self.assertTrue((report.best_state == self.solution).all())
        self.assertEqual(report.best_cost, self.problem.evaluate_cost(
            report.best_state))

    def test_solver_creates_optimizationsolver_process(self):
        self.solver._create_solver_process(config=SolverConfig(backend="CPU"))
        class_name = type(self.solver.solver_process).__name__
        self.assertEqual(class_name, "OptimizationSolverProcess")

    def test_solves_creates_macrostate_reader_processes(self):
        self.assertIsNone(self.solver.solver_process)
        self.solver.solve(config=SolverConfig(timeout=1))
        mr = self.solver.solver_process.model_class(
            self.solver.solver_process
        ).macrostate_reader
        self.assertIsInstance(mr.read_gate, ReadGate)
        self.assertIsInstance(mr.solution_readout, SolutionReadout)
        self.assertEqual(
            mr.solution_readout.solution.shape,
            (self.problem.variables.num_variables,),
        )
        self.assertIsInstance(mr.cost_convergence_check, CostConvergenceChecker)

    def test_macrostate_reader_processes_connections(self):
        self.assertIsNone(self.solver.solver_process)
        self.solver.solve(config=SolverConfig(timeout=1))
        mr = self.solver.solver_process.model_class(
            self.solver.solver_process
        ).macrostate_reader
        self.assertIs(
            mr.cost_convergence_check.update_buffer.out_connections[0].process,
            mr.read_gate,
        )
        self.assertIs(
            mr.read_gate.cost_out.out_connections[0].process,
            mr.solution_readout,
        )
        self.assertIs(
            self.solver.solver_process.variable_assignment.aliased_var,
            mr.solution_readout.solution,
        )
        self.assertIs(
            self.solver.solver_process.variable_assignment.aliased_var.process,
            mr.solution_readout,
        )

    def test_cost_checker_is_connected_to_variables_population(self):
        self.assertIsNone(self.solver.solver_process)
        self.solver.solve(config=SolverConfig(timeout=1))
        pm = self.solver.solver_process.model_class(
            self.solver.solver_process
        )
        mr = pm.macrostate_reader
        self.assertIs(
            mr.cost_convergence_check.cost_components.in_connections[0].process,
            pm.variables.discrete,
        )

    def test_qubo_cost_defines_weights(self):
        self.solver.solve(config=SolverConfig(timeout=1))
        pm = self.solver.solver_process.model_class(
            self.solver.solver_process
        )
        q_no_diag = np.copy(self.problem.cost.get_coefficient(2))
        np.fill_diagonal(q_no_diag, 0)
        wgts = pm.cost_minimizer.coefficients_2nd_order.weights
        condition = (wgts.init == q_no_diag).all()
        self.assertTrue(condition)

    def test_qubo_cost_defines_biases(self):
        self.solver.solve(config=SolverConfig(timeout=1))
        pm = self.solver.solver_process.model_class(
            self.solver.solver_process
        )
        condition = (pm.variables.discrete.cost_diagonal == self.problem.cost
                     .get_coefficient(2).diagonal()).all()
        self.assertTrue(condition)

    def test_qubo_cost_defines_num_vars_in_discrete_variables_process(self):
        self.solver.solve(config=SolverConfig(timeout=1))
        pm = self.solver.solver_process.model_class(
            self.solver.solver_process
        )
        self.assertEqual(
            pm.variables.discrete.num_variables,
            self.problem.variables.num_variables,
        )
        self.assertEqual(
            self.solver.solver_process.variable_assignment.size,
            self.problem.variables.num_variables,
        )

    def test_cost_tracking(self):
        print("test_cost_tracking")
        config = SolverConfig(
            timeout=50,
            target_cost=-20,
            backend="CPU",
            probe_cost=True
        )
        report = self.solver.solve(config=config)
        self.assertIsInstance(report.cost_timeseries, np.ndarray)
        self.assertEqual(report.cost_timeseries[0][report.best_timestep],
                         report.best_cost)


def solve_workload(problem, reference_solution, noise_precision=3,
                   noise_amplitude=1, on_tau=-3):
    expected_cost = problem.evaluate_cost(reference_solution)
    np.random.seed(2)
    solver = OptimizationSolver(problem)
    report = solver.solve(config=SolverConfig(
        timeout=20000,
        target_cost=expected_cost,
        hyperparameters={
            'neuron_model': 'nebm',
            'temperature': 1
        }
    ))
    return report, expected_cost


class TestWorkloads(unittest.TestCase):

    def test_solve_polynomial_minimization(self):
        """Polynomial minimization with y=-5x_1 -3x_2 -8x_3 -6x_4 +
        4x_1x_2+8x_1x_3+2x_2x_3+10x_3x_4
        """
        problem = QUBO(q=np.array([[-5, 2, 4, 0],
                                   [2, -3, 1, 0],
                                   [4, 1, -8, 5],
                                   [0, 0, 5, -6]]))
        reference_solution = np.asarray([1, 0, 0, 1]).astype(int)
        report, expected_cost = solve_workload(problem, reference_solution,
                                               noise_precision=5)
        self.assertEqual(problem.evaluate_cost(report.best_state),
                         expected_cost)
        self.assertTrue((report.best_state == reference_solution).all())

    def test_solve_set_packing(self):
        problem = QUBO(q=-np.array([[1, -3, -3, -3],
                                    [-3, 1, 0, 0],
                                    [-3, 0, 1, -3],
                                    [-3, 0, -3, 1]]))

        reference_solution = np.zeros(4)
        np.put(reference_solution, [1, 2], 1)
        report, expected_cost = solve_workload(problem, reference_solution,
                                               noise_precision=5)
        self.assertEqual(problem.evaluate_cost(report.best_state),
                         expected_cost)

    def test_solve_max_cut_problem(self):
        """Max-Cut Problem"""
        problem = QUBO(q=-np.array([[2, -1, -1, 0, 0],
                                    [-1, 2, 0, -1, 0],
                                    [-1, 0, 3, -1, -1],
                                    [0, -1, -1, 3, -1],
                                    [0, 0, -1, -1, 2]]))
        reference_solution = np.zeros(5)
        np.put(reference_solution, [1, 2], 1)
        report, expected_cost = solve_workload(problem, reference_solution,
                                               noise_precision=5)
        self.assertEqual(problem.evaluate_cost(report.best_state),
                         expected_cost)

    def test_solve_set_partitioning(self):
        problem = QUBO(q=np.array([[-17, 10, 10, 10, 0, 20],
                                   [10, -18, 10, 10, 10, 20],
                                   [10, 10, -29, 10, 20, 20],
                                   [10, 10, 10, -19, 10, 10],
                                   [0, 10, 20, 10, -17, 10],
                                   [20, 20, 20, 10, 10, -28]]))
        reference_solution = np.zeros(6)
        np.put(reference_solution, [0, 4], 1)
        report, expected_cost = solve_workload(problem, reference_solution,
                                               noise_precision=6,
                                               noise_amplitude=1)
        self.assertEqual(problem.evaluate_cost(report.best_state),
                         expected_cost)

    def test_solve_map_coloring(self):
        p = QUBO(q=np.array([[-4, 4, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
                             [4, -4, 4, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
                             [4, 4, -4, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2],
                             [2, 0, 0, -4, 4, 4, 2, 0, 0, 2, 0, 0, 2, 0, 0],
                             [0, 2, 0, 4, -4, 4, 0, 2, 0, 0, 2, 0, 0, 2, 0],
                             [0, 0, 2, 4, 4, -4, 0, 0, 2, 0, 0, 2, 0, 0, 2],
                             [0, 0, 0, 2, 0, 0, -4, 4, 4, 2, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 2, 0, 4, -4, 4, 0, 2, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 2, 4, 4, -4, 0, 0, 2, 0, 0, 0],
                             [0, 0, 0, 2, 0, 0, 2, 0, 0, -4, 4, 4, 2, 0, 0],
                             [0, 0, 0, 0, 2, 0, 0, 2, 0, 4, -4, 4, 0, 2, 0],
                             [0, 0, 0, 0, 0, 2, 0, 0, 2, 4, 4, -4, 0, 0, 2],
                             [2, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, -4, 4, 4],
                             [0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 4, -4, 4],
                             [0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 2, 4, 4, -4]]))
        reference_solution = np.zeros(15)
        np.put(reference_solution, [1, 3, 8, 10, 14], 1)
        report, expected_cost = solve_workload(p, reference_solution,
                                               noise_precision=5,
                                               noise_amplitude=1,
                                               on_tau=-1)
        self.assertEqual(p.evaluate_cost(report.best_state), expected_cost)

    def test_solve_mis(self):
        mis = MISProblem(
            num_vertices=15,
            connection_prob=0.9,
            seed=0
        )
        problem = mis.get_as_qubo(1, 8)
        reference_solution = mis.find_maximum_independent_set()
        report, expected_cost = solve_workload(problem, reference_solution,
                                               noise_precision=5,
                                               noise_amplitude=1,
                                               on_tau=-1)
        self.assertEqual(problem.evaluate_cost(report.best_state),
                         expected_cost)


if __name__ == "__main__":
    unittest.main()
