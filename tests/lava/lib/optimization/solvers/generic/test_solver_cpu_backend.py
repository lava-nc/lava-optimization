# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import os
import unittest

import numpy as np
from lava.lib.optimization.problems.problems import QP, QUBO
from lava.lib.optimization.solvers.generic.solution_finder.process import (
    SolutionFinder,
)
from lava.lib.optimization.solvers.generic.solution_reader.process import (
    SolutionReader,
)
from lava.lib.optimization.solvers.generic.solver import (
    OptimizationSolver,
    SolverConfig,
)
from lava.lib.optimization.utils.generators.mis import MISProblem
from lava.lib.optimization.solvers.generic.solution_finder.models\
    import SolutionFinderModel


@unittest.skip("CPU backend of QUBO solver temporarily disabled.")
class TestOptimizationSolverQUBOSetup(unittest.TestCase):
    def setUp(self) -> None:
        self.problem = QUBO(
            np.array(
                [[-5, 2, 4, 0],
                 [2, -3, 1, 0],
                 [4, 1, -8, 5],
                 [0, 0, 5, -6]]
            )
        )
        self.solution = np.asarray([1, 0, 0, 1]).astype(int)
        self.solver = OptimizationSolver(problem=self.problem)

    def test_create_obj(self):
        self.assertIsInstance(self.solver, OptimizationSolver)

    def test_solution_has_expected_shape(self):
        report = self.solver.solve(config=SolverConfig(timeout=3000))
        self.assertEqual(report.best_state.shape, self.solution.shape)

    def test_solve_method_nebm(self):
        np.random.seed(2)
        config = SolverConfig(
            timeout=2000,
            target_cost=-11,
            backend="CPU",
            hyperparameters={"neuron_model": "nebm"},
        )
        report = self.solver.solve(config=config)
        self.assertTrue((report.best_state == self.solution).all())
        self.assertEqual(
            report.best_cost, self.problem.evaluate_cost(report.best_state)
        )

    def test_solve_method_scif(self):
        np.random.seed(2)
        report = self.solver.solve(
            config=SolverConfig(
                timeout=200,
                target_cost=-11,
                hyperparameters={"neuron_model": "scif", "noise_precision": 5},
            )
        )
        self.assertTrue((report.best_state == self.solution).all())
        self.assertEqual(
            report.best_cost, self.problem.evaluate_cost(report.best_state)
        )

    def test_solver_creates_optimizationsolver_process(self):
        self.solver._create_solver_process(config=SolverConfig(backend="CPU"))
        class_name = type(self.solver.solver_process).__name__
        self.assertEqual(class_name, "OptimizationSolverProcess")

    def test_solves_creates_subprocesses(self):
        self.assertIsNone(self.solver.solver_process)
        self.solver.solve(config=SolverConfig(timeout=1))
        pm = self.solver.solver_process.model_class(self.solver.solver_process)
        self.assertIsInstance(pm.finder_0, SolutionFinder)
        self.assertIsInstance(pm.solution_reader, SolutionReader)

    def test_subprocesses_connections(self):
        # TODO split into a test for SolutionReader and one for SolutionFinder
        self.assertIsNone(self.solver.solver_process)
        self.solver.solve(config=SolverConfig(timeout=1))
        pm = self.solver.solver_process.model_class(self.solver.solver_process)
        solution_finder = pm.finder_0
        solution_reader = pm.solution_reader
        best_assignment = self.solver.solver_process.best_variable_assignment
        self.assertIs(
            solution_finder.cost_out_last_bytes.out_connections[0].process,
            solution_reader,
        )
        self.assertIs(best_assignment.aliased_var, solution_reader.solution)
        self.assertIs(
            best_assignment.aliased_var.process,
            solution_reader,
        )

    def test_qubo_cost_defines_weights(self):
        self.solver.solve(config=SolverConfig(timeout=1))
        pm = self.solver.solver_process.model_class(self.solver.solver_process)
        q_no_diag = np.copy(self.problem.cost.get_coefficient(2))
        np.fill_diagonal(q_no_diag, 0)
        sfpm = SolutionFinderModel(pm.finder_0)
        wgts = sfpm.cost_minimizer.coefficients_2nd_order.weights.init
        condition = (wgts == q_no_diag).all()
        self.assertTrue(condition)

    def test_cost_tracking(self):
        np.random.seed(77)
        config = SolverConfig(
            timeout=50, target_cost=-20, backend="CPU", probe_cost=True
        )
        report = self.solver.solve(config=config)
        self.assertIsInstance(report.cost_timeseries, np.ndarray)
        self.assertEqual(
            report.best_cost, report.cost_timeseries.T[0][report.best_timestep]
        )

    def test_state_tracking(self):
        np.random.seed(77)
        config = SolverConfig(
            timeout=50, target_cost=-20, backend="CPU", probe_state=True
        )
        report = self.solver.solve(config=config)
        states = report.state_timeseries
        self.assertIsInstance(states, np.ndarray)
        self.assertTrue(
            np.all(report.best_state == states[report.best_timestep])
        )


class TestOptimizationSolverQP(unittest.TestCase):
    def setUp(self) -> None:
        root = os.path.dirname(os.path.abspath(__file__))
        qp_data = np.load(root + "/data/qp/ex_qp_small.npz")
        Q, self.A, p, k = [qp_data[i] for i in qp_data]
        p, self.k = np.squeeze(p), np.squeeze(k)
        qp_workloads = {
            "anymal": QP(
                hessian=Q,
                linear_offset=p,
                equality_constraints_weights=self.A,
                equality_constraints_biases=self.k,
            )
        }
        self.problem = qp_workloads["anymal"]
        self.problem.precondition_problem(iterations=5, type="ruiz")
        self.solver = OptimizationSolver(problem=self.problem)
        self.solution_shape = (self.problem.hessian.shape[0],)

    def test_create_obj(self):
        self.assertIsInstance(self.solver, OptimizationSolver)

    def test_solution_has_expected_shape(self):
        report = self.solver.solve(config=SolverConfig(timeout=3000))
        self.assertEqual(report.best_state.shape, self.solution_shape)

    def test_constraint_satisfaction_flt_pt_cpu(self):
        np.random.seed(2)
        mu, sigma, lamda = 0.11, 8.14, 1.6
        alpha_init = 2 / (mu + 2 * lamda)
        beta_init = mu / (2 * sigma)
        config = SolverConfig(
            timeout=1000 * 2,
            backend="CPU",
            hyperparameters={
                "neuron_model": "qp-lp_pipg",
                "alpha": alpha_init,
                "beta": beta_init,
                "lr_change_type": "indices",
                "alpha_decay_indices": (
                    np.array(
                        [50, 100, 200, 350, 550, 800, 1100, 1450, 1850, 2300]
                    )
                    * 2
                ),
                "beta_growth_indices": (
                    np.array(
                        [
                            1,
                            3,
                            7,
                            15,
                            31,
                            63,
                            127,
                            255,
                            511,
                            1023,
                            2047,
                            4095,
                        ]
                    )
                    * 2
                    + 1
                ),
                "decay_schedule_parameters": (100, 100, 0),
                "growth_schedule_parameters": (3, 2),
            },
        )
        report = self.solver.solve(config=config)
        solution = self.problem.postconditioner @ report.best_state
        self.assertLess(
            np.linalg.norm(self.A @ solution - self.k),
            10,
            "Solver didn't converge",
        )

    def test_solver_creates_optimizationsolver_process(self):
        self.solver._create_solver_process(config=SolverConfig(backend="CPU"))
        class_name = type(self.solver.solver_process).__name__
        self.assertEqual(class_name, "OptimizationSolverProcess")

    def test_solves_creates_subprocesses(self):
        self.assertIsNone(self.solver.solver_process)
        self.solver.solve(config=SolverConfig(timeout=1))
        pm = self.solver.solver_process.model_class(self.solver.solver_process)
        self.assertIsInstance(pm.finder_0, SolutionFinder)


def solve_workload(problem, reference_solution):
    expected_cost = problem.evaluate_cost(reference_solution)
    np.random.seed(2)
    solver = OptimizationSolver(problem)
    report = solver.solve(
        config=SolverConfig(
            timeout=1000,
            target_cost=expected_cost,
            probe_cost=True,
            hyperparameters={
                "neuron_model": "nebm",
                "temperature": 2,
                "refract_counter": np.random.randint(0, 10, problem.q.shape[0]),
                "refract": np.random.randint(2, 8, problem.q.shape[0])
            },
        )
    )
    return report, expected_cost


@unittest.skip("CPU backend of QUBO solver temporarily disabled.")
class TestOptimizationSolverQUBO(unittest.TestCase):
    def test_solve_polynomial_minimization(self):
        """Polynomial minimization with y=-5x_1 -3x_2 -8x_3 -6x_4 +
        4x_1x_2+8x_1x_3+2x_2x_3+10x_3x_4
        """
        problem = QUBO(
            q=np.array(
                [[-5, 2, 4, 0],
                 [2, -3, 1, 0],
                 [4, 1, -8, 5],
                 [0, 0, 5, -6]]
            )
        )
        reference_solution = np.asarray([1, 0, 0, 1]).astype(int)
        report, expected_cost = solve_workload(problem, reference_solution)
        self.assertEqual(problem.evaluate_cost(report.best_state),
                         expected_cost)
        self.assertTrue((report.best_state == reference_solution).all())

    def test_solve_set_packing(self):
        problem = QUBO(
            q=-np.array(
                [
                    [1, -3, -3, -3],
                    [-3, 1, 0, 0],
                    [-3, 0, 1, -3],
                    [-3, 0, -3, 1],
                ]
            )
        )

        reference_solution = np.zeros(4)
        np.put(reference_solution, [1, 2], 1)
        report, expected_cost = solve_workload(problem, reference_solution)
        self.assertEqual(problem.evaluate_cost(report.best_state),
                         expected_cost)

    def test_solve_max_cut_problem(self):
        """Max-Cut Problem"""
        problem = QUBO(
            q=-np.array(
                [
                    [2, -1, -1, 0, 0],
                    [-1, 2, 0, -1, 0],
                    [-1, 0, 3, -1, -1],
                    [0, -1, -1, 3, -1],
                    [0, 0, -1, -1, 2],
                ]
            )
        )
        reference_solution = np.zeros(5)
        np.put(reference_solution, [1, 2], 1)
        report, expected_cost = solve_workload(problem, reference_solution)
        self.assertEqual(problem.evaluate_cost(report.best_state),
                         expected_cost)

    def test_solve_set_partitioning(self):
        problem = QUBO(
            q=np.array(
                [
                    [-17, 10, 10, 10, 0, 20],
                    [10, -18, 10, 10, 10, 20],
                    [10, 10, -29, 10, 20, 20],
                    [10, 10, 10, -19, 10, 10],
                    [0, 10, 20, 10, -17, 10],
                    [20, 20, 20, 10, 10, -28],
                ]
            )
        )
        reference_solution = np.array([1, 0, 0, 0, 1, 0])
        report, expected_cost = solve_workload(problem, reference_solution)
        self.assertEqual(problem.evaluate_cost(report.best_state),
                         expected_cost)

    def test_solve_map_coloring(self):
        p = QUBO(
            q=np.array(
                [
                    [-4, 4, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
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
                    [0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 2, 4, 4, -4],
                ]
            )
        )
        reference_solution = np.zeros(15)
        np.put(reference_solution, [1, 3, 8, 10, 14], 1)
        report, expected_cost = solve_workload(p, reference_solution)
        self.assertEqual(p.evaluate_cost(report.best_state), expected_cost)

    def test_solve_mis(self):
        mis = MISProblem(num_vertices=15, connection_prob=0.9, seed=0)
        problem = mis.get_as_qubo(1, 8)
        reference_solution = mis.find_maximum_independent_set()
        report, expected_cost = solve_workload(problem, reference_solution)
        self.assertEqual(problem.evaluate_cost(report.best_state),
                         expected_cost)


if __name__ == "__main__":
    unittest.main()
