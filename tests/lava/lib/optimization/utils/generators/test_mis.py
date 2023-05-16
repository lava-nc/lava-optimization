# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np

from lava.lib.optimization.solvers.generic.solver import (
    OptimizationSolver, SolverConfig
)
from lava.lib.optimization.utils.generators.mis import MISProblem


class TestMISProblem(unittest.TestCase):
    """Unit tests for MISProblem class."""

    def setUp(self):
        self.adjacency = np.array([
            [0, 1, 1, 0], [1, 0, 0, 0], [1, 0, 0, 1], [0, 0, 1, 0]
        ])
        self.num_vertices = self.adjacency.shape[0]
        self.num_edges = np.count_nonzero(self.adjacency)
        self.problem = MISProblem(adjacency_matrix=self.adjacency)

    def test_create_obj(self):
        """Tests correct instantiation of MISProblem object."""
        self.assertIsInstance(self.problem, MISProblem)

    def test_num_vertices_prop(self):
        """Tests correct value of num_vertices property."""
        self.assertEqual(self.problem.num_vertices, self.num_vertices)

    def test_num_edges_prop(self):
        """Tests correct value of num_edges property."""
        self.assertEqual(self.problem.num_edges, self.num_edges)

    def test_get_graph(self):
        """Tests that the graph contains the correct number of nodes and
        edges."""
        graph = self.problem.get_graph()
        self.assertEqual(graph.number_of_nodes(), self.num_vertices)
        self.assertEqual(2 * graph.number_of_edges(), self.num_edges)

    def test_get_complement_graph(self):
        """Tests that the complement graph contains the correct number of
        nodes and edges."""
        graph = self.problem.get_complement_graph()
        number_of_nodes = 10
        number_of_edges = 8
        self.assertEqual(graph.number_of_nodes(), self.num_vertices)
        self.assertEqual(2 * graph.number_of_edges(), self.num_edges)

    def test_get_complement_graph_matrix(self):
        """Tests the correct complement graph adjacency matrix is returned."""
        matrix = self.problem.get_complement_graph_matrix()
        correct_matrix = np.array([[0, 0, 0, 1],
                                   [0, 0, 1, 1],
                                   [0, 1, 0, 0],
                                   [1, 1, 0, 0]])
        self.assertTrue((matrix == correct_matrix).all())

    def test_get_as_qubo(self):
        """Tests the conversion to QUBO returns the correct cost matrix."""
        w_diag = 1
        w_off = 4
        qubo = self.problem.get_as_qubo(w_diag, w_off)
        correct_matrix = np.array([[-1,  2,  2,  0],
                                   [2, -1,  0,  0],
                                   [2,  0, -1,  2],
                                   [0,  0,  2, -1]])
        self.assertTrue((correct_matrix == qubo.q).all())

    def test_find_maximum_independent_set(self):
        """Tests the correct maximum independent set is returned."""
        mis = self.problem.find_maximum_independent_set()
        correct_mis = np.array([0, 1, 1, 0])
        self.assertTrue((correct_mis == mis).all())

    def test_qubo_solution(self):
        """Tests that a solution with the optimal cost is found by
        OptimizationSolver with the QUBO formulation."""
        optimal_cost = -2
        qubo = self.problem.get_as_qubo(w_diag=1, w_off=4)

        config = SolverConfig(
            timeout=100,
            target_cost=optimal_cost,
            backend="CPU",
            hyperparameters={
                "neuron_model": "nebm",
                "temperature": int(1)
            }
        )

        solver = OptimizationSolver(qubo)
        report = solver.solve(config=config)
        self.assertEqual(qubo.evaluate_cost(report.best_state), optimal_cost)

    def test_qubo_solution_erdos_renyi(self):
        """Tests that a solution with the optimal cost is found by
        OptimizationSolver with the QUBO formulation."""
        problem = MISProblem.from_erdos_renyi(n=25, p=0.95, seed=4000)
        qubo = problem.get_as_qubo(w_diag=1, w_off=4)
        optimal_cost = - \
            np.count_nonzero(problem.find_maximum_independent_set())

        config = SolverConfig(
            timeout=5000,
            target_cost=optimal_cost,
            backend="CPU",
            hyperparameters={
                "neuron_model": "nebm",
                "temperature": int(1),
                "refract": np.random.randint(1, 7, size=(25, ))
            }
        )

        solver = OptimizationSolver(qubo)
        report = solver.solve(config=config)
        self.assertEqual(qubo.evaluate_cost(report.best_state), optimal_cost)

    def test_qubo_solution_watts_strogatz(self):
        """Tests that a solution with the optimal cost is found by
        OptimizationSolver with the QUBO formulation."""
        problem = MISProblem.from_watts_strogatz(n=25, k=5, p=0.9, seed=0)
        qubo = problem.get_as_qubo(w_diag=1, w_off=4)
        optimal_cost = - \
            np.count_nonzero(problem.find_maximum_independent_set())

        config = SolverConfig(
            timeout=3000,
            target_cost=optimal_cost,
            backend="CPU",
            hyperparameters={
                "neuron_model": "nebm",
                "temperature": int(1),
                "refract": np.random.randint(1, 5, size=(25, ))
            }
        )

        solver = OptimizationSolver(qubo)
        report = solver.solve(config=config)
        self.assertEqual(qubo.evaluate_cost(report.best_state), optimal_cost)


if __name__ == "__main__":
    unittest.main()
