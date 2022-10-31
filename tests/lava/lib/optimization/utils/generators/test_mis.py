# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import networkx as netx
import numpy as np

from lava.lib.optimization.problems.problems import QUBO
from lava.lib.optimization.solvers.generic.solver import OptimizationSolver
from lava.lib.optimization.utils.generators.mis import MISProblem


class TestMISProblemInterface(unittest.TestCase):
    def setUp(self):
        self.num_vertices = 15
        self.connection_prob = 0.75
        self.seed = 42
        self.problem = MISProblem(num_vertices=self.num_vertices,
                                  connection_prob=self.connection_prob,
                                  seed=self.seed)

    def test_create_obj(self):
        self.assertIsInstance(self.problem, MISProblem)

    def test_num_vertices_prop(self):
        self.assertEqual(self.problem.num_vertices, self.num_vertices)

    def test_connection_prob_prop(self):
        self.assertEqual(self.problem.connection_prob, self.connection_prob)

    def test_seed_prob(self):
        self.assertEqual(self.problem.seed, self.seed)

    def test_get_graph(self):
        graph = self.problem.get_graph()
        self.assertIsInstance(graph, netx.Graph)

    def test_get_graph_matrix(self):
        matrix = self.problem.get_graph_matrix()
        self.assertIsInstance(matrix, np.ndarray)
        m, n = matrix.shape
        self.assertEqual(m, self.num_vertices)
        self.assertEqual(n, self.num_vertices)

    def test_get_complement_graph(self):
        graph = self.problem.get_complement_graph()
        self.assertIsInstance(graph, netx.Graph)

    def test_get_complement_graph_matrix(self):
        matrix = self.problem.get_complement_graph_matrix()
        self.assertIsInstance(matrix, np.ndarray)
        m, n = matrix.shape
        self.assertEqual(m, self.num_vertices)
        self.assertEqual(n, self.num_vertices)

    def test_get_as_qubo(self):
        w_diag = 1.0
        w_off = 4.0
        qubo = self.problem.get_as_qubo(w_diag, w_off)
        self.assertIsInstance(qubo, QUBO)

    def test_find_maximum_independent_set(self):
        mis = self.problem.find_maximum_independent_set()
        self.assertIsInstance(mis, np.ndarray)


class TestMISProblemSolution(unittest.TestCase):

    def setUp(self):
        self.num_vertices = 25
        self.connection_prob = 0.75
        self.seed = 42
        self.problem = MISProblem(num_vertices=self.num_vertices,
                                  connection_prob=self.connection_prob,
                                  seed=self.seed)

        self.qubo_problem = self.problem.get_as_qubo(w_diag=1, w_off=4)
        solution_opt = self.problem.find_maximum_independent_set()
        self.cost_opt = self.qubo_problem.evaluate_cost(solution=solution_opt)

    def test_qubo_solution(self):
        params = {"timeout": 5000,
                  "target_cost": self.cost_opt,
                  "backend": "CPU",
                  "hyperparameters": {
                      "steps_to_fire": 11,
                      "noise_amplitude": 1,
                      "noise_precision": 4,
                      "step_size": 11,
                  }}

        solver = OptimizationSolver(self.qubo_problem)
        solution = solver.solve(**params)
        self.assertEqual(self.qubo_problem.evaluate_cost(solution),
                         self.cost_opt)


if __name__ == "__main__":
    unittest.main()
