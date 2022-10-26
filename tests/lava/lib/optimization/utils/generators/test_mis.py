# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import networkx as netx
import numpy as np

from lava.lib.optimization.problems.problems import QUBO
from lava.lib.optimization.utils.generators.mis import MISProblem, \
    find_maximum_independent_set, indices_to_binary_vector


class TestMISProblem(unittest.TestCase):
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
        self.assertEqual(matrix.shape[0], self.num_vertices)
        self.assertEqual(matrix.shape[1], self.num_vertices)

    def test_get_complement_graph(self):
        graph = self.problem.get_complement_graph()
        self.assertIsInstance(graph, netx.Graph)

    def test_get_complement_graph_matrix(self):
        matrix = self.problem.get_complement_graph_matrix()
        self.assertIsInstance(matrix, np.ndarray)
        self.assertEqual(matrix.shape[0], self.num_vertices)
        self.assertEqual(matrix.shape[1], self.num_vertices)

    def test_get_as_qubo(self):
        qubo = self.problem.get_as_qubo(1.0, 4.0)
        self.assertIsInstance(qubo, QUBO)


class TestFindMaximumIndependentSet(unittest.TestCase):
    def setUp(self):
        self.num_vertices = 5
        self.connection_prob = 0.75
        self.seed = 42
        self.problem = MISProblem(num_vertices=self.num_vertices,
                                  connection_prob=self.connection_prob,
                                  seed=self.seed)

    def test_find_maximum_independent_set(self):
        mis = find_maximum_independent_set(self.problem)
        self.assertIsInstance(mis, np.ndarray)


class TestIndicestoBinaryVector(unittest.TestCase):

    def test_function(self):
        indices = [2, 3, 8]
        lenght = 10
        result = indices_to_binary_vector(indices, lenght)
        expected = [0, 0, 1, 1, 0, 0, 0, 0, 1, 0]
        self.assertTrue(np.all(expected == result))


if __name__ == '__main__':
    unittest.main()
