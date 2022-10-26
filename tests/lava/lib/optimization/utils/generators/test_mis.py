# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import networkx as netx
import numpy as np

from lava.lib.optimization.problems.problems import QUBO
from lava.lib.optimization.utils.generators.mis import MISProblem


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
        w_diag = 1.0
        w_off = 4.0
        qubo = self.problem.get_as_qubo(w_diag, w_off)
        self.assertIsInstance(qubo, QUBO)

    def test_find_maximum_independent_set(self):
        """Tests that find_maximum_independent_set returns the correct type."""
        mis = self.problem.find_maximum_independent_set()
        self.assertIsInstance(mis, np.ndarray)
