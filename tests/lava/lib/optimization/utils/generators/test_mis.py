# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

import numpy as np

from lava.lib.optimization.utils.generators.mis import MISProblem


class TestMISProblem(unittest.TestCase):
    """Unit tests for MISProblem class."""

    def setUp(self):
        self.adjacency = np.array(
            [[0, 1, 1, 0], [1, 0, 0, 0], [1, 0, 0, 1], [0, 0, 1, 0]]
        )
        self.num_vertices = self.adjacency.shape[0]
        self.num_edges = np.count_nonzero(self.adjacency)
        self.problem = MISProblem(adjacency_matrix=self.adjacency)

    def test_create_obj(self):
        """Tests correct instantiation of MISProblem object."""
        self.assertIsInstance(self.problem, MISProblem)

    def test_num_vertices_property(self):
        """Tests correct value of num_vertices property."""
        self.assertEqual(self.problem.num_vertices, self.num_vertices)

    def test_num_edges_property(self):
        """Tests correct value of num_edges property."""
        self.assertEqual(self.problem.num_edges, self.num_edges)

    def test_adjacency_matrix_property(self):
        """Tests correct value of adjacency_matrix."""
        self.assertTrue((self.adjacency == self.problem.adjacency_matrix).all())

    def test_get_graph(self):
        """Tests that the graph contains the correct number of nodes and
        edges."""
        graph = self.problem.get_graph()
        self.assertEqual(graph.number_of_nodes(), self.num_vertices)
        self.assertEqual(graph.number_of_edges() * 2, self.num_edges)

    def test_get_complement_graph(self):
        """Tests that the complement graph contains the correct number of
        nodes and edges."""
        graph = self.problem.get_complement_graph()
        self.assertEqual(graph.number_of_nodes(), self.num_vertices)
        self.assertEqual(graph.number_of_edges() * 2, self.num_edges)

    def test_get_complement_graph_matrix(self):
        """Tests the correct complement graph adjacency matrix is returned."""
        matrix = self.problem.get_complement_graph_matrix()
        correct_matrix = np.array(
            [[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 0, 0], [1, 1, 0, 0]]
        )
        self.assertTrue((matrix == correct_matrix).all())

    def test_get_as_qubo(self):
        """Tests the conversion to QUBO returns the correct cost matrix."""
        w_diag = 1
        w_off = 4
        qubo = self.problem.get_as_qubo(w_diag, w_off)
        self.assertTrue(np.all(qubo.q[np.diag_indices(4)] == -w_diag))
        self.assertTrue(np.all(qubo.q[qubo.q > 0] == w_off / 2))

    def test_find_maximum_independent_set(self):
        """Tests the correct maximum independent set is returned."""
        mis = self.problem.find_maximum_independent_set()
        correct_set_size = 2
        self.assertEqual(mis.sum(), correct_set_size)


if __name__ == "__main__":
    unittest.main()
