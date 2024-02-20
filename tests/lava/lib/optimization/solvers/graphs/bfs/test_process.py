# Code to test Process interface of Breadth First Search based graph search

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

# Initialization tests for all the processes in BFS
import unittest
import numpy as np

from lava.lib.optimization.solvers.graphs.bfs.process import (
    BreadthFirstSearch,
    BFSNeuron,
)

connection_config = []


class TestBFShierarchicalProcess(unittest.TestCase):
    """Test the initialization of the hierarchical process interface for
    Breadth First Search
    """

    def test_hier_process_bfs(self):
        np.random.seed(5)
        dim = 5
        rand_mat = np.random.randint(2, size=(dim, dim))
        sym_mat = np.triu(rand_mat) + np.triu(rand_mat).T
        sym_mat = np.where(sym_mat != 0, 1, 0)
        process = BreadthFirstSearch(
            connection_config, adjacency_matrix=sym_mat
        )

        self.assertEqual(
            np.all(process.vars.adjacency_matrix.get() == sym_mat), True
        )

        # test port shapes
        self.assertEqual(np.all(process.start_search.shape == (2,)), True)

    def test_hier_process_not_sym_connections(self):
        np.random.seed(4)
        dim = 5
        rand_mat = np.random.randint(2, size=(dim, dim))
        with self.assertRaises(ValueError):
            BreadthFirstSearch(connection_config, adjacency_matrix=rand_mat)

    def test_hier_process_incorrect_matrix_entries(self):
        np.random.seed(3)
        dim = 5
        rand_mat = np.random.randint(5, size=(dim, dim))
        with self.assertRaises(ValueError):
            BreadthFirstSearch(connection_config, adjacency_matrix=rand_mat)

        rand_mat = np.random.randint(-5, 4, size=(dim, dim))
        with self.assertRaises(ValueError):
            BreadthFirstSearch(connection_config, adjacency_matrix=rand_mat)


class TestBFSNeuronProcess(unittest.TestCase):
    """Test the initialization of the process parameters for the BFS neuron
    model
    """

    def test_bfs_neuron_process(self):
        # neuron 1 is start node and neuron 4 is destination
        dim = 6
        status = np.array([0, 128, 0, 0, 64, 0])
        process = BFSNeuron(shape=(dim,), status=status)

        # test vars
        self.assertEqual(np.all(process.vars.counter_mem.get() == 0), True)
        self.assertEqual(np.all(process.vars.global_depth.get() == 0), True)
        self.assertEqual(np.all(process.vars.status_reg.get() == status), True)

        # test port shapes
        self.assertEqual(np.all(process.a_in_1.shape == (dim,)), True)
        self.assertEqual(np.all(process.a_in_2.shape == (dim,)), True)
        self.assertEqual(np.all(process.a_in_3.shape == (dim,)), True)
        self.assertEqual(np.all(process.a_in_4.shape == (dim,)), True)

        self.assertEqual(np.all(process.s_out_1.shape == (dim,)), True)
        self.assertEqual(np.all(process.s_out_2.shape == (dim,)), True)
        self.assertEqual(np.all(process.s_out_3.shape == (dim,)), True)
        self.assertEqual(np.all(process.s_out_4.shape == (dim,)), True)

    def test_insane_status_values(self):
        # pass nonsensical values for status register here
        dim = 7
        # 2 destinations
        status = np.array([0, 128, 0, 0, 64, 0, 64])
        with self.assertRaises(AssertionError):
            BFSNeuron(shape=(dim,), status=status)

        # No destinations
        status = np.array([0, 128, 0, 0, 0, 0, 0])
        with self.assertRaises(AssertionError):
            BFSNeuron(shape=(dim,), status=status)

        # 3 starts
        status = np.array([0, 128, 128, 128, 64, 0, 0])
        with self.assertRaises(AssertionError):
            BFSNeuron(shape=(dim,), status=status)

        # No starts
        status = np.array([0, 0, 0, 64, 0, 0, 0])
        with self.assertRaises(AssertionError):
            BFSNeuron(shape=(dim,), status=status)

        # unallowed values
        status = np.array([64, 62, 0, 128, 0, 0, 0])
        with self.assertRaises(AssertionError):
            BFSNeuron(shape=(dim,), status=status)


if __name__ == "__main__":
    unittest.main()
