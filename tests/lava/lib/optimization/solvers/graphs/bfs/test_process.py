# Code to test Process interface of Breadth First Search based graph search

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

# Initialization tests for all the processes in QP
import unittest
import numpy as np

from lava.lib.optimization.solvers.graphs.bfs.process import (
    BreadthFirstSearch,
    BFSNeuron
)

connection_config = []

class TestBFShierarchicalProcess(unittest.TestCase):
    """ Test the initialization of the hierarchical process interface for 
    Breadth First Search
    """

    def test_hier_process_bfs(self):
        rand_mat = np.random.randint(2, size=(5, 5))
        sym_mat = np.triu(rand_mat) + np.triu(rand_mat).T
        process = BreadthFirstSearch(connection_config,
                                     adjacency_matrix=sym_mat)
        
        self.assertEqual(np.all(process.vars.adjacency_matrix.get() == sym_mat), 
                         True)
        
        # test port shapes

    def test_hier_process_not_sym_connections(self):
        pass

class TestBFSNeuronProcess(unittest.TestCase):
   """ Test the initialization of the process parameters for the BFS neuron 
   model
   """
   def test_bfs_neuron_process(self):
        process = BFSNeuron()

        # test port shapes 
        # 
        
   def test_insane_status_values(self):
       # pass nonsensical values for status register here
       pass

if __name__ == "__main__":
    unittest.main()