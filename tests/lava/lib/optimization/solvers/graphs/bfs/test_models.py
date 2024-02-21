# Code to test Process interface of Breadth First Search based graph search

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

# Tests Hierarchical process models in BFS model

import unittest
from lava.lib.optimization.solvers.qubo.solution_readout.process import (
    SolutionReadoutEthernet,
    SpikeIntegrator,
)
import numpy as np

from lava.lib.optimization.solvers.graphs.bfs.process import (
    BFSNeuron,
    BreadthFirstSearch,
)
from lava.lib.optimization.solvers.graphs.bfs.models import (
    BreadthFirstSearchModel,
)


connection_config = []


class TestBreadthFirstSearchModel(unittest.TestCase):
    """Test the models chosen in Breadth First Search
    Breadth First Search
    """

    def test_initalization_of_process(self):
        np.random.seed(5)
        dim = 5
        # Generate random symmetric matrix with 1s and 0s
        rand_mat = np.random.randint(2, size=(dim, dim))
        sym_mat = np.triu(rand_mat) + np.triu(rand_mat).T
        sym_mat = np.where(sym_mat != 0, 1, 0)

        # Specify start and target nodes
        status = np.array([0, 128, 0, 64, 0])

        process = BreadthFirstSearch(
            connection_config,
            adjacency_matrix=sym_mat,
            node_description=status,
        )

        model_obj = BreadthFirstSearchModel(process)
        graph_edges = model_obj.graph_edges.weights.get().toarray()
        self.assertTrue(model_obj.num_nodes == dim)
        self.assertTrue(np.array_equal(sym_mat, graph_edges))
        self.assertIsInstance(
            model_obj.distributor_neurons,
            SpikeIntegrator,
            "Distributor neurons have not been made SpikeIntegrator objects",
        )
        self.assertIsInstance(
            model_obj.aggregator_neurons,
            SpikeIntegrator,
            "Aggregator neurons have not been made SpikeIntegrator objects",
        )
        self.assertIsInstance(
            model_obj.graph_nodes,
            BFSNeuron,
            "Graph Nodes have not been made BFSNeuron objects",
        )
        self.assertIsInstance(
            model_obj.solution_readout,
            SolutionReadoutEthernet,
            "Solution readout object is not properly defined",
        )

    def test_connectivity_matrices(self):
        np.random.seed(5)
        dim = 7
        # Generate random symmetric matrix with 1s and 0s
        rand_mat = np.random.randint(2, size=(dim, dim))
        sym_mat = np.triu(rand_mat) + np.triu(rand_mat).T
        sym_mat = np.where(sym_mat != 0, 1, 0)

        # Specify start and target nodes
        status = np.array([0, 128, 0, 64, 0, 0, 0])

        process = BreadthFirstSearch(
            connection_config,
            adjacency_matrix=sym_mat,
            node_description=status,
            num_nodes_per_aggregator=3,
            num_nodes_per_distributor=2,
        )

        model_obj = BreadthFirstSearchModel(process)
        graph_neur_to_agg = (
            model_obj.graph_neur_to_agg_conn.weights.get().toarray()
        )
        test_graph_neur_to_agg = np.array(
            [
                [1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )

        agg_to_glbl_depth = (
            model_obj.agg_to_glbl_dpth_conn.weights.get().toarray()
        )
        test_agg_to_glbl_depth = np.array([[1, 1, 1]])

        graph_neur_to_dist = (
            model_obj.dist_to_graph_neur_conn.weights.get().toarray()
        )

        test_dist_to_graph_neur = np.array(
            [
                [1, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        ).T

        glbl_depth_to_dist = (
            model_obj.glbl_dpth_to_dist_conn.weights.get().toarray()
        )
        test_glbl_depth_to_dist = np.array([[1, 1, 1, 1]]).T

        self.assertTrue(
            np.array_equal(test_graph_neur_to_agg, graph_neur_to_agg)
        )
        self.assertTrue(
            np.array_equal(test_agg_to_glbl_depth, agg_to_glbl_depth)
        )

        self.assertTrue(
            np.array_equal(test_glbl_depth_to_dist, glbl_depth_to_dist)
        )

        self.assertTrue(
            np.array_equal(test_dist_to_graph_neur, graph_neur_to_dist)
        )

    def test_connectivity_matrices_if_agg_dist_per_neuron_more_than_graph_nodes(
        self,
    ):
        np.random.seed(5)
        dim = 9
        # Generate random symmetric matrix with 1s and 0s
        rand_mat = np.random.randint(2, size=(dim, dim))
        sym_mat = np.triu(rand_mat) + np.triu(rand_mat).T
        sym_mat = np.where(sym_mat != 0, 1, 0)

        # Specify start and target nodes
        status = np.array([0, 128, 0, 64, 0, 0, 0, 0, 0])

        # increasing number of nodes per agg/dist beyond num_nodes
        process = BreadthFirstSearch(
            connection_config,
            adjacency_matrix=sym_mat,
            node_description=status,
            num_nodes_per_aggregator=50,
            num_nodes_per_distributor=40,
        )

        model_obj = BreadthFirstSearchModel(process)
        graph_neur_to_agg = (
            model_obj.graph_neur_to_agg_conn.weights.get().toarray()
        )
        test_graph_neur_to_agg = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1]])

        agg_to_glbl_depth = (
            model_obj.agg_to_glbl_dpth_conn.weights.get().toarray()
        )
        test_agg_to_glbl_depth = np.array([[1]])

        graph_neur_to_dist = (
            model_obj.dist_to_graph_neur_conn.weights.get().toarray()
        )

        test_dist_to_graph_neur = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1]]).T

        glbl_depth_to_dist = (
            model_obj.glbl_dpth_to_dist_conn.weights.get().toarray()
        )
        test_glbl_depth_to_dist = np.array([[1]]).T

        self.assertTrue(
            np.array_equal(test_graph_neur_to_agg, graph_neur_to_agg)
        )
        self.assertTrue(
            np.array_equal(test_agg_to_glbl_depth, agg_to_glbl_depth)
        )

        self.assertTrue(
            np.array_equal(test_glbl_depth_to_dist, glbl_depth_to_dist)
        )

        self.assertTrue(
            np.array_equal(test_dist_to_graph_neur, graph_neur_to_dist)
        )


if __name__ == "__main__":
    unittest.main()
