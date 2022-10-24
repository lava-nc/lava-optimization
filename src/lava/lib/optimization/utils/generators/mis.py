# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import numpy.typing as npt
import networkx as netx
from lava.lib.optimization.problems.problems import QUBO


class MISProblem:
    def __init__(self, num_vertices=45, connection_probability=0.7, seed=None):
        """A maximum-independent set problem for a random graph defined by the
        number of vertices and connection probability specified by the user."""
        self._num_vertices = num_vertices
        self._connection_probability = connection_probability
        self._seed = seed
        self._adjacency = self._get_random_graph_mtx()

    @property
    def num_vertices(self):
        return self._num_vertices

    @property
    def connection_probability(self):
        return self._connection_probability

    @property
    def seed(self):
        return self._seed

    def get_graph(self):
        graph = self._get_graph_from_adjacency_matrix(self._adjacency)
        return graph

    def get_graph_matrix(self):
        return self._adjacency

    def get_complement_graph(self):
        c_adjacency = self._get_adjacency_of_complement_graph(self._adjacency)
        c_graph = self._get_graph_from_adjacency_matrix(c_adjacency)
        return c_graph

    def get_complement_graph_matrix(self):
        c_adjacency = self._get_adjacency_of_complement_graph(self._adjacency)
        return c_adjacency

    def get_as_qubo(self, w_diag, w_off):
        q = self._get_qubo_cost_from_adjacency(self._adjacency, w_diag, w_off)
        qubo = QUBO(q)
        return qubo

    def set_parameters(self, num_vertices, connection_probability, seed):
        self._num_vertices = num_vertices
        self._connection_probability = connection_probability
        self._seed = seed

    def _get_random_graph_mtx(self):
        """Creates an undirected graph with random connectivity between nodes
        and returns its adjacency matrix.

        Parameters
        ----------
        num_vertices : int
            Number of vertices in the graph
        connection_probability: float
            Random probability [0, 1] that two vertices are connected
        seed: int
            Seed for random number calculator

        Returns
        -------
        adj : Array[binary]
            Adjacency matrix
        """

        np.random.seed(self.seed)

        # generate a random binary matrix of size n_vert x n_vert
        adjacency = (np.random.rand(self.num_vertices, self.num_vertices) <
                     self.connection_probability).astype(int)

        # delete diagonal elements as nodes have no self-connectivity
        adjacency = np.triu(adjacency)
        # ensure that the matrix is symmetric
        adjacency += adjacency.T - 2 * np.diag(adjacency.diagonal())

        return adjacency

    def _get_graph_from_adjacency_matrix(self, adjacency_matrix):
        """Create a Networkx graph from an adjacency matrix"""
        num_vertices = adjacency_matrix.shape[0]
        # create Graph
        G = netx.Graph()
        G.add_nodes_from(range(num_vertices))

        # add edges
        for v1 in range(num_vertices):
            for v2 in range(v1 + 1, num_vertices):
                if adjacency_matrix[v1, v2] == 1:
                    G.add_edge(v1, v2)
        return G

    def _get_adjacency_of_complement_graph(self,
                                           adjacency_matrix: npt.ArrayLike):
        """Get the adjacency matrix for the graph H={V,E_c}, where E_c is the
        complement of E for the graph G={V,E}.
        """
        adj_mc = np.logical_not(adjacency_matrix)
        adj_mc = adj_mc.astype(int)
        adj_mc = adj_mc - np.diag(adj_mc.diagonal())
        return adj_mc

    def _get_qubo_cost_from_adjacency(self, adjacency_matrix, w_diag, w_off):
        """Creates a QUBO whose solution corresponds to the maximum independent
        set (MIS) of the graph defined by the input adjacency matrix.

        The goal of the QUBO is to minimize the cost
            min x^T * Q * x ,
        where the vector x is defined as:
            x_i = 1 if vertex i is part of the MIs
            x_i = 0 if vertex i is not part of the MIS,
        and the QUBO matrix is given by
            Q_ii = w_diag
            Q_ij = w_off (for i~=j) .

        Parameters
        ----------
        adjacency_matrix : Array[binary]
            Adjacency matrix
        w_diag: float
            Weights of diagonal elements of Q.
        w_off: int
            Weights of off-diagonal elements of Q

        Returns
        -------
        q : Array[float, float]
            2D QUBO matrix.
        """

        if w_off <= 2 * w_diag:
            raise ValueError(
                "Off-diagonal weights must be > 2 x diagonal weights.")

        # Translate the connectivity matrix to a QUBO matrix
        num_variables = adjacency_matrix.shape[0]
        q = - w_diag * np.eye(num_variables) + w_off / 2 * adjacency_matrix
        return q.astype(int)


def find_maximum_independent_set(mis_problem: MISProblem) -> \
        npt.ArrayLike:
    """Find and return the maximum independent set of a graph based on its
    adjacency matrix.

    Uses Networkx to solve the equivalent maximum clique problem.
    Get a graph whose maximum clique corresponds to the maximum independent
    set of that defined by the input connectivity matrix.

    The  maximum independent set for the graph G={V,E} is equivalent to the
    maximum-clique of the graph H={V,E_c}, where E_c is the complement of E.

    Parameters
    ----------
    adjacency_matrix: An array encoding the connectivity of a random
    graph, a value in the array represents the weight of an edge between
    two graph vertices given by the row and column indices.

    Returns
    -------
    solution:  Array[binary]
    Vector of length equal to the number of vertices in the graph.
    The ith entry of the vector determines if the ith vertex is a
    member of the MIS.
    """
    c_graph = mis_problem.get_complement_graph()
    maximum_clique = find_maximum_clique(undirected_graph=c_graph)
    mis = indices_to_binary_vector(maximum_clique, mis_problem.num_vertices)
    return mis


def find_maximum_clique(undirected_graph: netx.Graph):
    """Find the maximum clique problem for a given undirected graph."""
    maximum_clique, weights = netx.max_weight_clique(undirected_graph,
                                                     weight=None)
    return maximum_clique


def indices_to_binary_vector(indices: npt.ArrayLike, lenght: int):
    """Transform an array of node indices into a one-hot encoding given by a
    binary array where values are 1 if value index is present in indices. """
    bin_vector = np.zeros((lenght,))
    bin_vector[indices] = 1
    return bin_vector
