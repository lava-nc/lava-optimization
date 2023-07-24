# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/


import numpy as np
import numpy.typing as npt
import networkx as netx

from lava.lib.optimization.problems.problems import QUBO


class MISProblem:
    """
    Utility class to instantiate Maximum Independent Set problems and
    convert them to a QUBO formulation.
    """

    def __init__(self, num_vertices: int, connection_prob: float,
                 seed: int = None):
        """
        Instantiate a new random MIS problem, with the given number of
        vertices and connection probability.

        Parameters
        ----------
        num_vertices: int
            Number of vertices of the random graph
        connection_prob: float
            Connection probability
        seed: int, optional
            Seed for numpy random generator
        """
        self._num_vertices = num_vertices
        self._connection_prob = connection_prob
        self._seed = seed
        self._adjacency = self._get_random_graph_mtx()

    @property
    def num_vertices(self) -> int:
        """Returns the number of vertices in the graph."""
        return self._num_vertices

    @property
    def connection_prob(self) -> float:
        """Returns the connection probability of the graph."""
        return self._connection_prob

    @property
    def seed(self):
        """Returns the seed used for generating the graph."""
        return self._seed

    @staticmethod
    def _get_graph_from_adjacency_matrix(adjacency_matrix):
        num_vertices = adjacency_matrix.shape[0]
        # create Graph
        G = netx.Graph()
        G.add_nodes_from(range(num_vertices))

        for v1 in range(num_vertices):
            for v2 in range(v1 + 1, num_vertices):
                if adjacency_matrix[v1, v2] == 1:
                    G.add_edge(v1, v2)
        return G

    @staticmethod
    def _get_adjacency_of_complement_graph(
            adjacency_matrix: np.ndarray) -> np.ndarray:
        adj_mc = np.logical_not(adjacency_matrix)
        adj_mc = adj_mc.astype(int)
        adj_mc = adj_mc - np.diag(adj_mc.diagonal())
        return adj_mc

    @staticmethod
    def _get_qubo_cost_from_adjacency(adjacency_matrix: np.ndarray,
                                      w_diag: float,
                                      w_off: float) -> np.ndarray:
        if w_off <= 2 * w_diag:
            raise ValueError(
                "Off-diagonal weights must be > 2 x diagonal weights.")

        num_variables = adjacency_matrix.shape[0]
        q = - w_diag * np.eye(num_variables) + w_off / 2 * adjacency_matrix
        return q.astype(int)

    def get_graph(self) -> netx.Graph:
        """Returns the graph in networkx format."""
        graph = self._get_graph_from_adjacency_matrix(self._adjacency)
        return graph

    def get_graph_matrix(self) -> np.ndarray:
        """Returns the adjacency matrix of the graph."""
        return self._adjacency

    def get_complement_graph(self) -> netx.Graph:
        """Returns the complement graph in networkx format."""
        c_adjacency = self.get_complement_graph_matrix()
        c_graph = self._get_graph_from_adjacency_matrix(c_adjacency)
        return c_graph

    def get_complement_graph_matrix(self) -> np.ndarray:
        """Returns the adjacency matrix of the complement graph."""
        c_adjacency = np.logical_not(self._adjacency)
        c_adjacency = c_adjacency.astype(int)
        c_adjacency = c_adjacency - np.diag(c_adjacency.diagonal())
        return c_adjacency

    def get_as_qubo(self, w_diag: float, w_off: float) -> QUBO:
        """
        Creates a QUBO whose solution corresponds to the maximum independent
        set (MIS) of the graph defined by the input adjacency matrix.

        The goal of the QUBO is to minimize the cost
            min x^T * Q * x ,
        where the vector x is defined as:
            x_i = 1 if vertex i is part of the MIS
            x_i = 0 if vertex i is not part of the MIS,
        and the QUBO matrix is given by
            Q_ii = w_diag
            Q_ij = w_off (for i!=j) .
        """
        q = self._get_qubo_cost_from_adjacency(self._adjacency, w_diag, w_off)
        qubo = QUBO(q)
        return qubo

    def get_qubo_matrix(self, w_diag: float, w_off: float) -> np.ndarray:
        """
        Creates a QUBO whose solution corresponds to the maximum independent
        set (MIS) of the graph defined by the input adjacency matrix.

        The goal of the QUBO is to minimize the cost
            min x^T * Q * x ,
        where the vector x is defined as:
            x_i = 1 if vertex i is part of the MIS
            x_i = 0 if vertex i is not part of the MIS,
        and the QUBO matrix is given by
            Q_ii = w_diag
            Q_ij = w_off (for i!=j) .
        """
        return self._get_qubo_cost_from_adjacency(self._adjacency,
                                                  w_diag,
                                                  w_off)

    def _get_random_graph_mtx(self) -> np.ndarray:
        """
        Creates an undirected graph with random connectivity between nodes
        and returns its adjacency matrix.
        """
        np.random.seed(self.seed)

        random_matrix = np.random.rand(self.num_vertices, self.num_vertices)
        adjacency = (random_matrix < self.connection_prob).astype(int)

        adjacency = np.triu(adjacency)
        adjacency += adjacency.T - 2 * np.diag(adjacency.diagonal())

        return adjacency

    def find_maximum_independent_set(self) -> \
            npt.ArrayLike:
        """
        Find and return the maximum independent set of a graph based on its
        adjacency matrix.

        *Please note that this function addresses the maximum and not just
        the maximal independent set. A maximal independent set is an
        independent set that is not a subset of any other independent set.
        The largest of these sets is the maximum independent set, which is
        determined by the present function.*Uses Networkx to solve the
        equivalent maximum clique problem.

        Get a graph whose maximum clique corresponds to the maximum independent
        set of that defined by the input connectivity matrix.
        The  maximum independent set for the graph G={V,E} is equivalent to the
        maximum-clique of the graph H={V,E_c}, where E_c is the complement of E.

        Returns
        -------
        solution:  Array[binary]
            Vector of length equal to the number of vertices in the graph.
            The ith entry of the vector determines if the ith vertex is a
            member of the MIS.
        """
        c_graph = self.get_complement_graph()
        maximum_clique, weights = netx.max_weight_clique(c_graph, weight=None)

        # convert array of indices to binary array
        mis = np.zeros((self.num_vertices,))
        mis[maximum_clique] = 1
        return mis
