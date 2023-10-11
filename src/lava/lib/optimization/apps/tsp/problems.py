# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import networkx as ntx
import numpy as np
import typing as ty


class TravellingSalesmanProblem:
    """Travelling Salesman Problem specification.

    N customer nodes need to be visited by a travelling salesman,
    while minimizing the overall distance of the traversal.
    """
    def __init__(self,
                 waypt_coords: ty.List[ty.Tuple[int, int]],
                 starting_pt: ty.Tuple[int, int],
                 edges: ty.Optional[ty.List[ty.Tuple[int, int]]] = None):
        """
        Parameters
        ----------
        node_coords : list(tuple(int, int))
            A list of integer tuples corresponding to node coordinates.
            Nodes signify the "customers" in a VRP, which need to be visited
            by the vehicles.
        vehicle_coords : list(tuple(int, int))
            A list of integer tuples corresponding to the initial vehicle
            coordinates. If the length of the list is 1, then it is
            assumed that all vehicles begin from the same depot.
        edges: (Optional) list(tuple(int, int))
            An optional list of edges connecting nodes, given as a list of
            node ID pairs. If None provided, assume all-to-all connectivity
            between nodes.

        Notes
        -----
        The vehicle IDs and node IDs are assigned serially. The IDs 1 to M
        correspond to vehicles and (M+1) to (M+N) correspond to nodes to be
        visited by the vehicles.
        """
        super().__init__()
        self._waypt_coords = waypt_coords
        self._starting_pt_coords = starting_pt
        self._num_waypts = len(self._waypt_coords)
        self._starting_pt_id = 1
        self._waypt_ids = list(np.arange(2, self._num_waypts + 2))
        self._nodes = {self._starting_pt_id: self._starting_pt_coords}
        self._nodes.update(dict(zip(self._waypt_ids, self._waypt_coords)))
        if edges:
            self._edges = edges
        else:
            self._edges = []

        self._problem_graph = None

    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, nodes: ty.Dict[int, ty.Tuple[int, int]]):
        self._nodes = nodes

    @property
    def node_ids(self):
        return list(self._nodes.keys())

    @property
    def node_coords(self):
        return list(self._nodes.values())

    @property
    def num_nodes(self):
        return len(list(self._nodes.keys()))

    @property
    def edges(self):
        return self._edges

    @property
    def waypt_coords(self):
        return self._waypt_coords

    @property
    def waypt_ids(self):
        return self._waypt_ids

    @property
    def num_waypts(self):
        return len(self._waypt_coords)

    @property
    def problem_graph(self):
        """NetworkX problem graph is created and returned.

        If edges are specified, they are taken into account.
        Returns
        -------
        A graph object corresponding to the problem.
        """
        if not self._problem_graph:
            self._generate_problem_graph()
        return self._problem_graph

    def _generate_problem_graph(self):
        if len(self.edges) > 0:
            gph = ntx.DiGraph()
            # Add the nodes to be visited
            gph.add_nodes_from(self.node_ids)
            # If there are user-provided edges, add them between the nodes
            gph.add_edges_from(self.edges)
        else:
            gph = ntx.complete_graph(self.node_ids, create_using=ntx.DiGraph())

        ntx.set_node_attributes(gph, self.nodes, name="Coordinates")

        # Compute Euclidean distance along all edges and assign them as edge
        # weights
        # ToDo: Replace the loop with independent distance matrix computation
        #  and then assign the distances as attributes
        for edge in gph.edges.keys():
            gph.edges[edge]["cost"] = np.linalg.norm(
                np.array(gph.nodes[edge[1]]["Coordinates"]) - np.array(
                    gph.nodes[edge[0]]["Coordinates"]))

        self._problem_graph = gph
