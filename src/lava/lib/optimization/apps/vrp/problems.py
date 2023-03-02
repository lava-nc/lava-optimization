# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import networkx as ntx
import numpy as np
import typing as ty


class VRP:
    """Vehicle Routing Problem specification.

    N customer nodes need to be visited by M vehicles, while minimizing the
    overall cost of the traversal.

    For complete specification of a VRP, a dictionary mapping the node IDs to
    the node coordinates and a dictionary mapping the vehicle IDs to their
    initial coordinates are sufficient.
    """
    def __init__(self,
                 node_coords: ty.List[ty.Tuple[int, int]],
                 vehicle_coords: ty.Union[int, ty.List[ty.Tuple[int, int]]],
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
        """
        super().__init__()
        self._node_coords = node_coords
        self._vehicle_coords = vehicle_coords
        self._num_nodes = len(self._node_coords)
        self._num_vehicles = len(self._vehicle_coords)
        self._vehicle_ids = list(np.arange(1, self._num_vehicles + 1))
        self._node_ids = list(np.arange(
            self._num_vehicles + 1, self._num_vehicles + self._num_nodes + 1))
        self._nodes = dict(zip(self._node_ids, self._node_coords))
        if edges:
            self._edges = edges
        else:
            self._edges = []
        self._vehicles = dict(zip(self._vehicle_ids, self._vehicle_coords))
        self._problem_graph = self._generate_problem_graph()

    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, nodes: ty.Dict[int, ty.Tuple[int, int]]):
        self._nodes = nodes

    @property
    def node_ids(self):
        return self._node_ids

    @property
    def node_coords(self):
        return self._node_coords

    @property
    def num_nodes(self):
        return self._num_nodes

    @property
    def edges(self):
        return self._edges

    @property
    def vehicles(self):
        return self._vehicles

    @vehicles.setter
    def vehicles(self, vehicles: ty.Dict[int, ty.Tuple[int, int]]):
        self._vehicles = vehicles

    @property
    def vehicle_ids(self):
        return self._vehicle_ids

    @property
    def vehicle_init_coords(self):
        return self._vehicle_coords

    @property
    def num_vehicles(self):
        return self._num_vehicles

    @property
    def problem_graph(self):
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

        node_type_dict = dict(zip(self.node_ids,
                                  ["Node"] * len(self.node_ids)))
        # Associate node type as "Node" and node coordinates as attributes
        ntx.set_node_attributes(gph, node_type_dict, name="Type")
        ntx.set_node_attributes(gph, self.nodes, name="Coordinates")

        # Add vehicles as nodes
        gph.add_nodes_from(self.vehicle_ids)
        # Associate node type as "Vehicle" and vehicle coordinates as attributes
        vehicle_type_dict = dict(zip(self.vehicle_ids,
                                     ["Vehicle"] * len(self.vehicle_ids)))
        ntx.set_node_attributes(gph, vehicle_type_dict, name="Type")
        ntx.set_node_attributes(gph, self.vehicles, name="Coordinates")

        # Add edges from initial vehicle positions to all nodes (oneway edges)
        for vid in self.vehicle_ids:
            for nid in self.node_ids:
                gph.add_edge(vid, nid)

        # Compute Euclidean distance along all edges and assign them as edge
        # weights
        # ToDo: Replace the loop with independent distance matrix computation
        #  and then assign the distances as attributes
        for edge in gph.edges.keys():
            gph.edges[edge]["cost"] = np.linalg.norm(
                np.array(gph.nodes[edge[1]]["Coordinates"]) -
                np.array(gph.nodes[edge[0]]["Coordinates"]))

        return gph
