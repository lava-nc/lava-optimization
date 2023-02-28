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
                 nodes: ty.Dict[int, ty.Tuple[int, int]],
                 vehicles: ty.Union[int, ty.Dict[int, ty.Tuple[int, int]]],
                 edges: ty.Optional[ty.List[ty.Tuple[int, int]]] = None):
        """
        Parameters
        ----------
        nodes : dict(int, tuple(int, int))
            A dictionary mapping node IDs to node coordinates. Nodes signify
            the "customers" in a VRP, which need to be visited by the vehicles.
        vehicles : dict(int, tuple(int, int))
            A dictionary mapping the vehicle IDs to their initial
            coordinates. If all coordinate tuples are equal, then it is
            assumed that all vehicles begin from the same depot.
        edges: (Optional) list(tuple(int, int))
            An optional list of edges connecting nodes, given as a list of
            node ID pairs. If None provided, assume all-to-all connectivity
            between nodes.
        """
        super().__init__()
        self._nodes = nodes
        self._edges = edges
        self._vehicles = vehicles
        self._problem_graph = self._generate_problem_graph()

    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, nodes: ty.Dict[int, ty.Tuple[int, int]]):
        self._nodes = nodes

    @property
    def vehicles(self):
        return self._vehicles

    @vehicles.setter
    def vehicles(self, vehicles: ty.Dict[int, ty.Tuple[int, int]]):
        self._vehicles = vehicles

    @property
    def node_ids(self):
        return list(self._nodes.keys())

    @property
    def node_coords(self):
        return list(self._nodes.values())

    @property
    def vehicle_ids(self):
        return list(self._vehicles.keys())

    @property
    def vehicle_init_coords(self):
        return list(self._vehicles.values())

    @property
    def problem_graph(self):
        return self._problem_graph

    def _generate_problem_graph(self):
        if self.edges:
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
                                     ["Vehicles"] * len(self.vehicle_ids)))
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
            gph.edges[edge]["weight"] = np.linalg.norm(
                gph.nodes[edge[1]]["Coordinates"] -
                gph.nodes[edge[0]]["Coordinates"])

        return gph