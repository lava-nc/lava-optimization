# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import networkx as ntx
import numpy as np
import typing as ty


class ClusteringProblem:
    """Problem specification for a clustering problem.

    N points need to be clustered into M clusters.

    The cluster centers are *given*. Clustering is done to assign cluster IDs
    to points based on the closest cluster centers.
    """
    def __init__(self,
                 point_coords: ty.List[ty.Tuple[int, int]],
                 center_coords: ty.Union[int, ty.List[ty.Tuple[int, int]]],
                 edges: ty.Optional[ty.List[ty.Tuple[int, int]]] = None):
        """
        Parameters
        ----------
        point_coords : list(tuple(int, int))
            A list of integer tuples corresponding to the coordinates of
            points to be clustered.
        center_coords : list(tuple(int, int))
            A list of integer tuples corresponding to the coordinates of
            cluster-centers.
        edges : (Optional) list(tuple(int, int, float))
            An optional list of edges connecting points and cluster centers,
            given as a list of triples (ID1, ID2, weight). See the note
            below for ID-scheme. If None, assume all-to-all connectivity
            between points, weighted by their pairwise distances.

        Notes
        -----
        IDs 1 to M correspond to cluster centers and (M+1) to (M+N) correspond
        to the points to be clustered.
        """
        super().__init__()
        self._point_coords = point_coords
        self._center_coords = center_coords
        self._num_points = len(self._point_coords)
        self._num_clusters = len(self._center_coords)
        self._cluster_ids = list(np.arange(1, self._num_clusters + 1))
        self._point_ids = list(np.arange(
            self._num_clusters + 1, self._num_clusters + self._num_points + 1))
        self._points = dict(zip(self._point_ids, self._point_coords))
        self._cluster_centers = dict(zip(self._cluster_ids,
                                         self._center_coords))
        if edges:
            self._edges = edges
        else:
            self._edges = []

        self._problem_graph = None

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, points: ty.Dict[int, ty.Tuple[int, int]]):
        self._points = points

    @property
    def point_ids(self):
        return self._point_ids

    @property
    def point_coords(self):
        return self._point_coords

    @property
    def num_points(self):
        return self._num_points

    @property
    def edges(self):
        return self._edges

    @property
    def cluster_centers(self):
        return self._cluster_centers

    @cluster_centers.setter
    def cluster_centers(self, cluster_centers: ty.Dict[int, ty.Tuple[int,
                        int]]):
        self._cluster_centers = cluster_centers

    @property
    def cluster_ids(self):
        return self._cluster_ids

    @property
    def center_coords(self):
        return self._center_coords

    @property
    def num_clusters(self):
        return self._num_clusters

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
            gph.add_nodes_from(self.point_ids)
            # If there are user-provided edges, add them between the nodes
            gph.add_edges_from(self.edges)
        else:
            gph = ntx.complete_graph(self.point_ids, create_using=ntx.DiGraph())

        node_type_dict = dict(zip(self.point_ids,
                                  ["Point"] * len(self.point_ids)))
        # Associate node type as "Node" and node coordinates as attributes
        ntx.set_node_attributes(gph, node_type_dict, name="Type")
        ntx.set_node_attributes(gph, self.points, name="Coordinates")

        # Add vehicles as nodes
        gph.add_nodes_from(self.cluster_ids)
        # Associate node type as "Vehicle" and vehicle coordinates as attributes
        cluster_center_type_dict = dict(zip(self.cluster_ids,
                                            ["Cluster Center"] * len(
                                                self.cluster_ids)))
        ntx.set_node_attributes(gph, cluster_center_type_dict, name="Type")
        ntx.set_node_attributes(gph, self.cluster_centers, name="Coordinates")

        # Add edges from initial vehicle positions to all nodes (oneway edges)
        for cid in self.cluster_ids:
            for pid in self.points:
                gph.add_edge(cid, pid)

        # Compute Euclidean distance along all edges and assign them as edge
        # weights
        # ToDo: Replace the loop with independent distance matrix computation
        #  and then assign the distances as attributes
        for edge in gph.edges.keys():
            gph.edges[edge]["cost"] = np.linalg.norm(
                np.array(gph.nodes[edge[1]]["Coordinates"]) - np.array(
                    gph.nodes[edge[0]]["Coordinates"]))

        self._problem_graph = gph
