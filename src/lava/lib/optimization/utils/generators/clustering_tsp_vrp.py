# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty
import numpy as np
import numpy.typing as npty


class AbstractProblem:
    def __init__(self,
                 num_anchors: int = 10,
                 num_nodes: int = 100,
                 domain: ty.Union[
                     ty.List[ty.List],
                     ty.List[ty.Tuple],
                     ty.Tuple[ty.List],
                     npty.NDArray] = None):
        """
        Abstract class for randomly sampling anchor points and nodes in a
        given rectangular domain.

        - Clustering problem: the anchor points are cluster centers around
        which nodes are to be clustered
        - Traveling Salesman Problem: the anchor points can be starting
        points for tours to visit the nodes. Typical case would be to have
        only one anchor point.
        - Vehicle Routing Problem: the anchor points are the vehicle
        positions and nodes are the waypoints the vehicles need to visit.

        Parameters
        ----------
        num_anchors (int) : number of anchor points to be generated randomly.
        num_nodes (int) : number of nodes to be sampled from the `domain`.
        domain (int) : domain in which the anchor points and nodes are
        generated. Needs to be a rectangle, specified by lower left (LL) corner
        coordinates and upper right (UR) coordinates.
        """
        self._num_anchors = num_anchors
        self._num_nodes = num_nodes
        if domain is None:
            self._domain = np.array([[0, 0], [100, 100]])
        else:
            self._domain = np.array(domain)

        # The following will be populated in the concrete instances
        self._anchor_coords = None
        self._node_coords = None

    @property
    def num_anchors(self):
        return self._num_anchors

    @num_anchors.setter
    def num_anchors(self, val: int):
        self._num_anchors = val

    @property
    def num_nodes(self):
        return self._num_nodes

    @num_nodes.setter
    def num_nodes(self, val: int):
        self._num_nodes = val

    @property
    def num_pt_per_clust(self):
        return int(np.floor(self.num_nodes / self.num_anchors))

    @property
    def residual_num_per_clust(self):
        return int(self.num_nodes - self.num_anchors * np.floor(
            self.num_nodes / self.num_anchors))

    @property
    def domain(self):
        return self._domain

    @domain.setter
    def domain(self, d: ty.Union[ty.List[ty.List], ty.List[ty.Tuple],
               ty.Tuple[ty.List], npty.NDArray]):
        self._domain = d

    @property
    def domain_ll(self):
        return self.domain[0]

    @property
    def domain_ur(self):
        return self.domain[1]

    @property
    def anchor_coords(self) -> ty.List[ty.Tuple[int, int]]:
        return self._anchor_coords

    @property
    def node_coords(self) -> ty.List[ty.Tuple[int, int]]:
        return self._node_coords


class AbstractClusteringProblem(AbstractProblem):
    def __init__(self, **kwargs):
        num_clusters = kwargs.pop('num_clusters', 10)
        num_points = kwargs.pop('num_points', 100)
        self.num_clusters = num_clusters
        self.num_points = num_points
        kwargs.update({'num_anchors': num_clusters,
                       'num_nodes': num_points})
        super(AbstractClusteringProblem, self).__init__(**kwargs)
        self.center_coords: ty.List[ty.Tuple[int, int]] = self.anchor_coords
        self.point_coords: ty.List[ty.Tuple[int, int]] = self.node_coords


class AbstractTSP(AbstractProblem):
    def __init__(self, **kwargs):
        num_starting_pts = kwargs.pop('num_starting_pts', 1)
        num_dest_nodes = kwargs.pop('num_dest_nodes', 5)
        self.num_starting_pts = num_starting_pts
        self.num_dest_nodes = num_dest_nodes
        kwargs.update({'num_anchors': num_starting_pts,
                       'num_nodes': num_dest_nodes})
        super(AbstractTSP, self).__init__(**kwargs)
        self.starting_coords: ty.List[ty.Tuple[int, int]] = self.anchor_coords
        self.dest_coords: ty.List[ty.Tuple[int, int]] = self.node_coords


class AbstractVRP(AbstractProblem):
    def __init__(self, **kwargs):
        num_vehicles = kwargs.pop('num_vehicles', 10)
        num_waypoints = kwargs.pop('num_waypoints', 100)
        self.num_vehicles = num_vehicles
        self.num_waypoints = num_waypoints
        kwargs.update({'num_anchors': num_vehicles,
                       'num_nodes': num_waypoints})
        super(AbstractVRP, self).__init__(**kwargs)
        self.vehicle_coords = self.anchor_coords
        self.waypoint_coords = self.node_coords


class AbstractUniformProblem(AbstractProblem):
    def __init__(self, **kwargs):
        """Anchor points as well as nodes are uniformly randomly generated
        in the specified domain.
        """
        super(AbstractUniformProblem, self).__init__(**kwargs)
        total_pts_to_sample = self.num_anchors + self.num_nodes
        all_coords = np.random.randint(self.domain_ll, self.domain_ur,
                                       size=(total_pts_to_sample, 2))
        self._anchor_coords = all_coords[:self.num_anchors, :]
        self._node_coords = all_coords[self.num_anchors:, :]


class AbstractGaussianProblem(AbstractProblem):
    def __init__(self, **kwargs):
        """Anchor points are uniformly randomly generated in the specified
        domain. Nodes are generated by sampling from a Gaussian distribution
        around the anchor points.

        Parameters
        ----------
        variance : (int, float, or list of either) variance for Gaussian
        random sampling around each anchor point. If it is a scalar (int
        or float), same value is used for all anchor points. If it is a
        list, the length of the list must be equal to the number of anchor
        points. Each element of the list is used as the variance for random
        sampling around the corresponding anchor points.
        """
        variance = kwargs.pop('variance', 1)
        super(AbstractGaussianProblem, self).__init__(**kwargs)
        self._anchor_coords = np.random.randint(self.domain_ll, self.domain_ur,
                                                size=(self.num_anchors, 2))
        self._node_coords = np.zeros((self.num_nodes, 2), dtype=int)
        if isinstance(variance, (list, np.ndarray)):
            if not len(variance) == self._anchor_coords:
                raise AssertionError("The length of variances does not match "
                                     "the number of anchor points.")
        else:
            variance = [variance] * self.num_anchors
        clust_ids_for_extra = []
        if self.residual_num_per_clust > 0:
            clust_ids_for_extra = np.random.randint(
                0, self.num_anchors, size=(self.residual_num_per_clust,))
        prev_id = 0
        for j in range(self.num_anchors):
            num_to_sample = self.num_pt_per_clust + 1 if (
                j in clust_ids_for_extra) else self.num_pt_per_clust
            next_id = prev_id + num_to_sample
            self._node_coords[prev_id:next_id, :] = (
                np.random.normal(self._anchor_coords[j, :], variance[j],
                                 size=(num_to_sample, 2)).astype(int))
            prev_id = next_id


class UniformlySampledClusteringProblem(AbstractClusteringProblem,
                                        AbstractUniformProblem):
    def __init__(self, **kwargs):
        super(UniformlySampledClusteringProblem, self).__init__(**kwargs)


class GaussianSampledClusteringProblem(AbstractClusteringProblem,
                                       AbstractGaussianProblem):
    def __init__(self, **kwargs):
        super(GaussianSampledClusteringProblem, self).__init__(**kwargs)


class UniformlySampledTSP(AbstractTSP, AbstractUniformProblem):
    def __init__(self, **kwargs):
        super(UniformlySampledTSP, self).__init__(**kwargs)
