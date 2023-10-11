# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import time
import copy

import numpy as np
import numpy.typing as npty
from scipy.spatial import distance


class QMatrixClust:
    """Class to generate Q matrix for a clustering problem framed as a QUBO
    problem. The matrix values are computed based on the Euclidean distance
    between the nodes assuming all-to-all connectivity."""

    def __init__(
            self,
            input_nodes,
            num_clusters=1,
            lambda_dist=1,
            lambda_points=100,
            lambda_centers=100,
            fixed_pt=False,
            fixed_pt_range=(0, 127),
            clust_dist_sparse_params=None,
            profile_mat_gen=False
    ) -> None:
        """The Constructor of the class generates a Q matrix for clustering
        and assigns it the class variables for the matrix. Calls private
        functions to initialize Q. The matrix Q is considered to have all-to-all
        connectivity between the nodes that are specified.

        Args:
            input_nodes (list<tuples>): Input to matrix generator functions
            containing a list of nodes specified as tuples.

            num_clusters (int): Number of clusters to be formed after
            clustering is done. The first `num_clusters` nodes in
            `input_nodes` correspond to positions of cluster centers. Defaults
            to 1.

            lambda_dist (float, optional): relative weight of the
            pairwise distance term in the QUBO energy function. Default is 1.

            lambda_points (float, optional): relative weight (in the QUBO
            energy function) of the constraint that each point should belong
            to exactly one cluster. Higher values signify "hardness" of the
            constraint. Default is 100.

            lambda_centers (float, optional): relative weight (in the QUBO
            energy function) of the constraint that each cluster center
            should belong to exactly one cluster. Higher values signify
            "hardness of the constraint". Default is 100.

            fixed_pt (bool, optional): Specifies if the Q matrix should
            ultimately be rounded down to integer. If `True`, stochastic
            rounding to integer range of Loihi 2 is performed. Defaults to
            `False`.

            fixed_pt_range (tuple<int>, optional): Specifies the absolute
            value of  min and max values that the Q matrix can have if
            `fixed_pt =True`.

            clust_dist_sparse_params (dict, optional) : Dictionary of
            parameters for sparsification of the distance matrix used in
            clustering of the waypoint and the vehicle positions. The
            parameters are:
                - do_sparse (bool) : a toggle to enable/disable sparsification
                (default is False, i.e., disable sparsification)
                - algo (string) : the algorithm used for sparsification (
                default is "cutoff", which imposes a maximum cutoff distance
                on the distance matrix and subtracts it from the matrix.
                Another option is "edge-prune", which prunes edges longer
                than a cutoff from the connectivity graph of the entire problem
                - max_dist_cutoff_fraction (float) : a fraction between 0 and 1,
                which multiplies the max of the distance matrix, and the
                result is used as the cutoff in both algorithms.

            profile_mat_gen (bool, optional): Specifies if Q matrix
            generation needs to be timed using python's time.time()
        """
        if not clust_dist_sparse_params:
            self.clust_dist_sparse_params = {"do_sparse": False,
                                             "algo": "cutoff",
                                             "max_dist_cutoff_fraction": 1.0}
        else:
            self.clust_dist_sparse_params = \
                copy.deepcopy(clust_dist_sparse_params)
        self.fixed_pt = fixed_pt
        self.min_fixed_pt_mant = fixed_pt_range[0]
        self.max_fixed_pt_mant = fixed_pt_range[1]
        self.num_clusters = num_clusters
        self.max_cutoff_frac = self.clust_dist_sparse_params[
            "max_dist_cutoff_fraction"]
        self.dist_sparsity = 0.
        self.dist_proxy_sparsity = 0.
        self.time_to_gen_mat = 0.

        start_time = time.time()
        self.matrix, self.dist_sparsity, self.dist_proxy_sparsity = \
            self._gen_Q_matrix(input_nodes, lambda_dist, lambda_points,
                               lambda_centers)
        if profile_mat_gen:
            self.time_to_gen_mat = time.time() - start_time

    @staticmethod
    def _compute_matrix_sparsity(mat: npty.NDArray):
        return 1 - (np.count_nonzero(mat) / np.prod(mat.shape))

    def _sparsify_dist_using_cutoff(self, dist):
        # The following variants can be used as a proxy for Euclidean
        # distance, which may help in sparsifying the Q matrix.
        # Dist_proxy = np.zeros_like(Dist)
        # inv_dist_mtrx = (1 / Dist)
        # log_dist_mtrx = np.log(Dist)
        # Dist_proxy[Dist <= 1] = Dist[Dist <= 1]
        # Dist_proxy[Dist > 1] = 2 - log_dist_mtrx[Dist > 1]
        # Dist_proxy = 100 * (1 - np.exp(-Dist/100))
        if self.max_cutoff_frac == 1.0:
            return dist
        max_dist_cutoff = self.max_cutoff_frac * np.max(dist)
        dist_proxy = dist.copy()
        dist_proxy[dist_proxy >= max_dist_cutoff] = max_dist_cutoff
        dist_proxy = np.around(dist_proxy - max_dist_cutoff, 2)
        return dist_proxy

    def _sparsify_dist_using_edge_pruning(self, dist):
        if self.max_cutoff_frac == 1.0:
            return dist
        dist_proxy = dist.copy()
        num_nodes = dist.shape[0]
        max_per_row = np.max(dist_proxy, axis=1)
        max_per_row = max_per_row.reshape((num_nodes, 1))
        max_per_row = np.tile(max_per_row, (1, num_nodes))
        cut_off = self.max_cutoff_frac * max_per_row

        idxmat = dist_proxy >= cut_off
        dist_proxy[idxmat] = cut_off[idxmat]
        dist_proxy = dist_proxy - cut_off

        # Zero-out distances between vehicles. Constraints will later take
        # care of this
        dist_proxy[0:self.num_clusters, 0:self.num_clusters] = np.zeros((
            self.num_clusters, self.num_clusters))
        return dist_proxy

    def _sparsify_dist(self, dist):
        if self.clust_dist_sparse_params["algo"] == "cutoff":
            return self._sparsify_dist_using_cutoff(dist)
        elif self.clust_dist_sparse_params["algo"] == "edge_prune":
            return self._sparsify_dist_using_edge_pruning(dist)
        else:
            raise ValueError("Invalid algorithm chosen for sparsification of "
                             "the distance matrix in Q-matrix computation for "
                             "the clustering stage. Choose one of 'cutoff' "
                             "and 'edge_prune'.")

    def _gen_Q_matrix(
            self, input_nodes, lambda_dist, lambda_points, lambda_centers
    ):
        """Return the Q matrix that sets up the QUBO for a clustering
        problem.

        Args:
            input_nodes (list[tuples]): Input to matrix generator functions
            containing a list of nodes specified as tuples. First
            `num_vehicles` tuples correspond to the vehicle nodes.

            lambda_dist (float, optional): relative weight of the
            pairwise distance term in the QUBO energy function.

            lambda_points (float, optional): relative weight (in the QUBO
            energy function) of the constraint that each point should belong
            to exactly one cluster. Higher values signify "hardness" of the
            constraint.

            lambda_centers (float, optional): relative weight (in the QUBO
            energy function) of the constraint that each cluster center
            should belong to exactly one cluster. Higher values signify
            "hardness of the constraint".

        Returns:
            np.ndarray: Returns a 2 dimension connectivity matrix of size n*n
        """
        Dist = distance.cdist(input_nodes, input_nodes, "euclidean")
        # Normalize the distance matrix
        dist_sparsity = self._compute_matrix_sparsity(Dist)
        dist_proxy_sparsity = dist_sparsity
        num_nodes = Dist.shape[0]
        if self.clust_dist_sparse_params["do_sparse"]:
            Dist = self._sparsify_dist(Dist)
            dist_proxy_sparsity = self._compute_matrix_sparsity(Dist)

        # TODO: Introduce cut-off distancing later to sparsify distance
        #  matrix later using one of the proxies above.
        # Distance matrix for the encoding
        dist_mtrx = np.kron(np.eye(self.num_clusters), Dist)
        # Vehicles can only belong to one cluster
        # Off-diagonal elements are two, populating matrix with 2
        centers_mat_off_diag = 2 * np.ones(
            (self.num_clusters, self.num_clusters)
        )

        # Diag elements of -3 to subtract from earlier matrix and get -1 in the
        # diagonal later
        centers_mat_diag = -3 * np.eye(self.num_clusters, self.num_clusters)

        # Off-diag elements are two, diagonal elements are -1
        centers_mat = centers_mat_off_diag + centers_mat_diag

        # Only vehicle waypoints get affected by this constraint
        cnstrnt_centers = np.pad(centers_mat, (
            (0, num_nodes - self.num_clusters),
            (0, num_nodes - self.num_clusters),
        ), "constant", constant_values=(0),)
        cnstrnt_centers_blck = np.kron(
            np.eye(self.num_clusters), cnstrnt_centers
        )

        # waypoints can only belong to one cluster
        # Off-diagonal elements are two, populating matrix with 2
        # Encoding waypoint constraints here
        qubo_encdng_size = Dist.shape[0] * self.num_clusters

        point_cnstrnt_diag = np.eye(qubo_encdng_size, qubo_encdng_size)

        point_cnstrnt_vector = np.array([1 if i % num_nodes == 0 else 0 for i
                                        in range(qubo_encdng_size)])
        point_constraint_off_diag = point_cnstrnt_vector
        for i in range(qubo_encdng_size - 1):
            point_constraint_off_diag = np.vstack(
                (point_constraint_off_diag, np.roll(point_cnstrnt_vector,
                                                    i + 1),))
        cnstrnt_points_blck = \
            -3 * point_cnstrnt_diag + 2 * point_constraint_off_diag
        # Combine all matrices to get final Q matrix
        Q = (lambda_dist * dist_mtrx
             + lambda_centers * cnstrnt_centers_blck
             + lambda_points * cnstrnt_points_blck)
        if self.fixed_pt:
            Q = self._stochastic_rounding(Q)
        return Q, dist_sparsity, dist_proxy_sparsity

    def _stochastic_rounding(self, tensor):
        """A function to rescale and stochastically round tensor to fixed
        point values compatible with Unsigned Mode on Loihi 2.

        Args:
            tensor (np.ndarray): floating-point tensor

        Returns:
            (np.ndarray): fixed-point version of tensor that is passed as input
        """
        tensor_max = np.max(np.abs(tensor))
        tensor_min = np.min(np.abs(tensor))

        # Get sign mask of tensor to furnish signs for matrix later
        tensor_sign_mask = np.sign(tensor)
        scaled_tensor = \
            ((self.max_fixed_pt_mant - self.min_fixed_pt_mant)
                * (np.abs(tensor) - tensor_min)
                / (tensor_max - tensor_min))
        stchstc_rnded_tensor = \
            np.floor(scaled_tensor
                     + np.random.rand(tensor.shape[0], tensor.shape[1]))
        stchstc_rnded_tensor *= tensor_sign_mask
        return stchstc_rnded_tensor
