# INTEL CORPORATION CONFIDENTIAL AND PROPRIETARY
#
# Copyright © 2023 Intel Corporation.
#
# This software and the related documents are Intel copyrighted
# materials, and your use of them is governed by the express
# license under which they were provided to you (License). Unless
# the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit  this software or the
# related documents without Intel's prior written permission.
#
# This software and the related documents are provided as is, with
# no express or implied warranties, other than those that are
# expressly stated in the License.
import enum
import logging
import time
import copy
from pprint import pprint

import numpy as np
import numpy.typing as npty
from scipy.spatial import distance


class ProblemType(enum.IntEnum):
    RANDOM = 0
    CLUSTER = 1
    TSP = 2


class QMatrixVRP:
    """Class to generate Q matrix for VRP framed as QUBO problems. Currently,
    supports generation of Q matrices for TSP and clustering problems. The
    matrix values are computed based on the Euclidean distance between the
    nodes assuming all-to-all connectivity."""

    def __init__(
            self,
            input_nodes,
            num_vehicles=1,
            problem_type=ProblemType.RANDOM,
            mat_size_for_random=1,
            lamda_wypts=1,
            lamda_vhcles=1,
            lamda_dist=1,
            lamda_cnstrt=1,
            fixed_pt=False,
            fixed_pt_range=(0, 127),
            clust_dist_sparse_params=None,
            profile_mat_gen=False
    ) -> None:
        """The Constructor of the class generates Q matrices depending on the
        type of problem specified by the user and assigns it the class
        variables for the matrix. Calls private functions to initialize Q.
        None-type will raise an exception asking for the correct problem
        type to be specified. The matrix Q is considered to have all-to-all
        connectivity between the nodes that are specified.

        Args:
            input_nodes (list<tuples>): Input to matrix generator functions
            containing a list of nodes specified as tuples.

            num_vehicles (int): Number of vehicles in the Vehicle Routing
            Problem. The first `num_vehicles` nodes in the `input_nodes`
            variable correspond to positions of vehicles and the rest
            waypoints. Defaults to 1

            problem_type (str, optional): Specifies for the type of problem
            for which Q matrix has to be generated. Currently support for
            1. `tsp` : Travelling salesman problem
            2. `clustering` : Clustering problem framed as a QUBO. Defaults
            to 'None'.

            lamda_wypts (float, optional): penalty term for the wypt nodes
            in the Q matrix generator for clustering.

            lamda_vhcles (float, optional): penalty term for the vhcle nodes
            in the Q matrix generator for clustering

            lamda_dist (float, optional): penalty term for the distances
            between nodes in the Q matrix generator.

            lamda_cnstrt (float, optional): penalty term for the constraint
            between nodes in the Q matrix generator for tsp problem. This
            also corresponds to the unweighted part of the tsp problem.

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
        self.problem_type = problem_type
        self.num_vehicles = num_vehicles
        self.max_cutoff_frac = self.clust_dist_sparse_params[
            "max_dist_cutoff_fraction"]
        self.dist_sparsity = 0.
        self.dist_proxy_sparsity = 0.
        self.time_clust_mat = 0.
        self.time_tsp_mat = 0.

        if self.problem_type == ProblemType.RANDOM:
            self.matrix = np.random.randint(-128, 127, size=(
                mat_size_for_random, mat_size_for_random))
        elif self.problem_type == ProblemType.CLUSTER:
            start_time = time.time()
            self.matrix, self.dist_sparsity, self.dist_proxy_sparsity = \
                self._gen_clustering_Q_matrix(input_nodes, lamda_dist,
                                              lamda_wypts, lamda_vhcles)
            if profile_mat_gen:
                self.time_clust_mat = time.time() - start_time
                # pprint(f"Clustering Q matrix generated in"
                #        f" {self.time_clust_mat} s")
        elif self.problem_type == ProblemType.TSP:
            start_time = time.time()
            self.matrix = self._gen_tsp_Q_matrix(
                input_nodes, lamda_dist, lamda_cnstrt
            )
            if profile_mat_gen:
                self.time_tsp_mat = time.time() - start_time
                # pprint(f"TSP Q matrix generated in {self.time_tsp_mat} s")
        else:
            raise ValueError(
                "problem_type cannot be None or argument passed cannot be "
                "serviced"
            )

    def _gen_tsp_Q_matrix(self, input_nodes, lamda_dist, lamda_cnstrnt):
        """Return the Q matrix that sets up the QUBO for the clustering
        problem. The cluster centers are assumed to be uniformly distributed
        across the graph.

        Args:
            input_nodes (list[tuples]): Input to matrix generator functions
            containing a list of nodes specified as tuples. All the nodes
            correspond to waypoints relevant to the tsp problem. The
            distance between nodes is assumed to be symmetric i.e. A->B =
            B->A

        Returns:
            np.ndarray: Returns a 2 dimension connectivity matrix of size
            n^2 * n^2
        """
        # Euclidean distances between all nodes input to the graph
        Dist = distance.cdist(input_nodes, input_nodes, "euclidean")

        # number  of waypoints
        num_wypts = Dist.shape[0]

        # TSP distance encoding
        # Distance encoding based on manual calculations that scale to problems
        # of any size
        num_wypts_sq = num_wypts ** 2
        Q_dist_mtrx = np.zeros((num_wypts_sq, num_wypts_sq))
        for k in range(num_wypts):
            for m in range(num_wypts):
                # Sufficent to traverse only lower triangle
                if k == m:
                    break
                else:
                    q_inter_matrx = np.zeros((num_wypts_sq, num_wypts_sq))
                    u, v = k, m
                    for i in range(num_wypts):
                        for j in range(num_wypts - 1):
                            v_ind_row = \
                                (v + (j + 1) * num_wypts) % num_wypts_sq
                            u_ind_row = \
                                (u + (j + 1) * num_wypts) % num_wypts_sq
                            q_inter_matrx[v_ind_row, u] = 1
                            q_inter_matrx[u_ind_row, v] = 1
                        v = (v + num_wypts) % num_wypts_sq
                        u = (u + num_wypts) % num_wypts_sq
                    Q_dist_mtrx += Dist[k, m] * q_inter_matrx
        # TSP constraint encoding
        # Only one vrtx can be selected at a time instant
        # Off-diagonal elements are two, populating matrix with 2
        vrtx_mat_off_diag = 2 * np.ones((num_wypts, num_wypts))

        # Diag elements of -3 to subtract from earlier matrix and get -1 in the
        # diagonal later
        vrtx_mat_diag = -3 * np.eye(num_wypts, num_wypts)

        # Off-diag elements are two, diagonal elements are -1
        vrtx_mat = vrtx_mat_off_diag + vrtx_mat_diag
        vrtx_constraints = np.kron(np.eye(num_wypts), vrtx_mat)

        # Encoding sequence constraints here
        seq_constraint_diag = np.eye(
            num_wypts * num_wypts, num_wypts * num_wypts
        )
        seq_constraint_vector = np.array(
            [
                1 if i % num_wypts == 0 else 0
                for i in range(num_wypts * num_wypts)
            ]
        )
        seq_constraint_off_diag = seq_constraint_vector
        for i in range(num_wypts * num_wypts - 1):
            seq_constraint_off_diag = np.vstack(
                (
                    seq_constraint_off_diag,
                    np.roll(seq_constraint_vector, i + 1),
                )
            )

        # Off-diag elements are two, diagonal elements are -1
        seq_constraints = \
            (-3 * seq_constraint_diag + 2 * seq_constraint_off_diag)

        # matrix should contain non-zero elements with only value 2 now.
        Q_cnstrnts_blck_mtrx = vrtx_constraints + seq_constraints

        # Q_cnstrnts
        Q = lamda_dist * Q_dist_mtrx + lamda_cnstrnt * Q_cnstrnts_blck_mtrx

        if self.fixed_pt:
            Q = self._stochastic_rounding(Q)
        return Q

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
        max_dist_cutoff = self.max_cutoff_frac * np.max(dist)
        dist_proxy = dist.copy()
        dist_proxy[dist_proxy >= max_dist_cutoff] = max_dist_cutoff
        dist_proxy = np.around(dist_proxy - max_dist_cutoff, 2)
        return dist_proxy

    def _sparsify_dist_using_edge_pruning(self, dist):
        dist_proxy = dist.copy()
        return dist_proxy

    def _sparsify_dist(self, dist):
        if self.clust_dist_sparse_params["algo"] == "cutoff":
            return  self._sparsify_dist_using_cutoff(dist)
        elif self.clust_dist_sparse_params["algo"] == "edge_prune":
            return self._sparsify_dist_using_edge_pruning(dist)
        else:
            raise ValueError("Invalid algorithm chosen for sparsification of "
                             "the distance matrix in Q-matrix computation for "
                             "the clustering stage. Choose one of 'cutoff' "
                             "and 'edge_prune'.")

    def _gen_clustering_Q_matrix(
            self, input_nodes, lamda_dist, lamda_wypts, lamda_vhcles
    ):
        """Return the Q matrix that sets up the QUBO for the clustering
        problem. The cluster centers are assumed to be uniformly distributed
        across the graph.

        Args:
            input_nodes (list[tuples]): Input to matrix generator functions
            containing a list of nodes specified as tuples. First
            `num_vehicles` tuples correspond to the vehicle nodes.

            lamda_dist (float, optional): penalty term for the Euclidean
            distance between all nodes for the clustering Q matrix generator.

            lamda_wypts (float, optional): penalty term for the wypt nodes
            in the Q matrix generator for clustering.

            lamda_vhcles (float, optional): penalty term for the vhcle nodes in
            the Q matrix generator for clustering

        Returns:
            np.ndarray: Returns a 2 dimension connectivity matrix of size n*n
        """
        Dist = distance.cdist(input_nodes, input_nodes, "euclidean")
        dist_sparsity = self._compute_matrix_sparsity(Dist)
        dist_proxy_sparsity = dist_sparsity
        num_nodes = Dist.shape[0]
        if self.clust_dist_sparse_params["do_sparse"]:
            Dist = self._sparsify_dist(Dist)
            dist_proxy_sparsity = self._compute_matrix_sparsity(Dist)

        # TODO: Introduce cut-off distancing later to sparsify distance
        #  matrix later using one of the proxies above.
        # Distance matrix for the encoding
        dist_mtrx = np.kron(np.eye(self.num_vehicles), Dist)
        # Vehicles can only belong to one cluster
        # Off-diagonal elements are two, populating matrix with 2
        vhcle_mat_off_diag = 2 * np.ones(
            (self.num_vehicles, self.num_vehicles)
        )

        # Diag elements of -3 to subtract from earlier matrix and get -1 in the
        # diagonal later
        vhcle_mat_diag = -3 * np.eye(self.num_vehicles, self.num_vehicles)

        # Off-diag elements are two, diagonal elements are -1
        vhcle_mat = vhcle_mat_off_diag + vhcle_mat_diag

        # Only vehicle waypoints get affected by this constraint
        cnstrnt_vhcles = np.pad(
            vhcle_mat,
            (
                (0, num_nodes - self.num_vehicles),
                (0, num_nodes - self.num_vehicles),
            ),
            "constant",
            constant_values=(0),
        )
        cnstrnt_vhcles_blck = np.kron(
            np.eye(self.num_vehicles), cnstrnt_vhcles
        )

        # waypoints can only belong to one cluster
        # Off-diagonal elements are two, populating matrix with 2
        # Encoding waypoint constraints here
        qubo_encdng_size = Dist.shape[0] * self.num_vehicles

        wypnt_cnstrnt_diag = np.eye(qubo_encdng_size, qubo_encdng_size)

        wypt_cnstrnt_vector = np.array([1 if i % num_nodes == 0 else 0 for i
                                        in range(qubo_encdng_size)])
        wypt_constraint_off_diag = wypt_cnstrnt_vector
        for i in range(qubo_encdng_size - 1):
            wypt_constraint_off_diag = np.vstack(
                (wypt_constraint_off_diag, np.roll(wypt_cnstrnt_vector,
                                                   i + 1),))
        cnstrnt_wypts_blck = \
            -3 * wypnt_cnstrnt_diag + 2 * wypt_constraint_off_diag
        # Combine all matrices to get final Q matrix
        Q = (lamda_dist * dist_mtrx
             + lamda_vhcles * cnstrnt_vhcles_blck
             + lamda_wypts * cnstrnt_wypts_blck)
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
