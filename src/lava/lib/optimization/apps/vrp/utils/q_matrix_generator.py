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

import numpy as np
from scipy.spatial import distance
import typing as ty


class ProblemType(enum.IntEnum):
    RANDOM = 0
    CLUSTER = 1
    TSP = 2


class QMatrixVRP:
    """Class to generate Q matrix for VRP framed as QUBO problems. Currently 
    supports generation of Q matrices for TSP and clustering problems. The 
    matrix values are computed based on the Euclidean distance between the nodes 
    assuming all-to-all connectivity. 
    """

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

    ) -> None:
        """constructor of the class generates Q matrices depending on the type 
        of problem specified by the user and assings it the class variables for 
        the matrix. Calls private functions to initialize Q. Nonetype will raise 
        an exception asking for the correct problem type to be specified. The
        matrix Q is considered to have all-to-all connectivity between the nodes 
        that are specified. 

        Args:
            input_nodes (list<tuples>): Input to matrix generator functions 
            containing a list of nodes specifed as tuples. 

            num_vehicles (int): Number of vehicles in the Vehicle Routing 
            Problem. The first `num_vehicles` nodes in the `input_nodes` 
            variable correspond to positions of vehicles and the rest waypoints.
            Defaults to 1 

            problem_type (str, optional): Specifies for the type of problem for 
            which Q matrix has to be generated. Currently support for 
            1. `tsp` : Travelling salesman problem
            2. `clustering` : Clustering problem framed as a QUBO.
            Defaults to 'None'.
            
            lamda_wypts (float, optional): penalty term for the wypt nodes in
            the Q matrix generator for clustering.

            lamda_vhcles (float, optional): penalty term for the vhcle nodes in
            the Q matrix generator for clustering

            lamda_dist (float, optional): penalty term for the distances between
            nodes in the Q matrix generator.

            lamda_cnstrt (float, optional): penalty term for the constraint
            between nodes in the Q matrix generator for tsp problem. This also
            corresponds to the unweighted part of the tsp problem.

            fixed_pt (bool, optional): Specifies if the Q matrix should
            ultimately be rounded down to integer. If `True`, stochastic
            rounding to integer range of Loihi 2 is performed. Defaults to
            `False`.

            fixed_pt_range (tuple<int>, optional): Specifies the absolute value 
            of  min and max values that the Q matrix can have if 
            `fixed_pt =True`.
        """
        self.fixed_pt = fixed_pt
        self.min_fixed_pt_mant = fixed_pt_range[0]
        self.max_fixed_pt_mant = fixed_pt_range[1]
        self.problem_type = problem_type
        self.num_vehicles = num_vehicles

        if self.problem_type == ProblemType.RANDOM:
            self.matrix = np.random.randint(-128, 127, size=(
                mat_size_for_random, mat_size_for_random))
        elif self.problem_type == ProblemType.CLUSTER:
            self.matrix = self._gen_clustering_Q_matrix(
                input_nodes, lamda_dist, lamda_wypts, lamda_vhcles
            )
        elif self.problem_type == ProblemType.TSP:
            self.matrix = self._gen_tsp_Q_matrix(
                input_nodes, lamda_dist, lamda_cnstrt
            )
        else:
            raise ValueError(
                "problem_type cannot be None or argument passed cannot be "
                "serviced"
            )

    def _gen_tsp_Q_matrix(self, input_nodes, lamda_dist, lamda_cnstrnt):
        """Return the Q matrix that sets up the QUBO for the 
        clustering problem. The cluster centers are assumed to be uniformly 
        distributed across the graph.

        Args:
            input_nodes (list[tuples]): Input to matrix generator functions 
            containing a list of nodes specifed as tuples. All the nodes 
            correspond to waypoints relevant to the tsp problem. The distance
            between nodes is assumed to be symmetric i.e. A->B = B->A


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
            for l in range(num_wypts):
                # Sufficent to traverse only lower triangle
                if k == l:
                    break
                else:
                    q_inter_matrx = np.zeros((num_wypts_sq, num_wypts_sq))
                    u, v = k, l
                    for i in range(num_wypts):
                        for j in range(num_wypts - 1):
                            v_ind_row = (
                                v + (j + 1) * num_wypts
                            ) % num_wypts_sq
                            u_ind_row = (
                                u + (j + 1) * num_wypts
                            ) % num_wypts_sq
                            q_inter_matrx[v_ind_row, u] = 1
                            q_inter_matrx[u_ind_row, v] = 1
                        v = (v + num_wypts) % num_wypts_sq
                        u = (u + num_wypts) % num_wypts_sq
                    Q_dist_mtrx += Dist[k, l] * q_inter_matrx
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
        seq_constraints = (
            -3 * seq_constraint_diag + 2 * seq_constraint_off_diag
        )

        # matrix should contain non-zero elements with only value 2 now.
        Q_cnstrnts_blck_mtrx = vrtx_constraints + seq_constraints

        # Q_cnstrnts
        Q = lamda_dist * Q_dist_mtrx + lamda_cnstrnt * Q_cnstrnts_blck_mtrx

        if self.fixed_pt:
            Q = self._stochastic_rounding(Q)
        return Q

    def _gen_clustering_Q_matrix(
        self, input_nodes, lamda_dist, lamda_wypts, lamda_vhcles
    ):
        """Return the Q matrix that sets up the QUBO for the 
        clustering problem. The cluster centers are assumed to be uniformly 
        distributed across the graph.

        Args:
            input_nodes (list[tuples]): Input to matrix generator functions 
            containing a list of nodes specifed as tuples. First `num_vehicles`
            tuples correspond to the vehicle nodes.

            lamda_dist (float, optional): penalty term for the euclidean
            distance between all nodes for the clustering Q matrix generator.

            lamda_wypts (float, optional): penalty term for the wypt nodes in
            the Q matrix generator for clustering.

            lamda_vhcles (float, optional): penalty term for the vhcle nodes in
            the Q matrix generator for clustering

        Returns:
            np.ndarray: Returns a 2 dimension connectivity matrix of size 
            n*n
        """
        Dist = distance.cdist(input_nodes, input_nodes, "euclidean")
        num_nodes = Dist.shape[0]

        # TODO: Introduce cut-off distancing later to sparsify distance matrix
        # later
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

        qubo_encdng_size = num_nodes * self.num_vehicles
        wypnt_cnstrnt_diag = np.eye(qubo_encdng_size, qubo_encdng_size)

        # construct fundamental column that encodes waypoint constraints
        wypnt_cnstrnt_vector = []
        one_counter = 0
        for i in range(qubo_encdng_size):
            if (i % num_nodes) == 0 and (one_counter != self.num_vehicles):
                wypnt_cnstrnt_vector.append(1)
                one_counter += 1
            else:
                wypnt_cnstrnt_vector.append(0)
        wypnt_cnstrnt_vector = np.array(wypnt_cnstrnt_vector)

        # repeating fundamental unit constructed
        wypnt_cnstrnt_off_diag_unit = wypnt_cnstrnt_vector
        for i in range(self.num_vehicles - 1):
            wypnt_cnstrnt_off_diag_unit = np.vstack(
                (wypnt_cnstrnt_off_diag_unit, np.roll(wypnt_cnstrnt_vector, 1))
            )
        wypnt_cnstrnt_off_diag_unit = wypnt_cnstrnt_off_diag_unit.T
        # zero valued columns part of the repeating fundamental unit
        for i in range(num_nodes - self.num_vehicles):
            wypnt_cnstrnt_off_diag_unit = np.hstack(
                (
                    wypnt_cnstrnt_off_diag_unit,
                    np.zeros((wypnt_cnstrnt_vector.shape[0], 1)),
                )
            )
        wypnt_cnstrnt_off_diag_mtrx = wypnt_cnstrnt_off_diag_unit

        # num vehicles gives num of clusters for which the block needs to be
        # repeated
        for i in range(self.num_vehicles - 1):
            wypnt_cnstrnt_off_diag_mtrx = np.hstack(
                (wypnt_cnstrnt_off_diag_mtrx, wypnt_cnstrnt_off_diag_unit)
            )

        # Diag elements of -3 to subtract from earlier matrix and get -1 in the
        # diagonal later
        wypt_mat_diag = np.diag(np.diag(wypnt_cnstrnt_off_diag_mtrx))

        # Off-diag elements are two, diagonal elements are -1
        cnstrnt_wypts_blck = (
            2 * wypnt_cnstrnt_off_diag_mtrx + -3 * wypt_mat_diag
        )
        # Combine all matrices to get final Q matrix

        Q = (
            lamda_dist * dist_mtrx
            + lamda_vhcles * cnstrnt_vhcles_blck
            + lamda_wypts * cnstrnt_wypts_blck
        )
        if self.fixed_pt:
            Q = self._stochastic_rounding(Q)
        return Q

    def _stochastic_rounding(self, tensor):
        """function to rescale and stochastically round tensor to fixed point 
        values compatiable with Unsigned Mode on Loihi 2.

        Args:
            tensor (np.ndarray): floating-point tensor 

        Returns:
            (np.ndarray): fixed-point version of tensor that is passed as input
        """
        tensor_max = np.max(np.abs(tensor))
        tensor_min = np.min(np.abs(tensor))

        # Get sign mask of tensor to furnish signs for matrix later
        tensor_sign_mask = np.sign(tensor)
        scaled_tensor = (
            (self.max_fixed_pt_mant - self.min_fixed_pt_mant)
            * (np.abs(tensor) - tensor_min)
            / (tensor_max - tensor_min)
        )
        stchstc_rnded_tensor = np.floor(
            scaled_tensor + np.random.rand(tensor.shape[0], tensor.shape[1])
        )
        stchstc_rnded_tensor *= tensor_sign_mask
        return stchstc_rnded_tensor