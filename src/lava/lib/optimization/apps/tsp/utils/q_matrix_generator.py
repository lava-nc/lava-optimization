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
import time
import numpy as np
from scipy.spatial import distance


class QMatrixTSP:
    """Class to generate Q matrix for travelling salesman problems (TSPs)
    framed as QUBO problems. The matrix values are computed based on the
    Euclidean distance between the nodes assuming all-to-all connectivity."""

    def __init__(
            self,
            input_nodes,
            lamda_dist=1,
            lamda_cnstrt=1,
            fixed_pt=False,
            fixed_pt_range=(0, 127),
            profile_mat_gen=False
    ) -> None:
        """
        Parameters
        ----------
            input_nodes (list of 2-tuples): Input to matrix generator functions
            containing a list of node coordinates specified as 2-tuples.

            lamda_dist (float, optional): Weightage of the pairwise distance
            matrix in the Q matrix.

            lamda_cnstrt (float, optional): Weightage of the hard constraints
            in the Q matrix.

            fixed_pt (bool, optional): Specifies if the Q matrix should be
            rounded down to integer. If `True`, stochastic rounding to
            integer range specified `fixed_pt_range` as is performed.
            Defaults to `False`.

            fixed_pt_range (tuple<int>, optional): Specifies the absolute
            value of  min and max values that the Q matrix can have if
            `fixed_pt =True`.

            profile_mat_gen (bool, optional): Specifies if Q matrix
            generation needs to be timed using python's time.time()
        """

        self.fixed_pt = fixed_pt
        self.min_fixed_pt_mant = fixed_pt_range[0]
        self.max_fixed_pt_mant = fixed_pt_range[1]
        self.time_to_gen_mat = 0.

        start_time = time.time()
        self.matrix = self._gen_Q_matrix(
            input_nodes, lamda_dist, lamda_cnstrt
        )
        if profile_mat_gen:
            self.time_to_gen_mat = time.time() - start_time

    def _gen_Q_matrix(self, input_nodes, lamda_dist, lamda_cnstrnt):
        """Return the Q matrix that sets up the QUBO for the clustering
        problem. The cluster centers are assumed to be uniformly distributed
        across the graph.

        Parameters
        ----------
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

        # number of waypoints for the salesman
        num_wypts = Dist.shape[0]

        # The distance matrix component of the Q matrix
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
