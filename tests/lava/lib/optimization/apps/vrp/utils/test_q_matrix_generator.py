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
import unittest
import numpy as np
from scipy.spatial import distance
from lava.lib.optimization.apps.vrp.utils.q_matrix_generator import \
    QMatrixVRP, ProblemType


class TestMatrixGen(unittest.TestCase):
    def test_cluster_Q_gen(self) -> None:
        input_nodes = [(0, 1), (2, 3), (2, 1), (1, 1)]
        lamda_vhcles = 1.5
        lamda_wypts = 2.5
        lamda_dist = 2
        num_vehicles = 2
        # testing floating point Q matrix
        Q_clustering_fltg_pt = QMatrixVRP(
            input_nodes,
            num_vehicles=num_vehicles,
            problem_type=ProblemType.CLUSTER,
            lamda_dist=lamda_dist,
            lamda_vhcles=lamda_vhcles,
            lamda_wypts=lamda_wypts,
        ).matrix
        Q_dist_test = distance.cdist(input_nodes, input_nodes, "euclidean")
        Q_dist_blck_test = np.kron(np.eye(num_vehicles), Q_dist_test)

        Q_wypt_cnst_test = np.array(
            [
                [-1.,  0.,  0.,  0.,  2.,  0.,  0.,  0.],
                [0., -1.,  0.,  0.,  0.,  2.,  0.,  0.],
                [0.,  0., -1.,  0.,  0.,  0.,  2.,  0.],
                [0.,  0.,  0., -1.,  0.,  0.,  0.,  2.],
                [2.,  0.,  0.,  0., -1.,  0.,  0.,  0.],
                [0.,  2.,  0.,  0.,  0., -1.,  0.,  0.],
                [0.,  0.,  2.,  0.,  0.,  0., -1.,  0.],
                [0.,  0.,  0.,  2.,  0.,  0.,  0., -1.]
            ]
        )

        Q_vhcle_cnst_test = np.array(
            [
                [-1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -1.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 2.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        Q_test = (
            lamda_dist * Q_dist_blck_test
            + lamda_vhcles * Q_vhcle_cnst_test
            + lamda_wypts * Q_wypt_cnst_test
        )
        self.assertEqual(np.all(Q_clustering_fltg_pt == Q_test), True)

        # testing fixed point Q matrix
        # individual values not really testsed since the rounding
        # is stochastic
        Q_clustering_fixed_pt = QMatrixVRP(
            input_nodes,
            num_vehicles=2,
            problem_type=ProblemType.CLUSTER,
            fixed_pt=True,
        ).matrix
        fixed_pt_Q_max = np.max(np.abs(Q_clustering_fixed_pt))
        fixed_pt_Q_min = np.min(np.abs(Q_clustering_fixed_pt))
        self.assertGreaterEqual(fixed_pt_Q_min == 0, True)
        self.assertLessEqual(fixed_pt_Q_max == 127, True)

    def test_tsp_Q_gen(self) -> None:
        input_nodes = [(0, 1), (0, 0), (0, -1)]
        lamda_dist = 2.5
        lamda_cnstrt = 4  # testing floating point Q matrix
        Q_tsp_fltg_pt = QMatrixVRP(
            input_nodes,
            problem_type=ProblemType.TSP,
            lamda_dist=lamda_dist,
            lamda_cnstrt=lamda_cnstrt,
        ).matrix
        # Q_dist_test = distance.cdist(input_nodes, input_nodes, "euclidean")
        Q_dist_scaled_test = np.array(
            [
                [0, 0, 0, 0, 1, 2, 0, 1, 2],
                [0, 0, 0, 1, 0, 1, 1, 0, 1],
                [0, 0, 0, 2, 1, 0, 2, 1, 0],
                [0, 1, 2, 0, 0, 0, 0, 1, 2],
                [1, 0, 1, 0, 0, 0, 1, 0, 1],
                [2, 1, 0, 0, 0, 0, 2, 1, 0],
                [0, 1, 2, 0, 1, 2, 0, 0, 0],
                [1, 0, 1, 1, 0, 1, 0, 0, 0],
                [2, 1, 0, 2, 1, 0, 0, 0, 0],
            ]
        )
        Q_cnstrnts_test = 2 * np.array(
            [
                [-1, 1, 1, 1, 0, 0, 1, 0, 0],
                [1, -1, 1, 0, 1, 0, 0, 1, 0],
                [1, 1, -1, 0, 0, 1, 0, 0, 1],
                [1, 0, 0, -1, 1, 1, 1, 0, 0],
                [0, 1, 0, 1, -1, 1, 0, 1, 0],
                [0, 0, 1, 1, 1, -1, 0, 0, 1],
                [1, 0, 0, 1, 0, 0, -1, 1, 1],
                [0, 1, 0, 0, 1, 0, 1, -1, 1],
                [0, 0, 1, 0, 0, 1, 1, 1, -1],
            ]
        )

        Q_test = (
            lamda_dist * Q_dist_scaled_test + lamda_cnstrt * Q_cnstrnts_test
        )
        self.assertEqual(np.all(Q_tsp_fltg_pt == Q_test), True)
        # testing fixed point Q matrix
        # individual values not really testsed since the rounding
        # is stochastic
        Q_tsp_fixed_pt = QMatrixVRP(
            input_nodes, num_vehicles=2, problem_type=ProblemType.TSP,
            fixed_pt=True
        ).matrix
        fixed_pt_Q_max = np.max(np.abs(Q_tsp_fixed_pt))
        fixed_pt_Q_min = np.min(np.abs(Q_tsp_fixed_pt))
        self.assertGreaterEqual(fixed_pt_Q_min == 0, True)
        self.assertLessEqual(fixed_pt_Q_max == 127, True)


if __name__ == "__main__":
    unittest.main()