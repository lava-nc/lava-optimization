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
from lava.lib.optimization.apps.vrp.utils.q_matrix_generator import QMatrixVRP


class TestMatrixGen(unittest.TestCase):
    def test_cluster_Q_gen(self) -> None:
        input_nodes = [(0, 1), (2, 3), (2, 1), (1, 1), (2, 4)]
        lamda_vhcles = 1.5
        lamda_wypts = 2.5
        num_vehicles = 2
        # testing floating point Q matrix
        Q_clustering_fltg_pt = QMatrixVRP(
            input_nodes,
            num_vehicles=num_vehicles,
            problem_type="clustering",
            lamda_vhcles=lamda_vhcles,
            lamda_wypts=lamda_wypts,
        ).matrix

        Q_dist_test = distance.cdist(input_nodes, input_nodes, "euclidean")
        vhcle_mat_off_diag = 2 * np.ones((num_vehicles, num_vehicles))
        vhcle_mat_diag = -3 * np.eye(num_vehicles, num_vehicles)
        vhcle_mat = vhcle_mat_off_diag + vhcle_mat_diag
        Q_vhcle_cnst_test = np.pad(
            vhcle_mat,
            (
                (0, Q_dist_test.shape[0] - num_vehicles),
                (0, Q_dist_test.shape[1] - num_vehicles),
            ),
            "constant",
            constant_values=(0),
        )

        wypts_mat_off_diag = 2 * np.ones(Q_dist_test.shape)
        wypts_mat_diag = -3 * np.eye(
            Q_dist_test.shape[0], Q_dist_test.shape[1]
        )
        Q_wypt_cnst_test = wypts_mat_off_diag + wypts_mat_diag

        Q_test = (
            Q_dist_test
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
            problem_type="clustering",
            fixed_pt=True,
        ).matrix
        fixed_pt_Q_max = np.max(np.abs(Q_clustering_fixed_pt))
        fixed_pt_Q_min = np.min(np.abs(Q_clustering_fixed_pt))
        self.assertGreaterEqual(fixed_pt_Q_min == 0, True)
        self.assertLessEqual(fixed_pt_Q_max == 127, True)

    def test_tsp_Q_gen(self) -> None:
        input_nodes = [(0, 1), (2, 3), (2, 1)]
        lamda_dist = 2.5
        lamda_cnstrt = 4
        num_vehicles = 1
        # testing floating point Q matrix
        Q_tsp_fltg_pt = QMatrixVRP(
            input_nodes,
            num_vehicles=num_vehicles,
            problem_type="tsp",
            lamda_dist=lamda_dist,
            lamda_cnstrt=lamda_cnstrt
        ).matrix
        Q_dist_test = distance.cdist(input_nodes, input_nodes, "euclidean")
        Q_dist_blck_test = np.kron(np.eye(3), Q_dist_test)
        Q_cnstrnts_test = 2*np.array([[-1, 1, 1, 1, 0, 0, 1, 0, 0],
                                      [1, -1, 1, 0, 1, 0, 0, 1, 0],
                                      [1, 1, -1, 0, 0, 1, 0, 0, 1],
                                      [1, 0, 0, -1, 1, 1, 1, 0, 0],
                                      [0, 1, 0, 1, -1, 1, 0, 1, 0],
                                      [0, 0, 1, 1, 1, -1, 0, 0, 1],
                                      [1, 0, 0, 1, 0, 0, -1, 1, 1],
                                      [0, 1, 0, 0, 1, 0, 1, -1, 1],
                                      [0, 0, 1, 0, 0, 1, 1, 1, -1]])
        
        Q_test = lamda_dist*Q_dist_blck_test + lamda_cnstrt*Q_cnstrnts_test
        
        self.assertEqual(np.all(Q_tsp_fltg_pt == Q_test), True)

        # testing fixed point Q matrix
        # individual values not really testsed since the rounding
        # is stochastic
        Q_tsp_fixed_pt = QMatrixVRP(
            input_nodes, num_vehicles=2, problem_type="tsp", fixed_pt=True
        ).matrix
        fixed_pt_Q_max = np.max(np.abs(Q_tsp_fixed_pt))
        fixed_pt_Q_min = np.min(np.abs(Q_tsp_fixed_pt))
        self.assertGreaterEqual(fixed_pt_Q_min == 0, True)
        self.assertLessEqual(fixed_pt_Q_max == 127, True)


if __name__ == "__main__":
    unittest.main()
