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
from lava.lib.optimization.apps.clustering.utils.q_matrix_generator import \
    QMatrixClust


class TestMatrixGen(unittest.TestCase):
    def test_Q_gen(self) -> None:
        input_nodes = np.array([(0, 1), (2, 3), (2, 1), (1, 1)])
        lambda_centers = 1.5
        lambda_points = 2.5
        lambda_dist = 2
        num_clusters = 2
        # testing floating point Q matrix
        q_obj = QMatrixClust(
            input_nodes,
            num_clusters=num_clusters,
            lambda_dist=lambda_dist,
            lambda_centers=lambda_centers,
            lambda_points=lambda_points,
        )
        Q_clustering_fltg_pt = q_obj.matrix
        Q_dist_test = distance.cdist(input_nodes,
                                     input_nodes,
                                     "euclidean")
        Q_dist_blck_test = np.kron(np.eye(num_clusters), Q_dist_test)

        Q_wypt_cnst_test = np.array(
            [
                [-1., 0., 0., 0., 2., 0., 0., 0.],
                [0., -1., 0., 0., 0., 2., 0., 0.],
                [0., 0., -1., 0., 0., 0., 2., 0.],
                [0., 0., 0., -1., 0., 0., 0., 2.],
                [2., 0., 0., 0., -1., 0., 0., 0.],
                [0., 2., 0., 0., 0., -1., 0., 0.],
                [0., 0., 2., 0., 0., 0., -1., 0.],
                [0., 0., 0., 2., 0., 0., 0., -1.]
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
            lambda_dist * Q_dist_blck_test
            + lambda_centers * Q_vhcle_cnst_test
            + lambda_points * Q_wypt_cnst_test
        )
        self.assertEqual(np.all(Q_clustering_fltg_pt == Q_test), True)

        # testing fixed point Q matrix
        # individual values not really testsed since the rounding
        # is stochastic
        q_obj_2 = QMatrixClust(
            input_nodes,
            num_clusters=2,
            fixed_pt=True,
        )
        Q_clustering_fixed_pt = q_obj_2.matrix
        fixed_pt_Q_max = np.max(np.abs(Q_clustering_fixed_pt))
        fixed_pt_Q_min = np.min(np.abs(Q_clustering_fixed_pt))
        self.assertGreaterEqual(fixed_pt_Q_min == 0, True)
        self.assertLessEqual(fixed_pt_Q_max == 127, True)


if __name__ == "__main__":
    unittest.main()
