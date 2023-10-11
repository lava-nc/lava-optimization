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
from lava.lib.optimization.apps.tsp.utils.q_matrix_generator import \
    QMatrixTSP


class TestMatrixGen(unittest.TestCase):
    def test_Q_gen(self) -> None:
        input_nodes = [(0, 1), (0, 0), (0, -1)]
        lamda_dist = 2.5
        lamda_cnstrt = 4  # testing floating point Q matrix
        q_mat_obj_flp = QMatrixTSP(
            input_nodes,
            lamda_dist=lamda_dist,
            lamda_cnstrt=lamda_cnstrt,
        )
        q_mat_flp = q_mat_obj_flp.matrix
        q_dist_scaled_test = np.array(
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
        q_cnstrnts_test = 2 * np.array(
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

        q_test = (
            lamda_dist * q_dist_scaled_test + lamda_cnstrt * q_cnstrnts_test
        )
        self.assertTrue(np.all(q_mat_flp == q_test))
        # testing fixed point Q matrix
        # individual values not really testsed since the rounding
        # is stochastic
        q_mat_obj_fxp = QMatrixTSP(input_nodes,
                                   fixed_pt=True)
        q_mat_fxp = q_mat_obj_fxp.matrix
        q_max_fxp = np.max(np.abs(q_mat_fxp))
        q_min_fxp = np.min(np.abs(q_mat_fxp))
        self.assertGreaterEqual(q_min_fxp, 0, True)
        self.assertLessEqual(q_max_fxp, 127, True)


if __name__ == "__main__":
    unittest.main()
