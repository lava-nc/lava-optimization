# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import unittest
import numpy as np
from lava.lib.optimization.solvers.generic.scif.process import CspScif, QuboScif


class TestCspScifProcess(unittest.TestCase):
    """Tests for CspScif class"""

    def test_init(self) -> None:
        """Tests instantiation of CspScif"""
        scif = CspScif(shape=(10,), step_size=2, theta=8, sustained_on_tau=-10)

        self.assertEqual(scif.shape, (10,))
        self.assertEqual(scif.step_size.init, 2)
        self.assertEqual(scif.theta.init, 8)
        self.assertEqual(scif.sustained_on_tau.init, -10)


class TestQuboScifProcess(unittest.TestCase):
    """Tests for QuboScif class"""

    def test_init(self) -> None:
        """Tests instantiation of QuboScif"""
        scif = QuboScif(shape=(10,), theta=8, cost_diag=np.arange(1, 11))

        self.assertEqual(scif.shape, (10,))
        self.assertEqual(scif.theta.init, 8)
        self.assertTrue(np.all(scif.cost_diagonal.init == np.arange(1, 11)))


if __name__ == "__main__":
    unittest.main()
