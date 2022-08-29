# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

import numpy as np

from lava.lib.optimization.problems.problems import QUBO
from lava.lib.optimization.solvers.generic.solver import \
    OptimizationSolver


class TestOptimizationSolver(unittest.TestCase):
    def setUp(self) -> None:
        self.solver = OptimizationSolver()
        self.problem = QUBO(np.asarray([[-5, 2, 4, 0],
                                        [2, -3, 1, 0],
                                        [4, 1, -8, 5],
                                        [0, 0, 5, -6]]))
        self.solution = np.asarray([1, 0, 0, 1]).astype(int)

    def test_create_obj(self):
        self.assertIsInstance(self.solver, OptimizationSolver)


if __name__ == '__main__':
    unittest.main()
