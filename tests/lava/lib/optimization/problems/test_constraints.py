# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/


import unittest

import numpy as np
from lava.lib.optimization.problems.constraints import DiscreteConstraints


class TestDiscreteConstraint(unittest.TestCase):
    def setUp(self) -> None:
        self.constraints = [(0, 1, np.logical_not(np.eye(5))),
                            (1, 2, np.eye(5, 4))]
        self.relations_2d = [np.logical_not(np.eye(5)), np.eye(5, 4)]
        self.constraint = DiscreteConstraints(self.constraints)

    def test_create_obj(self):
        self.assertIsInstance(self.constraint, DiscreteConstraints)

    def test_has_var_subset(self):
        self.assertEqual(self.constraint.var_subsets, [(0, 1), (1, 2)])

    def test_var_subset_is_required(self):
        with self.assertRaises(TypeError):
            constraint = DiscreteConstraints()

    def test_has_relation(self):
        self.assertTrue((self.constraint.relations[0] == self.relations_2d[
            0]).all())

    def test__input_validation_relation_matches_var_subset_dimension(self):
        constraints = [(0, 1, 2, np.logical_not(np.eye(5))),
                       (1, 2, np.eye(5, 4))]
        with self.assertRaises(AssertionError):
            DiscreteConstraints(constraints)


if __name__ == '__main__':
    unittest.main()
