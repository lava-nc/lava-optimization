# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/


import unittest

import numpy as np
from lava.lib.optimization.problems.constraints import DiscreteConstraints


class TestDiscreteConstraint(unittest.TestCase):
    def setUp(self) -> None:
        constraints = [(0, 1, np.logical_not(np.eye(5))),
                       (1, 2, np.eye(5, 4))]
        self.relations_2d = [np.logical_not(np.eye(5)), np.eye(5, 4)]
        self.dconstraint = DiscreteConstraints(constraints)

    def test_create_obj(self):
        self.assertIsInstance(self.dconstraint, DiscreteConstraints)

    def test_var_subset(self):
        self.assertEqual(self.dconstraint.var_subsets, [(0, 1), (1, 2)])

    def test_var_subset_is_required(self):
        with self.assertRaises(TypeError):
            constraint = DiscreteConstraints()

    def test_relation(self):
        for n, relation in enumerate(self.relations_2d):
            with self.subTest(msg=f'Test id {n}'):
                self.assertTrue((self.dconstraint.relations[n] ==
                                 relation).all())

    def test__input_validation_relation_matches_var_subset_dimension(self):
        constraints = [(0, 1, 2, np.logical_not(np.eye(5))),
                       (1, 2, np.eye(5, 4))]
        with self.assertRaises(AssertionError):
            DiscreteConstraints(constraints)

    def test_set_constraints(self):
        new_constraints = [(1, 2, np.logical_not(np.eye(5, 4))),
                           (0, 1, np.eye(5))]
        self.dconstraint.constraints = new_constraints
        self.assertIs(self.dconstraint.constraints, new_constraints)

    def test_setted_relation(self):
        new_constraints = [(1, 2, np.logical_not(np.eye(5, 4))),
                           (0, 1, np.eye(5))]
        self.dconstraint.constraints = new_constraints
        new_relations = [np.logical_not(np.eye(5, 4)),
                         np.eye(5)]
        for n, relation in enumerate(new_relations):
            with self.subTest(msg=f'Test id {n}'):
                self.assertTrue((self.dconstraint.relations[n] ==
                                 relation).all())

    def test_setted_var_subset(self):
        new_constraints = [(1, 2, np.logical_not(np.eye(5, 4))),
                           (0, 1, np.eye(5))]
        self.dconstraint.constraints = new_constraints
        self.assertEqual(self.dconstraint.var_subsets, [(1, 2), (0, 1)])

    def test_var_subsets_from_function_set_relations_var_subsets(self):
        new_constraints = [(1, 2, np.logical_not(np.eye(5, 4))),
                           (0, 1, np.eye(5))]
        self.dconstraint.set_relations_var_subsets(new_constraints)
        self.assertEqual(self.dconstraint._var_subset, [(1, 2), (0, 1)])

    def test__relations_from_function_set_relations_var_subsets(self):
        new_constraints = [(1, 2, np.logical_not(np.eye(5, 4))),
                           (0, 1, np.eye(5))]
        self.dconstraint.set_relations_var_subsets(new_constraints)
        for n, relation in enumerate([np.logical_not(np.eye(5, 4)), np.eye(5)]):
            with self.subTest(msg=f'Relation index {n}'):
                self.assertTrue(
                    (self.dconstraint._relations[n] == relation).all())


if __name__ == '__main__':
    unittest.main()
