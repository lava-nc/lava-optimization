# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np
from src.lava.lib.optimization.problems.constraints import (
    AbstractConstraint,
    DiscreteConstraint,
    # ContinuousConstraint,
    # MixedConstraint,
    # EqualityConstraint,
    # InequalityConstraint,
)


class TestAbstractConstraint(unittest.TestCase):
    def setUp(self) -> None:
        self.var_subset_1d = np.random.randint(0, 10, 5)
        self.constraint = AbstractConstraint(var_subset=self.var_subset_1d)

    def test_create_obj(self):
        self.assertIsInstance(self.constraint, AbstractConstraint)

    def test_is_linear_property_is_none(self):
        self.assertIsNone(self.constraint.is_linear)

    def test_is_discrete_property_is_None(self):
        self.assertIsNone(self.constraint.is_discrete)

    def test_has_var_subset(self):
        self.assertIs(self.constraint.var_subset, self.var_subset_1d)

    def test_can_set_is_discrete_attribute(self):
        self.constraint.is_discrete = True
        self.assertTrue(self.constraint.is_discrete)


class TestDiscreteConstraint(unittest.TestCase):
    def setUp(self) -> None:
        self.var_subset_1d = np.random.randint(0, 10, 5)
        self.relation_2d = np.random.randint(0, 10, (5, 5))
        self.constraint = DiscreteConstraint(
            var_subset=self.var_subset_1d, relation=self.relation_2d
        )

    def test_create_obj(self):
        self.assertIsInstance(self.constraint, AbstractConstraint)
        self.assertIsInstance(self.constraint, DiscreteConstraint)

    def test_relation_defaults_to_none(self):
        constraint = DiscreteConstraint(var_subset=self.var_subset_1d)
        self.assertIsNone(constraint.relation)

    def test_has_relation(self):
        self.assertIs(self.constraint.relation, self.relation_2d)

    def test_is_discrete_is_true(self):
        self.assertTrue(self.constraint)

    def test_is_discrete_property_is_true(self):
        self.assertTrue(self.constraint.is_discrete)


class TestContinuousConstraint(unittest.TestCase):
    pass


class TestMixedConstraint(unittest.TestCase):
    pass


class TestInequalityConstraint(unittest.TestCase):
    pass


class TestEqualityConstraint(unittest.TestCase):
    pass


class TestUnconstrained(unittest.TestCase):
    pass


if __name__ == "__main__":
    unittest.main()
