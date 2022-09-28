# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/


import unittest

import numpy as np

from lava.lib.optimization.problems.coefficients import CoefficientTensorsMixin
from lava.lib.optimization.problems.constraints import (ArithmeticConstraints,
                                                        Constraints,
                                                        DiscreteConstraints,
                                                        EqualityConstraints,
                                                        InequalityConstraints)


class TestDiscreteConstraint(unittest.TestCase):
    def setUp(self) -> None:
        constraints = [(0, 1, np.logical_not(np.eye(5))), (1, 2, np.eye(5, 4))]
        self.relations_2d = [np.logical_not(np.eye(5)), np.eye(5, 4)]
        self.dconstraint = DiscreteConstraints(constraints)

    def test_create_obj(self):
        self.assertIsInstance(self.dconstraint, DiscreteConstraints)

    def test_var_subset(self):
        self.assertEqual(self.dconstraint.var_subsets, [(0, 1), (1, 2)])

    def test_var_subset_is_required(self):
        with self.assertRaises(TypeError):
            DiscreteConstraints()

    def test_relation(self):
        for n, relation in enumerate(self.relations_2d):
            with self.subTest(msg=f"Test id {n}"):
                self.assertTrue(
                    (self.dconstraint.relations[n] == relation).all()
                )

    def test__input_validation_relation_matches_var_subset_dimension(self):
        constraints = [
            (0, 1, 2, np.logical_not(np.eye(5))),
            (1, 2, np.eye(5, 4)),
        ]
        with self.assertRaises(ValueError):
            DiscreteConstraints(constraints)

    def test_set_constraints(self):
        new_constraints = [
            (1, 2, np.logical_not(np.eye(5, 4))),
            (0, 1, np.eye(5)),
        ]
        self.dconstraint.constraints = new_constraints
        self.assertIs(self.dconstraint.constraints, new_constraints)

    def test_setted_relation(self):
        new_constraints = [
            (1, 2, np.logical_not(np.eye(5, 4))),
            (0, 1, np.eye(5)),
        ]
        self.dconstraint.constraints = new_constraints
        new_relations = [np.logical_not(np.eye(5, 4)), np.eye(5)]
        for n, relation in enumerate(new_relations):
            with self.subTest(msg=f"Test id {n}"):
                self.assertTrue(
                    (self.dconstraint.relations[n] == relation).all()
                )

    def test_setted_var_subset(self):
        new_constraints = [
            (1, 2, np.logical_not(np.eye(5, 4))),
            (0, 1, np.eye(5)),
        ]
        self.dconstraint.constraints = new_constraints
        self.assertEqual(self.dconstraint.var_subsets, [(1, 2), (0, 1)])

    def test_var_subsets_from_function_set_relations_var_subsets(self):
        new_constraints = [
            (1, 2, np.logical_not(np.eye(5, 4))),
            (0, 1, np.eye(5)),
        ]
        self.dconstraint.set_relations_var_subsets(new_constraints)
        self.assertEqual(self.dconstraint._var_subset, [(1, 2), (0, 1)])

    def test__relations_from_function_set_relations_var_subsets(self):
        new_constraints = [
            (1, 2, np.logical_not(np.eye(5, 4))),
            (0, 1, np.eye(5)),
        ]
        self.dconstraint.set_relations_var_subsets(new_constraints)
        for n, relation in enumerate([np.logical_not(np.eye(5, 4)), np.eye(5)]):
            with self.subTest(msg=f"Relation index {n}"):
                self.assertTrue(
                    (self.dconstraint._relations[n] == relation).all()
                )


class TestEqualityConstraint(unittest.TestCase):
    def setUp(self) -> None:
        coefficients_np = (
            np.asarray(1),
            np.ones(2),
            np.ones((2, 2)),
            np.ones((2, 2, 2)),
        )
        self.constraint = EqualityConstraints(*coefficients_np)

    def test_create_obj(self):
        self.assertIsInstance(self.constraint, EqualityConstraints)

    def test_created_obj_includes_mixin(self):
        self.assertIsInstance(self.constraint, CoefficientTensorsMixin)


class TestInequalityConstraint(unittest.TestCase):
    def setUp(self) -> None:
        coefficients_np = (
            np.asarray(1),
            np.ones(2),
            np.ones((2, 2)),
            np.ones((2, 2, 2)),
        )
        self.constraint = InequalityConstraints(*coefficients_np)

    def test_create_obj(self):
        self.assertIsInstance(self.constraint, InequalityConstraints)

    def test_created_obj_includes_mixin(self):
        self.assertIsInstance(self.constraint, CoefficientTensorsMixin)


class TestArithmeticConstraint(unittest.TestCase):
    def setUp(self) -> None:
        self.constraint = ArithmeticConstraints()

    def test_create_obj(self):
        self.assertIsInstance(self.constraint, ArithmeticConstraints)

    def test_set_arithmetic_constraints_equality(self):
        new_constraints_eq = (
            np.asarray(1),
            np.ones(2),
            np.ones((2, 2)),
        )
        self.constraint.equality = new_constraints_eq
        for n, coefficient in enumerate(new_constraints_eq):
            with self.subTest(msg=f"{n}"):
                coeffs = self.constraint.equality.coefficients
                self.assertTrue((coefficient == coeffs[n]).all())

    def test_set_arithmetic_constraints_inequality(self):
        new_constraints_ineq = (
            np.asarray(1),
            np.ones(2),
            np.ones((2, 2)),
        )
        self.constraint.inequality = new_constraints_ineq
        for n, coefficient in enumerate(new_constraints_ineq):
            with self.subTest(msg=f"{n}"):
                self.assertTrue(
                    (coefficient == self.constraint.inequality.coefficients[n]
                     ).all())


class TestConstraints(unittest.TestCase):
    def setUp(self) -> None:
        self.constraints = Constraints()

    def test_create_obj(self):
        self.assertIsInstance(self.constraints, Constraints)

    def test_discrete_defaults_to_none(self):
        self.assertIsNone(self.constraints.discrete)

    def test_arithmetic_defaults_to_none(self):
        self.assertIsNone(self.constraints.arithmetic)

    def test_set_discrete_constraints(self):
        new_constraints = [(0, 1, np.eye(5))]
        self.constraints.discrete = DiscreteConstraints(new_constraints)
        self.assertIs(self.constraints.discrete._constraints, new_constraints)

    def test_class_of_setted_discrete_constraints(self):
        new_constraints = [(0, 1, np.eye(5))]
        self.constraints.discrete = DiscreteConstraints(new_constraints)
        self.assertIsInstance(self.constraints.discrete, DiscreteConstraints)

    def test_set_arithmetic_constraint(self):
        new_constraint = ArithmeticConstraints()
        self.constraints.arithmetic = new_constraint
        self.assertIs(self.constraints.arithmetic, new_constraint)

    def test_class_of_setted_arithmetic_constraints(self):
        new_constraint = ArithmeticConstraints()
        self.constraints.arithmetic = new_constraint
        self.assertIsInstance(
            self.constraints.arithmetic, ArithmeticConstraints
        )


if __name__ == "__main__":
    unittest.main()
