# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

import numpy as np
from src.lava.lib.optimization.problems.constraints import Constraints
from src.lava.lib.optimization.problems.cost import Cost
from src.lava.lib.optimization.problems.problems import (OptimizationProblem,
                                                         QUBO)
from src.lava.lib.optimization.problems.variables import (Variables,
                                                          DiscreteVariables)


class CompliantInterfaceInheritance(OptimizationProblem):
    @property
    def cost(self):
        return None

    @property
    def constraints(self):
        return None

    @property
    def variables(self):
        return None


class NotCompliantInterfaceInheritance(OptimizationProblem):
    def constraints(self):
        return None

    def variables(self):
        return None


class TestOptimizationProblem(unittest.TestCase):
    def setUp(self) -> None:
        self.compliant_instantiation = CompliantInterfaceInheritance()

    def test_cannot_create_instance(self):
        with self.assertRaises(TypeError):
            OptimizationProblem()

    def test_compliant_sublcass(self):
        self.assertIsInstance(self.compliant_instantiation, OptimizationProblem)

    def test_not_compliant_sublcass(self):
        with self.assertRaises(TypeError):
            NotCompliantInterfaceInheritance()

    def test_variables_attribute(self):
        self.assertIsInstance(self.compliant_instantiation._variables,
                              Variables)

    def test_constraints_attribute(self):
        self.assertIsInstance(self.compliant_instantiation._constraints,
                              Constraints)


class TestQUBO(unittest.TestCase):
    def setUp(self) -> None:
        self.qubo = QUBO(np.eye(10))

    def test_create_obj(self):
        self.assertIsInstance(self.qubo, QUBO)

    def test_q_is_square_matrix(self):
        n, m = self.qubo.cost.coefficients[2].shape
        self.assertEqual(n, m)

    def test_cost_class(self):
        self.assertIsInstance(self.qubo.cost, Cost)

    def test_only_quadratic_term_in_cost(self):
        self.assertEqual(list(self.qubo.cost.coefficients.keys()), [2])

    def test_variables_class(self):
        self.assertIsInstance(self.qubo.variables, DiscreteVariables)

    def test_variables_are_binary(self):
        for n, size in enumerate(self.qubo.variables.domain_sizes):
            with self.subTest(msg=f'Var ID {n}'):
                self.assertEqual(size, 2)

    def test_constraints_is_none(self):
        self.assertIsNone(self.qubo.constraints)

    def test_cannot_set_constraints(self):
        with self.assertRaises(AttributeError):
            self.qubo.constraints = np.eye(10)

    def test_set_cost(self):
        new_cost = np.eye(10)
        self.qubo.cost = new_cost
        self.assertIs(self.qubo.cost.get_coefficient(2), new_cost)

    def test_class_of_setted_cost(self):
        new_cost = np.eye(10)
        self.qubo.cost = new_cost
        self.assertIsInstance(self.qubo.cost, Cost)

    def test_cannot_set_nonquadratic_cost(self):
        with self.assertRaises(AssertionError):
            self.qubo.cost = np.eye(10).reshape(5, 20)

    def test_assertion_raised_if_q_is_not_square(self):
        with self.assertRaises(AssertionError):
            QUBO(np.eye(10).reshape(5, 20))

    def test_validate_input_method_fails_assertion(self):
        with self.assertRaises(AssertionError):
            self.qubo.validate_input(np.eye(10).reshape(5, 20))

    def test_validate_input_method_does_not_fail_assertion(self):
        try:
            self.qubo.validate_input(np.eye(10))
        except AssertionError as ex:
            self.fail("Assertion failed with correct input!")


if __name__ == '__main__':
    unittest.main()
