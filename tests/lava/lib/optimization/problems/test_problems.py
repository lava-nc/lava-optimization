# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

import numpy as np

from lava.lib.optimization.problems.constraints import (
    Constraints,
    DiscreteConstraints,
)
from lava.lib.optimization.problems.cost import Cost
from lava.lib.optimization.problems.problems import (
    IQP,
    QP,
    OptimizationProblem,
    CSP,
    QUBO,
    ILP,
)
from lava.lib.optimization.problems.variables import (
    ContinuousVariables,
    DiscreteVariables,
    Variables,
)


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
        self.assertIsInstance(
            self.compliant_instantiation, OptimizationProblem
        )

    def test_not_compliant_sublcass(self):
        with self.assertRaises(TypeError):
            NotCompliantInterfaceInheritance()

    def test_variables_attribute(self):
        self.assertIsInstance(
            self.compliant_instantiation._variables, Variables
        )

    def test_constraints_attribute(self):
        self.assertIsInstance(
            self.compliant_instantiation._constraints, Constraints
        )


class TestQUBO(unittest.TestCase):
    def setUp(self) -> None:
        self.qubo = QUBO(np.eye(10, dtype=int))

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
        self.assertIsInstance(self.qubo.variables.discrete, DiscreteVariables)

    def test_variables_are_binary(self):
        for n, size in enumerate(self.qubo.variables.discrete.domain_sizes):
            with self.subTest(msg=f"Var ID {n}"):
                self.assertEqual(size, 2)

    def test_number_of_variables(self):
        self.assertEqual(self.qubo.variables.discrete.num_variables, 10)

    def test_constraints_is_none(self):
        self.assertIsNone(self.qubo.constraints)

    def test_cannot_set_constraints(self):
        with self.assertRaises(AttributeError):
            self.qubo.constraints = np.eye(10)

    def test_set_cost(self):
        new_cost = np.eye(4, dtype=int)
        self.qubo.cost = new_cost
        self.assertIs(self.qubo.cost.get_coefficient(2), new_cost)

    def test_variables_update_after_setting_cost(self):
        new_cost = np.eye(4, dtype=int)
        self.qubo.cost = new_cost
        self.assertEqual(self.qubo.variables.discrete.num_variables, 4)

    def test_class_of_setted_cost(self):
        new_cost = np.eye(10, dtype=int)
        self.qubo.cost = new_cost
        self.assertIsInstance(self.qubo.cost, Cost)

    def test_cannot_set_nonquadratic_cost(self):
        with self.assertRaises(ValueError):
            self.qubo.cost = np.eye(10, dtype=int).reshape(5, 20)

    def test_assertion_raised_if_q_is_not_square(self):
        with self.assertRaises(ValueError):
            QUBO(np.eye(10, dtype=int).reshape(5, 20))

    def test_validate_input_method_fails_assertion(self):
        with self.assertRaises(ValueError):
            self.qubo.validate_input(np.eye(10).reshape(5, 20))

    def test_validate_input_method_does_not_fail_assertion(self):
        try:
            self.qubo.validate_input(np.eye(10, dtype=int))
        except AssertionError:
            self.fail("Assertion failed with correct input!")


class TestCSP(unittest.TestCase):
    def setUp(self) -> None:
        self.domain_sizes = [5, 5, 4]
        self.constraints = [
            (0, 1, np.logical_not(np.eye(5))),
            (1, 2, np.eye(5, 4)),
        ]
        self.csp = CSP(domains=[5, 5, 4], constraints=self.constraints)

    def test_create_obj(self):
        self.assertIsInstance(self.csp, CSP)

    def test_cost_class(self):
        self.assertIsInstance(self.csp.cost, Cost)

    def test_cost_is_constant(self):
        self.assertEqual(self.csp.cost.max_degree, 0)

    def test_cannot_set_cost(self):
        with self.assertRaises(AttributeError):
            self.csp.cost = np.eye(10)

    def test_variables_class(self):
        self.assertIsInstance(self.csp.variables, DiscreteVariables)

    def test_variables_domain_sizes(self):
        for n, size in enumerate(self.csp.variables.domain_sizes):
            with self.subTest(msg=f"Var ID {n}"):
                self.assertEqual(size, self.domain_sizes[n])

    def test_constraints_class(self):
        self.assertIsInstance(self.csp.constraints, DiscreteConstraints)

    def test_constraints_var_subsets(self):
        for n, (v1, v2, rel) in enumerate(self.constraints):
            with self.subTest(msg=f"Constraint {n}."):
                self.assertEqual(self.csp.constraints.var_subsets[n], (v1, v2))

    def test_constraints_relations(self):
        for n, (v1, v2, rel) in enumerate(self.constraints):
            with self.subTest(msg=f"Constraint {n}."):
                self.assertTrue(
                    (self.csp.constraints.relations[n] == rel).all()
                )

    def test_set_constraints(self):
        new_constraints = [(0, 1, np.eye(5))]
        self.csp.constraints = new_constraints
        self.assertIs(self.csp.constraints._constraints, new_constraints)

    @unittest.skip("WIP")
    def test_validate_input_method_fails_assertion(self):
        with self.assertRaises(AssertionError):
            self.csp.validate_input(np.eye(10).reshape(5, 20))

    @unittest.skip("WIP")
    def test_validate_input_method_does_not_fail_assertion(self):
        try:
            self.csp.validate_input(np.eye(10))
        except AssertionError:
            self.fail("Assertion failed with correct input!")


class TestIQP(unittest.TestCase):
    def setUp(self):
        self.H = np.zeros((4, 4), dtype=np.int32)
        self.c = np.array([0, 1, 2, 3], dtype=np.int32).T
        self.A = np.array(
            [[0, 4, 2, 1], [2, 0, 1, 1], [1, 1, 0, 1]], dtype=np.int32
        )
        self.b = np.array([1, 1, 1], dtype=np.int32).T
        self.iqp = IQP(H=self.H, c=self.c, A=self.A, b=self.b)

    def test_create_obj(self):
        self.assertIsInstance(self.iqp, IQP)

    def test_variables_class(self):
        self.assertIsInstance(self.iqp.variables, DiscreteVariables)

    def test_cost_class(self):
        self.assertIsInstance(self.iqp.cost, Cost)

    def test_cost_is_quadratic(self):
        self.assertEqual(self.iqp.cost.max_degree, 2)

    def test_evaluate_cost(self):
        self.assertEqual(
            self.iqp.evaluate_cost(np.array([0, 1, 0, 0])), self.c[1]
        )

    def test_evaluate_constraints(self):
        self.assertTrue(
            np.all(
                self.iqp.evaluate_constraints(np.array([0, 0, 0, 0]))
                == -self.b
            )
        )


class TestQP(unittest.TestCase):
    def setUp(self):
        self.H = np.zeros((4, 4), dtype=np.int32)
        self.c = np.array([0, 1, 2, 3], dtype=np.int32).T
        self.A = np.array(
            [[0, 4, 2, 1], [2, 0, 1, 1], [1, 1, 0, 1]], dtype=np.int32
        )
        self.b = np.array([1, 1, 1], dtype=np.int32).T
        self.qp = QP(
            hessian=self.H,
            linear_offset=self.c,
            equality_constraints_weights=self.A,
            equality_constraints_biases=self.b,
        )

    def test_create_obj(self):
        self.assertIsInstance(self.qp, QP)

    def test_variables_class(self):
        self.assertIsInstance(
            self.qp.variables.continuous, ContinuousVariables
        )

    def test_cost_class(self):
        self.assertIsInstance(self.qp.cost, Cost)

    def test_cost_is_quadratic(self):
        self.assertEqual(self.qp.cost.max_degree, 2)

    def test_evaluate_cost(self):
        sol = np.array([0, 1, 0, 0])
        self.assertEqual(
            self.qp.evaluate_cost(sol), sol.T @ self.H @ sol + sol @ self.c
        )

    def test_evaluate_constraint_violations(self):
        sol = np.array([0, 0, 0, 0])
        self.assertTrue(
            np.all(
                self.qp.evaluate_constraint_violations(sol)
                == self.A @ sol - self.b
            )
        )


class TestILP(unittest.TestCase):
    def setUp(self):
        self.c = np.array([0, 1, 2, 3], dtype=np.int32).T
        self.A = np.array(
            [[0, 4, 2, 1], [2, 0, 1, 1], [1, 1, 0, 1]], dtype=np.int32
        )
        self.b = np.array([1, 1, 1], dtype=np.int32).T
        self.ilp = ILP(c=self.c, A=self.A, b=self.b)

    def test_create_obj(self):
        self.assertIsInstance(self.ilp, ILP)

    def test_variables_class(self):
        self.assertIsInstance(self.ilp.variables, DiscreteVariables)

    def test_cost_class(self):
        self.assertIsInstance(self.ilp.cost, Cost)

    def test_cost_is_quadratic(self):
        self.assertEqual(self.ilp.cost.max_degree, 1)

    def test_evaluate_cost(self):
        self.assertEqual(
            self.ilp.evaluate_cost(np.array([0, 1, 0, 0])), self.c[1]
        )

    def test_evaluate_constraints(self):
        self.assertTrue(
            np.all(
                self.ilp.evaluate_constraints(np.array([0, 0, 0, 0]))
                == -self.b
            )
        )


if __name__ == "__main__":
    unittest.main()
