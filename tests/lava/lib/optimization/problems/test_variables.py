# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/


import unittest

from lava.lib.optimization.problems.variables import (
    ContinuousVariables,
    DiscreteVariables,
    Variable,
    Variables,
)


class TestVariable(unittest.TestCase):
    def setUp(self) -> None:
        self.name = "var"
        self.var = Variable(name=self.name)

    def test_obj_creation(self):
        self.assertIsInstance(self.var, Variable)

    def test_value_assignment(self):
        self.var.value = 10
        self.assertEqual(self.var.value, 10)

    def test_initial_value_is_none(self):
        self.assertIsNone(self.var.value)

    def test_name_validation(self):
        self.assertIs(self.var.name, self.name)


class TestDiscreteVariables(unittest.TestCase):
    def setUp(self) -> None:
        self.domains = [10, 3, 11]
        self.dvars = DiscreteVariables(domains=self.domains)

    def test_obj_creation(self):
        self.assertIsInstance(self.dvars, DiscreteVariables)

    def test_variable_set_creation(self):
        for n, var in enumerate(self.dvars.variable_set):
            with self.subTest(msg=f"ID={n}"):
                self.assertIsInstance(var, Variable)

    def test_len_variable_set(self):
        self.assertEqual(len(self.dvars.variable_set), 3)

    def test_num_variables(self):
        self.assertEqual(self.dvars.num_variables, 3)

    def test_domain_sizes(self):
        self.assertEqual(self.dvars.domain_sizes, self.domains)

    def test_domains(self):
        for n, domain in enumerate(self.dvars.domains):
            with self.subTest(msg=f"{n}"):
                self.assertEqual(len(domain), self.domains[n])

    def test_set_domains(self):
        self.dvars.domains = [4, 5, 4, 5]
        self.assertEqual(self.dvars.domain_sizes, [4, 5, 4, 5])

    def test_domains_defaults_to_none(self):
        dvars = DiscreteVariables()
        self.assertIsNone(dvars.domains)

    @unittest.skip("WIP")
    def test_domains_validation(self):
        pass


class TestContinuousVariables(unittest.TestCase):
    def setUp(self) -> None:
        self.bounds = [(0, 1), (0, 20), (-1, 4)]
        self.cvars = ContinuousVariables(
            num_variables=len(self.bounds), bounds=self.bounds
        )

    def test_obj_creation(self):
        self.assertIsInstance(self.cvars, ContinuousVariables)

    def test_set_bounds(self):
        self.cvars.bounds = [(1, 2), (0, 5)]
        self.assertEqual(self.cvars.bounds, [(1, 2), (0, 5)])

    def test_num_variables(self):
        self.assertEqual(self.cvars.num_variables, 3)

    def test_variable_set_creation(self):
        for n, var in enumerate(self.cvars.variable_set):
            with self.subTest(msg=f"ID={n}"):
                self.assertIsInstance(var, Variable)

    def test_len_variable_set(self):
        self.assertEqual(len(self.cvars.variable_set), 3)

    def test_get_bounds(self):
        self.assertIs(self.cvars.bounds, self.bounds)

    @unittest.skip("WIP")
    def test_bounds_validation(self):
        pass


class TestVariables(unittest.TestCase):
    def setUp(self) -> None:
        self.vars = Variables()

    def test_obj_creation(self):
        self.assertIsInstance(self.vars, Variables)

    def test_has_discrete_bucket(self):
        self.assertIsInstance(self.vars.discrete, DiscreteVariables)

    def test_has_continuous_bucket(self):
        self.assertIsInstance(self.vars.continuous, ContinuousVariables)

    def test_set_discrete(self):
        new_vars = DiscreteVariables([3, 3, 3])
        self.vars.discrete = new_vars
        self.assertIs(self.vars.discrete, new_vars)

    def test_set_continuous(self):
        new_vars = ContinuousVariables([(1, 4), (0, 1)])
        self.vars.continuous = new_vars
        self.assertIs(self.vars.continuous, new_vars)

    def test_discrete_attribute_class(self):
        self.assertIsInstance(self.vars.discrete, DiscreteVariables)

    def test_continuous_attribute_class(self):
        self.assertIsInstance(self.vars.continuous, ContinuousVariables)


if __name__ == "__main__":
    unittest.main()
