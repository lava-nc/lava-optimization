# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

from src.lava.lib.optimization.problems.constraints import Constraints
from src.lava.lib.optimization.problems.problems import OptimizationProblem
from src.lava.lib.optimization.problems.variables import Variables


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


if __name__ == '__main__':
    unittest.main()
