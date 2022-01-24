# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/


import unittest

from src.lava.lib.optimization.problems.variables import (Variable)


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


if __name__ == '__main__':
    unittest.main()
