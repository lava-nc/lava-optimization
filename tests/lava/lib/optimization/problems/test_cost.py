# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/


import unittest

import numpy as np

from lava.lib.optimization.problems.coefficients import CoefficientTensorsMixin
from lava.lib.optimization.problems.cost import Cost


class TestCost(unittest.TestCase):
    def setUp(self) -> None:
        self.raw_coefficients = (
            np.asarray(1),
            np.ones(2),
            np.ones((2, 2)),
            np.ones((2, 2, 2)),
        )
        self.augmented_terms = (
            np.asarray(5),
            5 * np.ones(2),
            5 * np.ones((2, 2)),
            5 * np.ones((2, 2, 2)),
        )
        self.cost = Cost(*self.raw_coefficients)
        self.cost_augmented = Cost(*self.raw_coefficients,
                                   augmented_terms=self.augmented_terms)

    def test_create_obj(self):
        for cost, label in [
            (self.cost, "not augmented"),
            (self.cost_augmented, "augmented"),
        ]:
            with self.subTest(msg=f"{label}"):
                self.assertIsInstance(cost, Cost)

    def test_created_obj_includes_mixin(self):
        for cost, label in [
            (self.cost, "not augmented"),
            (self.cost_augmented, "augmented"),
        ]:
            with self.subTest(msg=f"{label}"):
                self.assertIsInstance(cost, CoefficientTensorsMixin)

    def test_coefficients(self):
        for n, coefficient in enumerate(self.raw_coefficients):
            with self.subTest(msg=f"{n}"):
                self.assertTrue(
                    (coefficient == self.cost.coefficients[n]).all()
                )

    def test_is_augmented(self):
        self.assertTrue(self.cost_augmented.is_augmented)

    def test_is_not_augmented(self):
        self.assertFalse(self.cost.is_augmented)

    def test_augmented_terms(self):
        for n, coefficient in enumerate(self.augmented_terms):
            with self.subTest(msg=f"{n}"):
                aug_terms = self.cost_augmented.augmented_terms
                self.assertTrue((coefficient == aug_terms[n]).all())

    def test_reset_augmented_terms(self):
        new_augmented_terms = (
            np.asarray(3),
            3 * np.ones(2),
            3 * np.ones((2, 2)),
            3 * np.ones((2, 2, 2)),
        )
        self.cost_augmented.augmented_terms = new_augmented_terms
        for n, coefficient in enumerate(new_augmented_terms):
            with self.subTest(msg=f"{n}"):
                aug_terms = self.cost_augmented.augmented_terms
                self.assertTrue((coefficient == aug_terms[n]).all())

    def test_set_augmented_terms(self):
        new_augmented_terms = (
            np.asarray(3),
            3 * np.ones(2),
            3 * np.ones((2, 2)),
            3 * np.ones((2, 2, 2)),
        )
        self.cost.augmented_terms = new_augmented_terms
        for n, coefficient in enumerate(new_augmented_terms):
            with self.subTest(msg=f"{n}"):
                self.assertTrue(
                    (coefficient == self.cost.augmented_terms[n]).all()
                )

    def test_augmented_terms_defaults_to_none(self):
        self.assertIsNone(self.cost.augmented_terms)


if __name__ == "__main__":
    unittest.main()
