# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

import numpy as np

from lava.lib.optimization.problems.coefficients import CoefficientTensorsMixin


class TestCoefficientsTensors(unittest.TestCase):
    def setUp(self) -> None:
        self.coefficients_np = CoefficientTensorsMixin(
            np.asarray(1), np.ones(2), np.ones((2, 2)), np.ones((2, 2, 2))
        )
        self.coefficients_list = CoefficientTensorsMixin(
            1, [1, 1], [[1, 1], [1, 1]], [[[1, 1], [1, 1]], [[1, 1], [1, 1]]]
        )

    def test_create_obj(self):
        for coeffs, name in (
                (self.coefficients_list, "Lists"),
                (self.coefficients_np, "Numpy"),
        ):
            with self.subTest(msg=f"Test for input as {name}"):
                self.assertIsInstance(coeffs, CoefficientTensorsMixin)

    def test_get_coefficient(self):
        coeff_rank0 = (np.ones(2),)
        coeff_rank1 = (np.ones((2, 2)),)
        coeff_rank2 = (np.ones((2, 2)),)
        coeff_rank3 = np.ones((2, 2, 2))
        expected = (coeff_rank0, coeff_rank1, coeff_rank2, coeff_rank3)
        for order in range(4):
            with self.subTest(msg=f"Test for order {order}"):
                result = self.coefficients_np.get_coefficient(order=order)
                self.assertTrue((result == expected[order]).all())

    def test_trying_to_get_absent_coefficient(self):
        with self.assertRaises(KeyError):
            self.coefficients_np.get_coefficient(order=5)

    def test_getting_the_max_degree(self):
        self.assertEqual(self.coefficients_np.max_degree, 3)

    def test_is_ok_not_providing_intermediate_term(self):
        coefficients = CoefficientTensorsMixin(
            np.asarray(1), np.ones((2, 2)), np.ones((2, 2, 2))
        )
        self.assertIsInstance(coefficients, CoefficientTensorsMixin)
        provided_ranks = sorted(list(coefficients.coefficients.keys()))
        self.assertEqual(provided_ranks, [0, 2, 3])

    def test_wrong_input_raises_exception(self):
        for value in ["input_string", (1, 2, 3)]:
            with self.subTest(msg=f"test for {value} as wrong input"):
                with self.assertRaises(ValueError):
                    CoefficientTensorsMixin(value)

    def test_set_coefficients(self):
        new_coefficients = (
            np.asarray(5),
            4 * np.ones(2),
            3 * np.ones((2, 2)),
            2 * np.ones((2, 2, 2)),
        )
        self.coefficients_np.coefficients = new_coefficients
        self.assertIs(self.coefficients_np.coefficients, new_coefficients)

    def test_evaluate_method(self):
        ctm = CoefficientTensorsMixin(
            1, [1, 1], [[1, 1], [1, 1]]
        )
        self.assertEqual(ctm.evaluate(np.array([1, 1])), 7)


if __name__ == "__main__":
    unittest.main()
