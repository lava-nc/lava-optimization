# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np


class OptimizationProblem:
    """Abstract optimization problem from which the actual problems derive."""

    def __init__(self, cost=None, constraints=None):
        self.cost = cost
        self.constraints = constraints


class QpProblem:
    def __init__(self, Q, p, A=None, k=None, A_eq=None, k_eq=None):
        """A Rudimentary interface for the QP solver. Inequality Constraints should be of
        the form Ax<=b.
            :param Q: Quadratic term of the cost function
            :param p: Linear term of the cost function
            :param A: Inequality constrainting hyperplanes
            :param k: Ineqaulity constraints offsets
            :param A_eq: Equality constrainting hyperplanes
            :param k_eq: Eqaulity constraints offsets

        """

        try:
            if (A is None and k is not None) or (A is not None and k is None):
                raise Exception(
                    "Please properly define your Inequality constraints"
                )
        except Exception as error:
            print("Input error:")

        try:
            if (A_eq is None and k_eq is not None) or (
                A_eq is not None and k_eq is None
            ):
                raise Exception(
                    "Please properly define your Equality constraints"
                )
        except Exception as error:
            print("Input error:")

        self._Q = Q
        self._p = p
        self._A = A
        self._k = k
        self._A_eq = A_eq
        self._k_eq = k_eq

        if A_eq is not None:
            A_eq_new = np.vstack((A_eq, -A_eq))
            k_eq_new = np.vstack((k_eq, -k_eq))
            if A is not None:
                self._A = np.vstack((self._A, A_eq_new))
                self._k = np.vstack((self._k, k_eq_new))
            else:
                self._A = A_eq_new
                self._k = k_eq_new

    @property
    def hessian(self):
        return self._Q

    @property
    def num_variables(self):
        return len(self._p)
