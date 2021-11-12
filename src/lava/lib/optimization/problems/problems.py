# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np


class QP:
    """A Rudimentary interface for the QP solver. Inequality Constraints
    should be of the form Ax<=k. Equality constraints are expressed as
    sandwiched inequality constraints. The cost of the QP is of the form
    1/2*x^t*Q*x + p^Tx

        Parameters
        ----------
        Q : 2-D or 1-D np.array
            Quadratic term of the cost function
        p : 1-D np.array, optional
            Linear term of the cost function, defaults vector of zeros of the
            size of the number of variables in the QP
        A : 2-D or 1-D np.array, optional
            Inequality constrainting hyperplanes, by default None
        k : 1-D np.array, optional
            Ineqaulity constraints offsets, by default None
        A_eq : 2-D or 1-D np.array, optional
            Equality constrainting hyperplanes, by default None
        k_eq : 1-D np.array, optional
            Eqaulity constraints offsets, by default None

        Raises
        ------
        ValueError
            ValueError exception raised if equality or inequality constraints
            are not properly defined. Ex: Defining A_eq while not defining k_eq
            and vice-versa.
    """

    def __init__(self, Q, p=None, A=None, k=None, A_eq=None, k_eq=None):
        if (A is None and k is not None) or (A is not None and k is None):
            raise ValueError(
                "Please properly define your Inequality constraints. Supply \
                all A and k "
            )

        if (A_eq is None and k_eq is not None) or (
            A_eq is not None and k_eq is None
        ):
            raise ValueError(
                "Please properly define your Equality constraints. Supply \
                all A_eq and k_eq."
            )

        self._Q = Q

        if p is not None:
            self._p = p
        else:
            self._p = np.zeros((Q.shape[0], 1))

        if A is not None:
            self._A = A
            self._k = k
        else:
            self._A = None
            self._k = None

        if A_eq is not None:
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
    def get_hessian(self):
        return self._Q

    @property
    def get_linear_cost(self):
        return self._p

    @property
    def get_constraint_matrix(self):
        return self._A

    @property
    def get_constraint_offset(self):
        return self._k

    @property
    def num_variables(self):
        return len(self._p)
