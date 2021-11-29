# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import typing as ty


class QP:
    """A Rudimentary interface for the QP solver. Inequality Constraints
    should be of the form Ax<=k. Equality constraints are expressed as
    sandwiched inequality constraints. The cost of the QP is of the form
    1/2*x^t*Q*x + p^Tx

        Parameters
        ----------
        hessian : 2-D or 1-D np.array
            Quadratic term of the cost function
        linear_offset : 1-D np.array, optional
            Linear term of the cost function, defaults vector of zeros of the
            size of the number of variables in the QP
        constraint_hyperplanes : 2-D or 1-D np.array, optional
            Inequality constrainting hyperplanes, by default None
        constraint_biases : 1-D np.array, optional
            Ineqaulity constraints offsets, by default None
        constraint_hyperplanes_eq : 2-D or 1-D np.array, optional
            Equality constrainting hyperplanes, by default None
        constraint_biases_eq : 1-D np.array, optional
            Eqaulity constraints offsets, by default None

        Raises
        ------
        ValueError
            ValueError exception raised if equality or inequality constraints
            are not properly defined. Ex: Defining A_eq while not defining k_eq
            and vice-versa.
    """

    def __init__(
        self,
        hessian: np.ndarray,
        linear_offset: ty.Optional[np.ndarray] = None,
        constraint_hyperplanes: ty.Optional[np.ndarray] = None,
        constraint_biases: ty.Optional[np.ndarray] = None,
        constraint_hyperplanes_eq: ty.Optional[np.ndarray] = None,
        constraint_biases_eq: ty.Optional[np.ndarray] = None,
    ):
        if (
            constraint_hyperplanes is None and constraint_biases is not None
        ) or (
            constraint_hyperplanes is not None and constraint_biases is None
        ):
            raise ValueError(
                "Please properly define your Inequality constraints. Supply \
                all A and k "
            )

        if (
            constraint_hyperplanes_eq is None
            and constraint_biases_eq is not None
        ) or (
            constraint_hyperplanes_eq is not None
            and constraint_biases_eq is None
        ):
            raise ValueError(
                "Please properly define your Equality constraints. Supply \
                all A_eq and k_eq."
            )

        self._hessian = hessian

        if linear_offset is not None:
            self._linear_offset = linear_offset
        else:
            self._linear_offset = np.zeros((hessian.shape[0], 1))

        if constraint_hyperplanes is not None:
            self._constraint_hyperplanes = constraint_hyperplanes
            self._constraint_biases = constraint_biases
        else:
            self._constraint_hyperplanes = None
            self._constraint_biases = None

        if constraint_hyperplanes_eq is not None:
            self._constraint_hyperplanes_eq = constraint_hyperplanes_eq
            self._constraint_biases_eq = constraint_biases_eq

        if constraint_hyperplanes_eq is not None:
            constraint_hyperplanes_eq_new = np.vstack(
                (constraint_hyperplanes_eq, -constraint_hyperplanes_eq)
            )
            constraint_biases_eq_new = np.vstack(
                (constraint_biases_eq, -constraint_biases_eq)
            )
            if constraint_hyperplanes is not None:
                self._constraint_hyperplanes = np.vstack(
                    (
                        self._constraint_hyperplanes,
                        constraint_hyperplanes_eq_new,
                    )
                )
                self._constraint_biases = np.vstack(
                    (self._constraint_biases, constraint_biases_eq_new)
                )
            else:
                self._constraint_hyperplanes = constraint_hyperplanes_eq_new
                self._constraint_biases = constraint_biases_eq_new

    @property
    def get_hessian(self) -> np.ndarray:
        return self._hessian

    @property
    def get_linear_offset(self) -> np.ndarray:
        return self._linear_offset

    @property
    def get_constraint_hyperplanes(self) -> np.ndarray:
        return self._constraint_hyperplanes

    @property
    def get_constraint_biases(self) -> np.ndarray:
        return self._constraint_biases

    @property
    def num_variables(self) -> int:
        return len(self._linear_offset)
