# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
from src.lava.lib.optimization.problems.constraints import Constraints
from src.lava.lib.optimization.problems.cost import Cost
from src.lava.lib.optimization.problems.variables import (Variables,
                                                          DiscreteVariables)


class OptimizationProblem(ABC):
    """Interface for any concrete optimization problem.

    Any optimization problem can be defined by a set of variables, cost and
    constraints.  Although for some problems some of these elements may be
    absent, the user still has to specify them, e.g., defining constraints as
    None.
    """

    def __init__(self):
        self._variables = Variables()
        self._constraints = Constraints()

    @property
    @abstractmethod
    def variables(self):
        """Variables over which the optimization problem is defined."""
        pass

    @property
    @abstractmethod
    def cost(self):
        """Function to be optimized and defined over the problem variables."""
        pass

    @property
    @abstractmethod
    def constraints(self):
        """Constrains to be satisfied by mutual assignments to variables."""
        pass


class QUBO(OptimizationProblem):
    def __init__(self, q: npt.ArrayLike):
        """A Quadratic Unconstrained Binary Optimization (QUBO) problem.

        The cost to be minimized is of the form $x^T \cdot Q \cdot x$.
        the problem is unconstrained by definition, thus, constraints are set to
        None. Variables are binary and their number must match the dimension of
        the Q matrix.

        :param q: squared Q matrix defining the QUBO problem over a binary
        vector x as: minimize x^T*Q*x.
        """
        super().__init__()
        self.validate_input(q)
        self._q_cost = Cost(q)
        self._b_variables = DiscreteVariables(domains=[2] * q.shape[0])

    @property
    def variables(self):
        """Binary variables of the QUBO problem."""
        return self._b_variables

    @property
    def cost(self):
        """Quadratic cost to be minimized."""
        return self._q_cost

    @cost.setter
    def cost(self, value):
        """Quadratic cost setter, binary variables are updated accordingly."""
        self.validate_input(value)
        q = Cost(value)
        assert list(q.coefficients.keys()) == [2], "Cost must be a quadratic " \
                                                   "matrix."
        self._b_variables = DiscreteVariables(domains=[2] * value.shape[0])
        self._q_cost = q

    @property
    def constraints(self):
        """As an unconstrained problem, QUBO constraints are None."""
        return None

    def validate_input(self, q):
        """Validate the cost coefficient is a square matrix."""
        m, n = q.shape
        assert m == n, "q matrix is not a square matrix."


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
