# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from lava.lib.optimization.problems.constraints import (
    Constraints,
    ArithmeticConstraints,
    DiscreteConstraints,
)
from lava.lib.optimization.problems.cost import Cost
from lava.lib.optimization.problems.variables import (
    Variables,
    ContinuousVariables,
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
        r"""A Quadratic Unconstrained Binary Optimization (QUBO) problem.

        The cost to be minimized is of the form $x^T \cdot Q \cdot x$.
        the problem is unconstrained by definition, thus, constraints are set to
        None. Variables are binary and their number must match the dimension of
        the Q matrix.

        Parameters
        ----------
        q: squared Q matrix defining the QUBO problem over a binary
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
        if list(q.coefficients.keys()) != [2]:
            raise ValueError("Cost must be a quadratic " "matrix.")
        self._b_variables = DiscreteVariables(domains=[2] * value.shape[0])
        self._q_cost = q

    @property
    def constraints(self):
        """As an unconstrained problem, QUBO constraints are None."""
        return None

    def validate_input(self, q):
        """Validate the cost coefficient is a square matrix.

        Parameters
        ----------
        q: Quadratic coefficient of the cost function.
        """
        m, n = q.shape
        if m != n:
            raise ValueError("q matrix is not a square matrix.")

    def verify_solution(self, solution):
        raise NotImplementedError


DType = ty.Union[ty.List[int], ty.List[ty.Tuple]]
CType = ty.List[ty.Tuple[int, int, npt.ArrayLike]]


class CSP(OptimizationProblem):
    """A constraint satisfaction problem.

     The CSP is in usually represented by the tuple (variables, domains,
     constraints). However, because every variable must have a domain, the
     user only provides the domains and constraints.

    Parameters
    ----------
    domains: either a list of tuples with values that each variable can take or
    a list of integers specifying the domain size for each variable.

    constraints: Discrete constraints defining mutually allowed values
    between variables. Has to be a list of n-tuples where the first n-1 elements
    are the variables related by the n-th element of the tuple. The n-th element
    is a tensor indicating what values of the variables are simultaneously
    allowed.
    """

    def __init__(self, domains: DType = None, constraints: CType = None):
        super().__init__()
        self._variables.discrete = DiscreteVariables(domains)
        self._constant_cost = Cost(0)
        self._constraints.discrete = DiscreteConstraints(constraints)

    @property
    def variables(self):
        """Discrete variables over which the problem is specified."""
        return self._variables.discrete

    @property
    def cost(self):
        """Constant cost function, CSPs require feasibility not minimization."""
        return self._constant_cost

    @property
    def constraints(self):
        """Specification of mutually allowed values between variables."""
        return self._constraints.discrete

    @constraints.setter
    def constraints(self, value):
        self._constraints.discrete = DiscreteConstraints(value)

    def verify_solution(self, solution):
        raise NotImplementedError


class QP(OptimizationProblem):
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
            hessian: npt.ArrayLike,
            linear_offset: ty.Optional[np.ndarray] = None,
            constraint_hyperplanes: ty.Optional[np.ndarray] = None,
            constraint_biases: ty.Optional[np.ndarray] = None,
            constraint_hyperplanes_eq: ty.Optional[np.ndarray] = None,
            constraint_biases_eq: ty.Optional[np.ndarray] = None, ):
        super().__init__()
        self.c_variables = ContinuousVariables(hessian.shape[0])
        self.q_cost = Cost(linear_offset, hessian)
        self._constraints.arithmetic = ArithmeticConstraints(
            ineq=[constraint_biases,
                  constraint_hyperplanes],
            eq=[constraint_biases_eq,
                constraint_hyperplanes_eq])

    @property
    def variables(self):
        return self.c_variables

    @property
    def cost(self):
        return self.q_cost

    @property
    def constraints(self):
        return self._constraints.arithmetic


    @property
    def get_hessian(self) -> np.ndarray:
        return self.cost.get_coefficient(order=2)

    @property
    def get_linear_offset(self) -> np.ndarray:
        return self.cost.get_coefficient(order=1)

    @property
    def get_constraint_hyperplanes(self) -> np.ndarray:
        return self.constraints.inequality.get_coefficient(order=2)

    @property
    def get_constraint_biases(self) -> np.ndarray:
        return self.constraints.inequality.get_coefficient(order=1)

    @property
    def num_variables(self) -> int:
        return self.variables.num_variables


class LP(OptimizationProblem):
    def __init__(self, c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
                 bounds=None, x0=None):
        """Example adopting the interface from scipy.optimize.linprog.
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html

        minimize c @ x
        such that:
        A_ub @ x <= b_ub
        A_eq @ x == b_eq
        lb <= x <= ub
        """
        super().__init__()
        self.c_variables = ContinuousVariables(bounds=bounds)
        self.l_cost = Cost(c)
        self.a_constraints = ArithmeticConstraints(ineq=[b_ub, A_ub],
                                                   eq=[b_eq, A_eq])

    @property
    def variables(self):
        return self.c_variables

    @property
    def cost(self):
        return self.l_cost

    @property
    def constraints(self):
        return self.a_constraints
