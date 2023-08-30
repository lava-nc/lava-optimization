# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty

import numpy.typing as npt
from lava.lib.optimization.problems.coefficients import CoefficientTensorsMixin

CTType = ty.Union[ty.List, npt.ArrayLike]


class DiscreteConstraints:
    """Set of constraints involving only discrete variables.

    Parameters
    ----------
    constraints: List of constraints each as an n-tuple where the
    first n-1 elements are the variables related by the n-th element of the
    tuple. The n-th element is a tensor indicating what values of the variables
    are simultaneously allowed.

    Although initially intended for tensors of rank=2 (binary variables),
    other ranks simply mean relations between any number of variables and
    thus are allowed. In this way, tensor's rank corresponds to the Arity of the
    constraint they define.
    """

    def __init__(
        self, constraints: ty.List[ty.Tuple[int, int, npt.ArrayLike]]
    ):
        self._constraints = constraints
        self.set_relations_var_subsets(self._constraints)

    def set_relations_var_subsets(self, constraints):
        """Set relations and variable subsets from constraints.

        Parameters
        ----------
        constraints:  List of constraints each as an n-tuple where the first
        n-1 elements are the variables related by the n-th element of the
        tuple (a tensor). See class docstring for more details.
        """
        var_subsets, relations = self.get_scope_and_relations(constraints)
        self.validate_subsets_and_relations_match(var_subsets, relations)
        self._var_subset = var_subsets
        self._relations = relations

    @property
    def relations(self):
        """List of tensors specifying discrete constraint over var subsets."""
        return self._relations

    @property
    def var_subsets(self):
        """List of variable subsets affected by the corresponding relation."""
        return self._var_subset

    @property
    def constraints(self):
        """User specified constraints."""
        return self._constraints

    @constraints.setter
    def constraints(self, value):
        """Set constraints as well as relations and variable subsets properties.

        Parameters
        ----------
        constraints:  List of constraints each as an n-tuple where the first
        n-1 elements are the variables related by the n-th element of the
        tuple (a tensor). See class docstring for more details.
        """
        self.set_relations_var_subsets(value)
        self._constraints = value

    def get_scope_and_relations(self, constraints):
        """Extract relations and variable subsets from constraints.

        Parameters
        ----------
        constraints:  List of constraints each as an n-tuple where the first
        n-1 elements are the variables related by the n-th element of the
        tuple (a tensor). See class docstring for more details.
        """
        var_subsets = [tuple(c[:-1]) for c in constraints]
        relations = [c[-1] for c in constraints]
        return var_subsets, relations

    def validate_subsets_and_relations_match(self, subsets, relations):
        """Verify relation size match domain sizes of affected variables.

        Parameters
        ----------
        subsets: List of variable subsets affected by the corresponding
        relation.
        relations: List of tensors specifying discrete constraint over var
        subsets.
        """
        for subset, relation in zip(subsets, relations):
            if len(subset) != relation.ndim:
                raise ValueError(
                    "Relation and variable subset dimensions "
                    "don't match for this constraint!"
                )


class EqualityConstraints(CoefficientTensorsMixin):
    r"""List of equality constraints defined by tensor coefficients.

    We consider generalized constraints of arbitrary degree:

    .. math::
        h(x) = 0

    where the terms of :math:`h(x)` have the form:

    .. math::
         g(x)= \sum_{ijk...} \epsilon_{ijk...} \cdot x_i \cdot x_j
        \cdot x_k \cdot ...

    Parameters
    ----------
    coefficients: tensor
        coefficients defining the constraints.

    """

    def __init__(self, *coefficients: CTType):
        super().__init__(*coefficients)


class InequalityConstraints(CoefficientTensorsMixin):
    r"""List of inequality constraints defined by tensor coefficients.

    We consider generalized constraints of arbitrary degree:

    .. math::
        g(x) \leq 0

    where the terms of :math:`g(x)` have the form:

    .. math::
         \sum_{ijk...} \epsilon_{ijk...} \cdot x_i \cdot x_j \cdot x_k
         \cdot ...

    Parameters
    ----------

    coefficients: tensor
        coefficients defining the constraints.

    """

    def __init__(self, *coefficients: CTType):
        super().__init__(*coefficients)


class ArithmeticConstraints:
    def __init__(self, eq: CTType = None, ineq: CTType = None):
        r"""Constraints defined via an arithmetic relation between tensors.

        These class includes two types of arithmetic constraints, inequality
        $g(x) \leq 0$ and equality $h(x) = 0$ constraints.

        Parameters
        ----------
        eq: tuple of tensor coefficients defining the equality constraints.
        ineq: tuple of tensor coefficients defining the inequality constraints.
        """
        self._equality = None if eq is None else EqualityConstraints(*eq)
        self._inequality = (
            None if ineq is None else InequalityConstraints(*ineq)
        )

    @property
    def equality(self):
        """EqualityConstraints object defined by tensor coefficients."""
        return self._equality

    @equality.setter
    def equality(self, value):
        """Set EqualityConstraints object from tensor coefficients.

        Parameters
        ----------
        value: tensor coefficients defining the constraints.
        """
        if value is None:
            self._equality = None
        else:
            self._equality = EqualityConstraints(*value)

    @property
    def inequality(self):
        """InequalityConstraints object defined by tensor coefficients."""
        return self._inequality

    @inequality.setter
    def inequality(self, value):
        """Set InequalityConstraints object from tensor coefficients.

        Parameters
        ----------
        value: tensor coefficients defining the constraints."""
        if value is None:
            self._inequality = None
        else:
            self._inequality = InequalityConstraints(*value)


class Constraints:
    """A set of constraints, including both discrete and arithmetic.

    Discrete constraints can be of any arity and are defined by a tuple
    containing variable subsets and a relation tensor. Arithmetic constraints
    include equality and inequality constraints and are defined by a series
    of tensors defining the coefficients of scalar function of the variable
    vectors.
    """

    def __init__(self):
        self._discrete = None
        self._arithmetic = None

    @property
    def arithmetic(self):
        """Constraints defined via an arithmetic relation."""
        return self._arithmetic

    @arithmetic.setter
    def arithmetic(self, value: ArithmeticConstraints):
        self._arithmetic = value

    @property
    def discrete(self):
        """Constraints over discrete variables only, defined via a relation."""
        return self._discrete

    @discrete.setter
    def discrete(self, value: DiscreteConstraints):
        self._discrete = value
