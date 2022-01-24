# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty

import numpy.typing as npt
from src.lava.lib.optimization.problems.coefficients import \
    CoefficientTensorsMixin

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

    def __init__(self, constraints: ty.List[ty.Tuple[int, int, npt.ArrayLike]]):
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
            assert len(subset) == relation.ndim


class EqualityConstraints(CoefficientTensorsMixin):
    """List of equality constraints defined by tensor coefficients.

    We consider generalized constraints of arbitrary degree:

    .. math::
        h(x) = 0
    where the terms of $h(x)$ have the form:
    .. math::
         g(x)= \sum_{ijk...} \epsilon_{ijk...} \cdot x_i \cdot x_j
        \cdot x_k \cdot ...

    Parameters
    ----------
    coefficients: tensor coefficients defining the constraints.
    """

    def __init__(self, *coefficients: CTType):
        super().__init__(*coefficients)


class InequalityConstraints(CoefficientTensorsMixin):
    """List of inequality constraints defined by tensor coefficients.

    We consider generalized constraints of arbitrary degree:

    .. math::
        g(x) \leq 0
    where the terms of $g(x)$ have the form:
    .. math::
         \sum_{ijk...} \epsilon_{ijk...} \cdot x_i \cdot x_j \cdot x_k \cdot
        ...

    Parameters
    ----------
    coefficients: tensor coefficients defining the constraints.
    """

    def __init__(self, *coefficients: CTType):
        super().__init__(*coefficients)


class ArithmeticConstraints():
    def __init__(self,
                 eq: CTType = None,
                 ineq: CTType = None
                 ):
        """Constraints defined via an arithmetic relation between tensors."""
        self._equality = None if eq is None else EqualityConstraints(*eq)
        self._inequality = None if ineq is None else InequalityConstraints(
            *ineq)

    @property
    def equality(self):
        return self._equality

    @equality.setter
    def equality(self, value):
        if value is None:
            self._equality = None
        else:
            self._equality = EqualityConstraints(*value)

    @property
    def inequality(self):
        return self._inequality

    @inequality.setter
    def inequality(self, value):
        if value is None:
            self._inequality = None
        else:
            self._inequality = InequalityConstraints(*value)
