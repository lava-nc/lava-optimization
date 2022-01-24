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
    """An equality constraint.

    :param coefficients: cost tensor coefficients.
    """

    def __init__(self, *coefficients: CTType):
        super().__init__(*coefficients)


class InequalityConstraints(CoefficientTensorsMixin):
    """An inequality constraint.

    :param coefficients: cost tensor coefficients.
    """

    def __init__(self, *coefficients: CTType):
        super().__init__(*coefficients)
