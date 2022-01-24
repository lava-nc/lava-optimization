# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty
import numpy.typing as npt

class DiscreteConstraints:
    """Constraint involving only discrete variables.

    :param relations: a binary tensor indicating what values of the variables
    related by this constraint are allowed.
    """

    def __init__(self, constraints: ty.List[ty.Tuple[int, int, npt.ArrayLike]]):
        self._constraints = constraints
        self.set_relations_var_subsets(self._constraints)


    def set_relations_var_subsets(self, constraints):
        var_subsets, relations = self.get_scope_and_relations(constraints)
        self.validate_subsets_and_relations_match(var_subsets, relations)
        self._var_subset = var_subsets
        self._relations = relations

    @property
    def relations(self):
        return self._relations

    @property
    def var_subsets(self):
        return self._var_subset

    @property
    def constraints(self):
        return self._constraints

    @constraints.setter
    def constraints(self, value):
        self.set_relations_var_subsets(value)
        self._constraints = value

    def get_scope_and_relations(self, constraints):
        var_subsets = [tuple(c[:-1]) for c in constraints]
        relations = [c[-1] for c in constraints]
        return var_subsets, relations

    def validate_subsets_and_relations_match(self, subsets, relations):
        for subset, relation in zip(subsets, relations):
            assert len(subset) ==  relation.ndim