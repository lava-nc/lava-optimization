# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/


class AbstractConstraint:
    """Abstract constraint class from which the actual constraints derive.

    :param var_subset: List of indexes of the set of variables affected by this
    constraint.
    """

    def __init__(self, **kwargs):
        self.var_subset = kwargs.get("var_subset", None)
        self._is_discrete = None

    @property
    def is_linear(self):
        """Whether the constraint only include linear terms."""
        return None

    @property
    def is_discrete(self):
        """Whether the variables are all discrete."""
        return self._is_discrete

    @is_discrete.setter
    def is_discrete(self, value):
        self._is_discrete = value


class DiscreteConstraint(AbstractConstraint):
    """Constraint involving only discrete variables.

    :param relation: a binary tensor indicating what values of the variables
    related by this constraint are allowed.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_discrete = True
        self.relation = kwargs.get("relation", None)


class ContinuousConstraint(AbstractConstraint):
    """Constraint involving only continuous variables."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._constraint_terms = kwargs

    @property
    def linear_term(self):
        """Linear term of this constraint if any."""
        return self._constraint_terms.get("linear_term", None)

    @property
    def quadratic_term(self):
        """Quadratic term of this constraint if any."""
        return self._constraint_terms.get("quadratic_term", None)


class MixedConstraint(ContinuousConstraint):
    """A constraint affecting both discrete and continuous variables."""

    pass


class EqualityConstraint(ContinuousConstraint):
    """An equality constraint."""

    pass


class InequalityConstraint(ContinuousConstraint):
    """An inequality constraint."""

    pass
