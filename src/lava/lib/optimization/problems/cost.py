# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/


class Cost:
    """Cost function of an optimization problem.

    :param linear_term: 1-D tensor, the linear term of the cost function.
    :param quadratic_term: 2-D tensor, the quadratic term of the cost function.
    :param augmented_terms: Tuple of terms not originally defined in the cost
    function, e.g. a regularization term or  those incorporating constraints
    into the cost.
    """

    def __init__(self, **kwargs):
        self._cost_terms = kwargs

    @property
    def is_linear(self):
        """Whether the cost function only include linear terms."""
        if self.quadratic_term is None:
            return True
        return False

    @property
    def is_augmented(self):
        """Whether augmented terms are present in the cost function."""
        if self.augmented_terms is None:
            return False
        return True

    @property
    def augmented_terms(self):
        """Augmented terms present in the cost function."""
        self._cost_terms.get("augmented_terms", None)
        return None

    @property
    def linear_term(self):
        """Linear term of the cost function in a optimization problem."""
        return self._cost_terms.get("linear_term", None)

    @property
    def quadratic_term(self):
        """Quadratic term of the cost function in a optimization problem."""
        return self._cost_terms.get("quadratic_term", None)
