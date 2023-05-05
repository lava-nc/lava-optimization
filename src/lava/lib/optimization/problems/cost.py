# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import numpy as np
import typing as ty

import numpy.typing as npt
from lava.lib.optimization.problems.coefficients import CoefficientTensorsMixin

CTType = ty.Union[ty.List, npt.ArrayLike]


class Cost(CoefficientTensorsMixin):
    """Cost function of an optimization problem.

    :param coefficients: cost tensor coefficients.
    :param augmented_terms: Tuple of terms not originally defined in the cost
    function, e.g. a regularization term or  those incorporating constraints
    into the cost. Tuple elements have the same type as coefficients.
    """

    def __init__(
        self,
        *coefficients: CTType,
        augmented_terms: ty.Tuple[CTType, ...] = None,
    ):
        super().__init__(*coefficients)
        self._augmented_terms = augmented_terms

    @property
    def augmented_terms(self):
        """Augmented terms present in the cost function."""
        if self._augmented_terms is None:
            return None
        else:
            at = CoefficientTensorsMixin(*self._augmented_terms)
            return at.coefficients

    @augmented_terms.setter
    def augmented_terms(self, value):
        self._augmented_terms = value

    @property
    def is_augmented(self):
        """Whether augmented terms are present in the cost function."""
        if self.augmented_terms is None:
            return False
        return True

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the cost at the given solution."""
        return super().evaluate(x)
