# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/


class OptimizationProblem:
    """Abstract optimization problem from which the actual problems derive.
    """

    def __init__(self, cost=None, constraints=None):
        self.cost = cost
        self.constraints = constraints
