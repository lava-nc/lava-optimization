# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
from dataclasses import dataclass

from lava.lib.optimization.solvers.generic.hierarchical_processes import (
    AugmentedTermsProcess,
    ContinuousConstraintsProcess,
    ContinuousVariablesProcess,
    DiscreteConstraintsProcess,
    DiscreteVariablesProcess,
    MixedConstraintsProcess,
)
from lava.proc.dense.process import Dense
from lava.proc.sparse.process import Sparse


@dataclass
class CostMinimizer:
    """Processes implementing an optimization problem's cost function."""

    coefficients_2nd_order: Sparse

    @property
    def state_in(self):
        """Port receiving input from dynamical systems representing
        variables."""
        return self.coefficients_2nd_order.s_in

    @property
    def gradient_out(self):
        """Port sending gradient descent components to the dynamical systems."""
        return self.coefficients_2nd_order.a_out


@dataclass
class ConstraintEnforcing:
    """Processes implementing an optimization problem's constraints and their
    enforcing."""

    continuous: ContinuousConstraintsProcess = None
    discrete: DiscreteConstraintsProcess = None
    mixed: MixedConstraintsProcess = None

    @property
    def state_in(self):
        return self.continuous.a_in

    @property
    def state_out(self):
        return self.continuous.s_out

    @property
    def variables_assignment(self):
        return self.continuous.constraint_assignment


@dataclass()
class VariablesImplementation:
    """Processes implementing the variables of an optimization problem."""

    continuous: ContinuousVariablesProcess = None
    discrete: DiscreteVariablesProcess = None

    @property
    def gradient_in(self):
        return self.discrete.a_in

    @property
    def gradient_in_cont(self):
        return self.continuous.a_in

    @property
    def state_out(self):
        return self.discrete.s_out

    @property
    def state_out_cont(self):
        return self.continuous.s_out

    @property
    def importances(self):
        return self.discrete.cost_diagonal

    @importances.setter
    def importances(self, value):
        self.discrete.cost_diagonal = value

    @property
    def local_cost(self):
        return self.discrete.local_cost

    @property
    def variables_assignment(self):
        return self.discrete.variable_assignment

    @property
    def variables_assignment_cont(self):
        return self.continuous.variable_assignment


@dataclass
class ProximalGradientMinimizer:
    augmented_terms: AugmentedTermsProcess
