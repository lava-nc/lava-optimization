# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
from dataclasses import dataclass

from lava.proc.dense.process import Dense

from lava.lib.optimization.solvers.generic.processes import (
    ContinuousConstraintsProcess,
    DiscreteConstraintsProcess,
    MixedConstraintsProcess,
    CostConvergenceChecker,
    SatConvergenceChecker,
    ReadGate,
    SolutionReadout,
    ContinuousVariablesProcess,
    DiscreteVariablesProcess,
    AugmentedTermsProcess,
)


@dataclass
class CostMinimizer:
    """Processes implementing the cost function"""

    coefficients_2nd_order: Dense

    @property
    def state_in(self):
        return self.coefficients_2nd_order.s_in

    @property
    def gradient_out(self):
        return self.coefficients_2nd_order.a_out


@dataclass
class ConstraintEnforcing:
    """Processes implementing the constraints and their enforcing."""

    continuous: ContinuousConstraintsProcess
    discrete: DiscreteConstraintsProcess
    mixed: MixedConstraintsProcess


@dataclass()
class VariablesProcesses:
    """Processes implementing the variables."""

    continuous: ContinuousVariablesProcess = None
    discrete: DiscreteVariablesProcess = None

    @property
    def gradient_in(self):
        return self.discrete.a_in

    @property
    def state_out(self):
        return self.discrete.s_out

    @property
    def importances(self):
        return self.discrete.importances

    @importances.setter
    def importances(self, value):
        self.discrete.importances = value

    @property
    def replace_assignment(self):
        return self.discrete.replace_assignment

    @property
    def local_cost(self):
        return self.discrete.local_cost

    @property
    def variables_assignment(self):
        return self.discrete.variable_assignment


@dataclass
class ProximalGradientMinimizer:
    augmented_terms: AugmentedTermsProcess


@dataclass
class MacroStateReader:
    """Processes for checking convergence and reading network state encoding
    the solution ."""

    read_gate: ReadGate
    solution_readout: SolutionReadout
    cost_convergence_check: CostConvergenceChecker = None
    sat_convergence_check: SatConvergenceChecker = None

    @property
    def cost_in(self):
        return self.cost_convergence_check.cost_components

    @property
    def update_buffer(self):
        return self.cost_convergence_check.update_buffer

    @property
    def ref_port(self):
        return self.solution_readout.ref_port

    @property
    def min_cost(self):
        return self.cost_convergence_check.min_cost

    @property
    def satisfaction(self):
        return self.sat_convergence_check.satisfaction

    @property
    def solution(self):
        return self.solution_readout.solution

    @property
    def read_gate_in_port(self):
        return self.read_gate.new_solution

    @property
    def read_gate_do_readout(self):
        return self.read_gate.do_readout
