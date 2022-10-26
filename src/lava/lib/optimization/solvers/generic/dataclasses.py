# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
from dataclasses import dataclass

from lava.lib.optimization.solvers.generic.hierarchical_processes import (
    AugmentedTermsProcess, ContinuousConstraintsProcess,
    ContinuousVariablesProcess, CostConvergenceChecker,
    DiscreteConstraintsProcess, DiscreteVariablesProcess,
    MixedConstraintsProcess, SatConvergenceChecker)
from lava.lib.optimization.solvers.generic.monitoring_processes\
    .solution_readout.process import SolutionReadout
from lava.proc.dense.process import Dense
from lava.proc.read_gate.process import ReadGate


@dataclass
class CostMinimizer:
    """Processes implementing an optimization problem's cost function."""

    coefficients_2nd_order: Dense

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

    continuous: ContinuousConstraintsProcess
    discrete: DiscreteConstraintsProcess
    mixed: MixedConstraintsProcess


@dataclass()
class VariablesImplementation:
    """Processes implementing the variables of an optimization problem."""

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
        return self.discrete.step_size

    @importances.setter
    def importances(self, value):
        self.discrete.step_size = value

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
    def solution_step(self):
        return self.solution_readout.solution_step

    @property
    def cost_in(self):
        return self.cost_convergence_check.cost_components

    @property
    def update_buffer(self):
        return self.cost_convergence_check.update_buffer

    @property
    def ref_port(self):
        return self.read_gate.solution_reader

    @property
    def min_cost(self):
        return self.solution_readout.min_cost

    @property
    def satisfaction(self):
        return self.sat_convergence_check.satisfaction

    @property
    def solution(self):
        return self.solution_readout.solution

    @property
    def read_gate_in_port(self):
        return self.read_gate.cost_in

    @property
    def read_gate_cost_out(self):
        return self.read_gate.cost_out

    @property
    def read_gate_req_stop(self):
        return self.read_gate.send_pause_request

    @property
    def read_gate_solution_out(self):
        return self.read_gate.solution_out

    @property
    def read_gate_ack(self):
        return self.read_gate.acknowledgemet

    @property
    def solution_readout_solution_in(self):
        return self.solution_readout.read_solution

    @property
    def solution_readout_cost_in(self):
        return self.solution_readout.cost_in

    @property
    def solution_readout_req_stop_in(self):
        return self.solution_readout.req_stop_in

    @property
    def solution_readout_ack(self):
        return self.solution_readout.acknowledgemet
