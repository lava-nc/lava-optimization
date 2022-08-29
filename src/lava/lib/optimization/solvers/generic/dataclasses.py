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
