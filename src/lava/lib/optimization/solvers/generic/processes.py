# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty
from dataclasses import dataclass

from lava.magma.core.decorator import implements
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.process.interfaces import AbstractProcessMember
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

from lava.lib.optimization.problems.coefficients import CoefficientTensorsMixin
from lava.lib.optimization.problems.problems import OptimizationProblem


def _in_ports_from_coefficients(coefficients: CoefficientTensorsMixin) -> \
        ty.List[AbstractProcessMember]:
    in_ports = [InPort(shape=coeff.shape) for coeff in
                coefficients.coefficients]
    return in_ports


class SolverProcessBuilder:
    def __init__(self):
        self._constructor = None

    def create_constructor(self, problem):
        """Create constructor for dynamically created
        OptimizationSolverProcess class."""

        def constructor(self):
            self.continuous_variables = InPort(
                shape=(problem.variables.continuous.num_vars, 2))
            self.discrete_variables = InPort(
                shape=(problem.variables.discrete.num_vars,))
            self.cost_coefficients = _in_ports_from_coefficients(
                problem.cost.coefficients)
            self.cost_augmented_terms = _in_ports_from_coefficients(
                problem.cost.augmented_terms)
            self.ineq_constraints = _in_ports_from_coefficients(
                problem.constraints.inequality.coefficients)
            self.eq_constraints = _in_ports_from_coefficients(
                problem.constraints.equality.coefficients)
            if problem.discrete.constraints is not None:
                d_constraints = problem.constraints.discrete
                num_dconstraints = len(d_constraints.var_subsets)
                max_arity = len(d_constraints.var_subsets[0])
                self.discrete_constraints_relations = InPort(
                    shape=(len(d_constraints.relations), max_arity))
                subsets_shape = (num_dconstraints,) + (
                    problem.variables.discrete.num_vars,) * max_arity
                self.discrete_constraints_subsets = InPort(shape=subsets_shape)
            self.variable_assignment = Var(shape=(problem.num_vars,))
            self.optimality = Var(shape=(1,))
            self.feasibility = Var(shape=(1,))
            self.solution = OutPort(shape=(problem.num_vars,))

        self._constructor = constructor

    @property
    def solver_process(self) -> AbstractProcess:
        """Hierarchical process implementing an optimization solver."""
        SolverProcess = type('OptimizationSolverProcess',
                             (AbstractProcess,),
                             {'__init__': self._constructor})
        return SolverProcess()


class ContinuousVariablesProcess(AbstractProcess):
    """Process implementing continuous variables, each corresponding to the
    state of a neuron."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        raise NotImplementedError


class DiscreteVariablesProcess(AbstractProcess):
    """Process implementing discrete variables as a set of winner-takes-all
    populations."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        raise NotImplementedError


@dataclass()
class Variables:
    """Processes implementing the variables."""
    continuous: ContinuousVariablesProcess
    discrete: DiscreteVariablesProcess


class CoefficientsProcess(AbstractProcess):
    """Process implementing cost coefficients as synapses."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        raise NotImplementedError


class AugmentedTermsProcess(AbstractProcess):
    """Process implementing cost coefficients as synapses."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        raise NotImplementedError


@dataclass
class CostMinimizer:
    """Processes implementing the cost function"""
    coefficients: CoefficientsProcess


@dataclass
class ProximalGradientMinimizer:
    augmented_terms: AugmentedTermsProcess


class ContinuousConstraintsProcess(AbstractProcess):
    """Process implementing continuous constraints via neurons and synapses."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        raise NotImplementedError


class DiscreteConstraintsProcess(AbstractProcess):
    """Process implementing discrete constraints via synapses."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        raise NotImplementedError


class MixedConstraintsProcess(AbstractProcess):
    """Process implementing continuous constraints via neurons and synapses."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        raise NotImplementedError


@dataclass
class ConstraintEnforcing:
    """Processes implementing the constraints and their enforcing."""
    continuous: ContinuousConstraintsProcess
    discrete: DiscreteConstraintsProcess
    mixed: MixedConstraintsProcess


class CostConvergenceChecker(AbstractProcess):
    """Process that continuously monitors cost convergence."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        raise NotImplementedError


class SatConvergenceChecker(AbstractProcess):
    """Process that continuously monitors satisfiability convergence."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        raise NotImplementedError


class ReadGate(AbstractProcess):
    """Process that triggers solution readout when problem is solved."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        raise NotImplementedError


class SolutionReadout(AbstractProcess):
    """Process to readout solution from SNN and make it available on host."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        raise NotImplementedError


@dataclass
class MacroStateReader:
    """Processes for checking convergence and reading network state encoding
    the solution ."""
    cost_convergence_check: CostConvergenceChecker
    sat_convergence_check: SatConvergenceChecker
    read_gate: ReadGate
    solution_readout: SolutionReadout


class Monitor(AbstractProcess):
    """Process which reads solution upon Readout process notification."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        raise NotImplementedError


class Readout(AbstractProcess):
    """Monitor Integrator spikes to identify when a solution is found."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        raise NotImplementedError


class Integrator(AbstractProcess):
    """Integrate satisfiability signal from SolverNet, spike if solution found.

    Integration and weights are calibrated to reach spiking threshold when
    optimal and feasible variable assignment is found."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        raise NotImplementedError


class SolverNet(AbstractProcess):
    """Network that represents a problem and whose dynamics solves it."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class AdjacencyMatrixFactory:
    """Creates an SNN connectivity matrix from a given OptimizationProblem."""

    def from_problem(self, problem: OptimizationProblem):
        raise NotImplementedError


class OptimizationSolverProcess(AbstractProcess):
    pass


@implements(proc=OptimizationSolverProcess, protocol=LoihiProtocol)
class OptimizationSolverModel(AbstractSubProcessModel):
    """Implements OptimizationSolver from processes implemented elsewhere."""

    def __init__(self, proc):
        self.variables = Variables(ContinuousVariablesProcess(),
                                   DiscreteVariablesProcess())
        self.cost_minimizer = CostMinimizer(CoefficientsProcess())
        self.proximal_gradient = ProximalGradientMinimizer(
            AugmentedTermsProcess())
        self.constraint_enforcing = ConstraintEnforcing(
            ContinuousConstraintsProcess(),
            DiscreteConstraintsProcess(),
            MixedConstraintsProcess())
        self.macrostate_reader = MacroStateReader(CostConvergenceChecker(),
                                                  SatConvergenceChecker(),
                                                  ReadGate(),
                                                  SolutionReadout())
        for var in proc.vars:
            var.alias(getattr(self, self._var_to_proc(var)).vars.var)

    def _var_to_proc(self, var):
        raise NotImplementedError
