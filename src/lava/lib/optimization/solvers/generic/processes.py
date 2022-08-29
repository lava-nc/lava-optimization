# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty
from dataclasses import dataclass

import numpy as np
from lava.magma.core.decorator import implements
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.process.interfaces import AbstractProcessMember
from lava.magma.core.process.ports.ports import InPort, OutPort, RefPort
from lava.magma.core.process.process import AbstractProcess, LogConfig
from lava.magma.core.process.variable import Var
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

from lava.lib.optimization.problems.coefficients import CoefficientTensorsMixin
from lava.lib.optimization.problems.problems import OptimizationProblem
from lava.lib.optimization.problems.variables import (
    DiscreteVariables, ContinuousVariables)


def _vars_from_coefficients(coefficients: CoefficientTensorsMixin) -> \
        ty.Dict[int, AbstractProcessMember]:
    vars = dict()
    for rank, coeff in coefficients.items():
        if rank == 1:
            init = -coeff
        if rank == 2:
            linear_component = -coeff.diagonal()
            quadratic_component = coeff * np.logical_not(np.eye(*coeff.shape))
            if 1 in vars.keys():
                vars[1].init = vars[1].init + linear_component
            else:
                vars[1] = Var(shape=linear_component.shape,
                              init=linear_component)
            init = -quadratic_component
        vars[rank] = Var(shape=coeff.shape, init=init)
    return vars


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

        def constructor(self,
                        name: ty.Optional[str] = None,
                        log_config: ty.Optional[LogConfig] = None) -> None:
            super(type(self), self).__init__(name=name,
                                             log_config=log_config)
            self.problem = problem
            if not hasattr(problem, 'variables'):
                raise Exception("An optimization problem must contain "
                                "variables.")
            if hasattr(problem.variables, 'continuous') or isinstance(
                    problem.variables, ContinuousVariables):
                self.continuous_variables = Var(
                    shape=(problem.variables.continuous.num_vars, 2))
            if hasattr(problem.variables, 'discrete') or isinstance(
                    problem.variables, DiscreteVariables):
                self.discrete_variables = Var(
                    shape=(problem.variables.num_variables,
                           # problem.variables.domain_sizes[0]
                           ))
            if hasattr(problem, 'cost'):
                self.cost_coefficients = _vars_from_coefficients(
                    problem.cost.coefficients)

            self.variable_assignment = Var(
                shape=(problem.variables.num_variables,))
            self.optimality = Var(shape=(1,))
            self.feasibility = Var(shape=(1,))

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

    def __init__(self, shape,
                 importances: ty.Optional[
                     ty.Union[int, list, np.ndarray]] = None,
                 name: ty.Optional[str] = None,
                 log_config: ty.Optional[LogConfig] = None) -> None:
        super().__init__(shape=shape,
                         name=name,
                         log_config=log_config)
        self.num_variables = shape[0]
        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)
        self.variable_assignment = Var(shape=shape)
        self._importances = importances
        self.satisfiability = OutPort(shape=shape)

    @property
    def importances(self):
        return self._importances

    @importances.setter
    def importances(self, value):
        self._importances = value


@dataclass()
class VariablesProcesses:
    """Processes implementing the variables."""
    continuous: ContinuousVariablesProcess = None
    discrete: DiscreteVariablesProcess = None


class StochasticIntegrateAndFire(AbstractProcess):
    def __init__(self, *, shape: ty.Tuple[int, ...] = (1,),
                 initial_value: ty.Union[int, list, np.ndarray] = 0,
                 increment: ty.Union[int, list, np.ndarray] = 100,
                 noise_amplitude: ty.Union[int, list, np.ndarray] = 0,
                 input_duration: ty.Union[int, list, np.ndarray] = 6,
                 min_state: ty.Union[int, list, np.ndarray] = 1000,
                 min_integration: ty.Union[int, list, np.ndarray] = -1000,
                 steps_to_fire: ty.Union[int, list, np.ndarray] = 10,
                 refractory_period: ty.Union[int, list, np.ndarray] = 1,
                 name: ty.Optional[str] = None,
                 log_config: ty.Optional[LogConfig] = None) -> None:
        super().__init__(shape=shape,
                         initial_state=initial_value,
                         noise_amplitude=noise_amplitude,
                         input_duration=input_duration,
                         min_state=min_state,
                         min_integration=min_integration,
                         steps_to_fire=steps_to_fire,
                         refractory_period=refractory_period,
                         name=name,
                         log_config=log_config)
        self.added_input = InPort(shape=shape)
        self.replace_assignment = InPort(shape=shape)
        self.messages = OutPort(shape=shape)
        self.satisfiability = OutPort(shape=shape)

        self.integration = Var(shape=shape, init=0)
        self.increment = Var(shape=shape, init=increment)
        self.state = Var(shape=shape, init=initial_value)
        self.noise_amplitude = Var(shape=shape, init=noise_amplitude)
        self.input_duration = Var(shape=shape, init=input_duration)
        self.min_state = Var(shape=shape, init=min_state)
        self.min_integration = Var(shape=shape, init=min_integration)
        self.steps_to_fire = Var(shape=shape, init=steps_to_fire)
        self.refractory_period = Var(shape=shape, init=refractory_period)
        self.prev_firing = Var(shape=shape, init=False)
        self.assignment = Var(shape=shape, init=False)
        self.satisfiability_var = Var(shape=shape, init=False)
        self.assignemnt_buffer = Var(shape=shape, init=False)
        self.min_cost = Var(shape=shape, init=False)


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

    def __init__(self,
                 shape,
                 name: ty.Optional[str] = None,
                 log_config: ty.Optional[LogConfig] = None) -> None:
        super().__init__(shape=shape,
                         name=name,
                         log_config=log_config)
        self.shape = shape
        self.cost = Var(shape=(1,))
        self.s_in = InPort(shape=shape)
        self.s_out = OutPort(shape=(1,))


class SatConvergenceChecker(AbstractProcess):
    """Process that continuously monitors satisfiability convergence."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        raise NotImplementedError


class ReadGate(AbstractProcess):
    """Process that triggers solution readout when problem is solved."""

    def __init__(self,
                 name: ty.Optional[str] = None,
                 log_config: ty.Optional[LogConfig] = None) -> None:
        super().__init__(name=name,
                         log_config=log_config)
        self.solved = Var(shape=(1,))
        self.in_port = InPort(shape=(1,))
        self.out_port = OutPort(shape=(1,))


class SolutionReadout(AbstractProcess):
    """Process to readout solution from SNN and make it available on host."""

    def __init__(self, shape,
                 name: ty.Optional[str] = None,
                 log_config: ty.Optional[LogConfig] = None) -> None:
        super().__init__(shape=shape,
                         name=name,
                         log_config=log_config)
        self.solution = Var(shape=shape, init=-1)
        self.in_port = InPort(shape=(1,))
        self.ref_port = RefPort(shape=shape)


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


class CostIntegrator(AbstractProcess):
    def __init__(self, *, shape: ty.Tuple[int, ...] = (1,),
                 name: ty.Optional[str] = None,
                 log_config: ty.Optional[LogConfig] = None) -> None:
        super().__init__(shape=shape,
                         name=name,
                         log_config=log_config)
        self.cost_components = InPort(shape=shape)
        self.cost_out = OutPort(shape=shape)

        self.min_cost = Var(shape=shape, init=2 ** 32)
