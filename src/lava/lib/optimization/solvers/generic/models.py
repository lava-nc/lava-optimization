# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/


from dataclasses import dataclass

import numpy as np
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort, PyRefPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.dense.process import Dense

from lava.lib.optimization.solvers.generic.processes \
    import CostConvergenceChecker, ReadGate, SolutionReadout, \
    SatConvergenceChecker, VariablesProcesses, DiscreteVariablesProcess, \
    ContinuousVariablesProcess, CostIntegrator, StochasticIntegrateAndFire


@dataclass
class CostMinimizer:
    """Processes implementing the cost function"""
    coefficients_2nd_order: Dense


@dataclass
class MacroStateReader:
    """Processes for checking convergence and reading network state encoding
    the solution ."""
    read_gate: ReadGate
    solution_readout: SolutionReadout
    cost_convergence_check: CostConvergenceChecker = None
    sat_convergence_check: SatConvergenceChecker = None


class SolverModelBuilder:
    def __init__(self, solver_process):
        self.constructor = None
        self.solver_process = solver_process

    def create_constructor(self):
        def constructor(self, proc):
            self.variables = VariablesProcesses()
            if hasattr(proc, 'discrete_variables'):
                self.variables.discrete = DiscreteVariablesProcess(
                    shape=proc.discrete_variables.shape)
            if hasattr(proc, 'continuous_variables'):
                self.variables.continuous = ContinuousVariablesProcess(
                    shape=proc.continuous_variables.shape)

            macrostate_reader = MacroStateReader(ReadGate(),
                                                 SolutionReadout(
                                                     shape=proc.variable_assignment.shape))
            if proc.problem.constraints:
                macrostate_reader.sat_convergence_check = \
                    SatConvergenceChecker(shape=proc.variable_assignment.shape)
                proc.vars.feasibility.alias(
                    macrostate_reader.sat_convergence_check.satisfaction)
            if hasattr(proc, 'cost_coefficients'):
                self.cost_minimizer = CostMinimizer(Dense(
                    # todo just using the last coefficient for now
                    weights=proc.cost_coefficients[2].init))
                self.variables.discrete.importances = proc.cost_coefficients[
                    1].init
                macrostate_reader.cost_convergence_check = \
                    CostConvergenceChecker(shape=proc.variable_assignment.shape)
                self.variables.discrete.satisfiability.connect(
                    macrostate_reader.cost_convergence_check.s_in)
                proc.vars.optimality.alias(
                    macrostate_reader.cost_convergence_check.cost)

            # Variable aliasing
            proc.vars.variable_assignment.alias(
                macrostate_reader.solution_readout.solution)
            # Connect processes
            macrostate_reader.cost_convergence_check.s_out.connect(
                macrostate_reader.read_gate.in_port)
            # macrostate_reader.cost_convergence_check.s_out.connect(
            #     self.variables.discrete.)
            macrostate_reader.read_gate.out_port.connect(
                macrostate_reader.solution_readout.in_port)
            macrostate_reader.solution_readout.ref_port.connect_var(
                self.variables.discrete.variable_assignment)
            self.cost_minimizer.coefficients_2nd_order.a_out.connect(
                self.variables.discrete.a_in)
            self.variables.discrete.s_out.connect(
                self.cost_minimizer.coefficients_2nd_order.s_in)
            self.macrostate_reader = macrostate_reader

        self.constructor = constructor

    @property
    def solver_model(self):
        SolverModel = type('OptimizationSolverModel',
                           (AbstractSubProcessModel,),
                           {'__init__': self.constructor}
                           )
        setattr(SolverModel, 'implements_process', self.solver_process)
        # setattr(SolverModel, 'implements_protocol', protocol)
        # Get requirements of parent class
        super_res = SolverModel.required_resources.copy()
        # Set new requirements on this cls to not overwrite parent class
        # requirements
        setattr(SolverModel, 'required_resources', super_res + [CPU])
        setattr(SolverModel, 'implements_protocol', LoihiProtocol)
        return SolverModel


@implements(ReadGate, protocol=LoihiProtocol)
@requires(CPU)
class ReadGatePyModel(PyLoihiProcessModel):
    solved: np.ndarray = LavaPyType(np.ndarray, np.int32, 32)
    in_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=32)
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32,
                                     precision=32)

    def run_spk(self):
        data = self.in_port.recv()
        self.out_port.send(data)
        self.solved[:] = data[0]
        if self.solved[0]:
            print('Cost', data)


@implements(SolutionReadout, protocol=LoihiProtocol)
@requires(CPU)
class SolutionReadoutPyModel(PyLoihiProcessModel):
    solution: np.ndarray = LavaPyType(np.ndarray, np.int32, 32)
    in_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=32)
    ref_port: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, np.int32,
                                     precision=32)
    solved = False

    def post_guard(self):
        return True

    def run_spk(self):
        data = self.in_port.recv()
        if data[0]:
            self.solved = True

    def run_post_mgmt(self):
        if self.solved:
            print('Reading solution')
            solution = self.ref_port.read()
            self.solution[:] = solution
            self._req_pause = True


@implements(proc=DiscreteVariablesProcess, protocol=LoihiProtocol)
@requires(CPU)
class DiscreteVariablesModel(AbstractSubProcessModel):

    def __init__(self, proc):
        # Instantiate child processes
        # The input shape is a 2D vector (shape of the weight matrix).
        wta_weight = -2
        shape = proc.proc_params.get("shape", (1,))
        weights = proc.proc_params.get("weights",
                                       wta_weight * np.logical_not(np.eye(
                                           shape[1] if len(shape) == 2 else 0)))
        noise_amplitude = proc.proc_params.get("noise_amplitude", 1)
        steps_to_fire = proc.proc_params.get("steps_to_fire", 10)
        importances = proc.proc_params.get("importances", 10)
        self.s_bit = StochasticIntegrateAndFire(shape=shape,
                                                increment=importances,
                                                noise_amplitude=noise_amplitude,
                                                steps_to_fire=steps_to_fire)

        if weights.shape != (0, 0):
            self.dense = Dense(weights=weights)
            self.s_bit.out_ports.messages.connect(self.dense.in_ports.s_in)
            self.dense.out_ports.a_out.connect(self.s_bit.in_ports.added_input)

        # Connect the parent InPort to the InPort of the Dense child-Process.
        proc.in_ports.a_in.connect(self.s_bit.in_ports.added_input)
        # Connect the OutPort of the LIF child-Process to the OutPort of the
        # parent Process.
        self.s_bit.out_ports.messages.connect(proc.out_ports.s_out)
        self.s_bit.out_ports.satisfiability.connect(
            proc.out_ports.satisfiability)
        proc.vars.variable_assignment.alias(self.s_bit.assignment)


@implements(proc=CostConvergenceChecker, protocol=LoihiProtocol)
@requires(CPU)
class CostConvergenceCheckerModel(AbstractSubProcessModel):
    def __init__(self, proc):
        # Instantiate child processes
        # The input shape is a 2D vector (shape of the weight matrix).
        shape = proc.proc_params.get("shape", (1,))
        weights = proc.proc_params.get("weights", np.ones((1, shape[0])))
        self.dense = Dense(weights=weights, num_message_bits=24)
        self.cost_integrator = CostIntegrator(shape=(1,))
        self.dense.out_ports.a_out.connect(
            self.cost_integrator.in_ports.cost_components)

        # Connect the parent InPort to the InPort of the Dense child-Process.
        proc.in_ports.s_in.connect(self.dense.in_ports.s_in)

        # Connect the OutPort of the LIF child-Process to the OutPort of the
        # parent Process.
        self.cost_integrator.out_ports.cost_out.connect(proc.out_ports.s_out)
        proc.vars.cost.alias(self.cost_integrator.vars.min_cost)


@implements(proc=CostIntegrator, protocol=LoihiProtocol)
@requires(CPU)
class CostIntegratorModel(PyLoihiProcessModel):
    cost_components: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)
    cost_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)

    min_cost: np.ndarray = LavaPyType(np.ndarray, int, 32)

    def run_spk(self):
        cost = self.cost_components.recv()
        if cost < self.min_cost:
            self.min_cost[:] = cost
        self.cost_out.send(cost)