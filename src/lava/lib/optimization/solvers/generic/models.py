# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/


import numpy as np
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort, PyRefPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.dense.process import Dense

from lava.lib.optimization.solvers.generic.dataclasses import (
    CostMinimizer,
    VariablesProcesses,
    MacroStateReader,
)
from lava.lib.optimization.solvers.generic.processes import (
    CostConvergenceChecker,
    ReadGate,
    SolutionReadout,
    SatConvergenceChecker,
    DiscreteVariablesProcess,
    ContinuousVariablesProcess,
    CostIntegrator,
    StochasticIntegrateAndFire,
)


class SolverModelBuilder:
    def __init__(self, solver_process):
        self.constructor = None
        self.solver_process = solver_process

    def create_constructor(self, target_cost):
        def constructor(self, proc):
            variables = VariablesProcesses()
            if hasattr(proc, "discrete_variables"):
                variables.discrete = DiscreteVariablesProcess(
                    shape=proc.discrete_variables.shape,
                    cost_diagonal=proc.cost_diagonal)
            if hasattr(proc, "continuous_variables"):
                variables.continuous = ContinuousVariablesProcess(
                    shape=proc.continuous_variables.shape
                )

            macrostate_reader = MacroStateReader(
                ReadGate(target_cost=target_cost),
                SolutionReadout(shape=proc.variable_assignment.shape,
                                target_cost=target_cost),
            )
            if proc.problem.constraints:
                macrostate_reader.sat_convergence_check = SatConvergenceChecker(
                    shape=proc.variable_assignment.shape
                )
                proc.vars.feasibility.alias(macrostate_reader.satisfaction)
            if hasattr(proc, "cost_coefficients"):
                cost_minimizer = CostMinimizer(
                    Dense(
                        # todo just using the last coefficient for now
                        weights=proc.cost_coefficients[2].init
                    )
                )
                variables.importances = proc.cost_coefficients[1].init
                macrostate_reader.cost_convergence_check = CostConvergenceChecker(
                    shape=proc.variable_assignment.shape
                )
                variables.local_cost.connect(macrostate_reader.cost_in)
                proc.vars.optimality.alias(macrostate_reader.min_cost)

            # Variable aliasing
            proc.vars.variable_assignment.alias(macrostate_reader.solution)
            # Connect processes
            macrostate_reader.update_buffer.connect(
                macrostate_reader.read_gate_in_port
            )
            # macrostate_reader.cost_convergence_check.s_out.connect(
            #     variables.discrete.)
            macrostate_reader.read_gate_do_readout.connect(
                macrostate_reader.solution_readout.read_solution
            )
            macrostate_reader.ref_port.connect_var(
                variables.variables_assignment

            )
            cost_minimizer.gradient_out.connect(variables.gradient_in)
            variables.state_out.connect(cost_minimizer.state_in)
            self.macrostate_reader = macrostate_reader
            self.variables = variables
            self.cost_minimizer = cost_minimizer

        self.constructor = constructor

    @property
    def solver_model(self):
        SolverModel = type(
            "OptimizationSolverModel",
            (AbstractSubProcessModel,),
            {"__init__": self.constructor},
        )
        setattr(SolverModel, "implements_process", self.solver_process)
        # setattr(SolverModel, 'implements_protocol', protocol)
        # Get requirements of parent class
        super_res = SolverModel.required_resources.copy()
        # Set new requirements on this cls to not overwrite parent class
        # requirements
        setattr(SolverModel, "required_resources", super_res + [CPU])
        setattr(SolverModel, "implements_protocol", LoihiProtocol)
        return SolverModel


@implements(ReadGate, protocol=LoihiProtocol)
@requires(CPU)
class ReadGatePyModel(PyLoihiProcessModel):
    solved: np.ndarray = LavaPyType(np.ndarray, np.int32, 32)
    new_solution: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32,
                                        precision=32)
    do_readout: PyOutPort = LavaPyType(
        PyOutPort.VEC_DENSE, np.int32, precision=32
    )
    target_cost: int = LavaPyType(int, np.int32, 32)
    min_cost: int = None

    def post_guard(self):
        return True

    def run_spk(self):
        data = self.new_solution.recv()
        self.do_readout.send(data)
        self.solved[:] = data[0]
        if data[0]:
            print(self.solved)
            self.min_cost = data[0]

    def run_post_mgmt(self):
        if self.min_cost:
            print("Cost", self.min_cost, end = "\r")
            if self.target_cost:
                if self.min_cost <= self.target_cost:
                    self._req_pause = True


@implements(SolutionReadout, protocol=LoihiProtocol)
@requires(CPU)
class SolutionReadoutPyModel(PyLoihiProcessModel):
    solution: np.ndarray = LavaPyType(np.ndarray, np.int32, 32)
    read_solution: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32,
                                         precision=32)
    ref_port: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, np.int32,
                                     precision=32)
    target_cost: int = LavaPyType(int, np.int32, 32)
    min_cost: int = None

    def post_guard(self):
        return True

    def run_spk(self):
        self.min_cost = None
        data = self.read_solution.recv()
        if data[0]:
            self.min_cost = data[0]

    def run_post_mgmt(self):
        if self.min_cost:
            solution = self.ref_port.read()
            self.solution[:] = solution
            print(f"Read solution: {self.solution}")
            # if self.target_cost:
            #     if self.min_cost <= self.target_cost:
            #         self._req_pause = True


@implements(proc=DiscreteVariablesProcess, protocol=LoihiProtocol)
@requires(CPU)
class DiscreteVariablesModel(AbstractSubProcessModel):
    def __init__(self, proc):
        # Instantiate child processes
        # The input shape is a 2D vector (shape of the weight matrix).
        wta_weight = -2
        shape = proc.proc_params.get("shape", (1,))
        diagonal = proc.proc_params.get("cost_diagonal")
        weights = proc.proc_params.get(
            "weights",
            wta_weight
            * np.logical_not(np.eye(shape[1] if len(shape) == 2 else 0)),
        )
        noise_amplitude = proc.proc_params.get("noise_amplitude", 1)
        steps_to_fire = proc.proc_params.get("steps_to_fire", 10)
        importances = proc.proc_params.get("importances", 10)
        self.s_bit = StochasticIntegrateAndFire(shape=shape,
                                                increment=importances,
                                                noise_amplitude=noise_amplitude,
                                                steps_to_fire=steps_to_fire,
                                                cost_diagonal=diagonal)
        if weights.shape != (0, 0):
            self.dense = Dense(weights=weights)
            self.s_bit.out_ports.messages.connect(self.dense.in_ports.s_in)
            self.dense.out_ports.a_out.connect(self.s_bit.in_ports.added_input)

        # Connect the parent InPort to the InPort of the Dense child-Process.
        proc.in_ports.a_in.connect(self.s_bit.in_ports.added_input)
        # Connect the OutPort of the LIF child-Process to the OutPort of the
        # parent Process.
        self.s_bit.out_ports.messages.connect(proc.out_ports.s_out)
        self.s_bit.out_ports.local_cost.connect(
            proc.out_ports.local_cost
        )
        proc.vars.variable_assignment.alias(self.s_bit.prev_assignment)


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
            self.cost_integrator.in_ports.cost_in
        )

        # Connect the parent InPort to the InPort of the Dense child-Process.
        proc.in_ports.cost_components.connect(self.dense.in_ports.s_in)

        # Connect the OutPort of the LIF child-Process to the OutPort of the
        # parent Process.
        self.cost_integrator.out_ports.update_buffer.connect(
            proc.out_ports.update_buffer)
        proc.vars.min_cost.alias(self.cost_integrator.vars.min_cost)


@implements(proc=CostIntegrator, protocol=LoihiProtocol)
@requires(CPU)
class CostIntegratorModel(PyLoihiProcessModel):
    cost_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)
    update_buffer: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)
    min_cost: np.ndarray = LavaPyType(np.ndarray, int, 32)

    def run_spk(self):
        cost = self.cost_in.recv()
        if cost < self.min_cost:
            self.min_cost[:] = cost
            self.update_buffer.send(cost)
        else:
            self.update_buffer.send(np.asarray([0]))


@implements(proc=StochasticIntegrateAndFire, protocol=LoihiProtocol)
@requires(CPU)
class StochasticIntegrateAndFireModel(PyLoihiProcessModel):
    added_input: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)
    replace_assignment: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)
    messages: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)
    local_cost: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)

    integration: np.ndarray = LavaPyType(np.ndarray, int, 32)
    increment: np.ndarray = LavaPyType(np.ndarray, int, 32)
    state: np.ndarray = LavaPyType(np.ndarray, int, 32)
    noise_amplitude: np.ndarray = LavaPyType(np.ndarray, int, 32)
    input_duration: np.ndarray = LavaPyType(np.ndarray, int, 32)
    min_state: np.ndarray = LavaPyType(np.ndarray, int, 32)
    min_integration: np.ndarray = LavaPyType(np.ndarray, int, 32)
    steps_to_fire: np.ndarray = LavaPyType(np.ndarray, int, 32)
    refractory_period: np.ndarray = LavaPyType(np.ndarray, int, 32)
    firing: np.ndarray = LavaPyType(np.ndarray, bool)
    prev_assignment: np.ndarray = LavaPyType(np.ndarray, int, 32)
    cost_diagonal: np.ndarray = LavaPyType(np.ndarray, int, 32)
    assignment: np.ndarray = LavaPyType(np.ndarray, int, 32)
    min_cost: np.ndarray = LavaPyType(np.ndarray, int, 32)

    def reset_state(self, firing_vector: np.ndarray):
        self.state[firing_vector] = 0

    def _update_buffers(self):
        self.prev_assignment[:] = self.assignment[:]
        self.assignment[:] = self.firing[:]

    def run_spk(self):
        self._update_buffers()
        added_input = self.added_input.recv()
        self.state = self.iterate_dynamics(added_input, self.state)
        local_cost = self.firing * (added_input + self.cost_diagonal
                                    * self.firing)
        self.firing[:] = self.do_fire(self.state)
        self.reset_state(firing_vector=self.firing[:])
        self.messages.send(self.firing[:])
        self.local_cost.send(-local_cost)

    def iterate_dynamics(self, added_input: np.ndarray, state: np.ndarray):
        integration_decay = 1
        state_decay = 0
        noise = self.noise_amplitude * np.random.randint(0, 2*self.increment,
                                                         self.integration.shape)
        self.integration[:] = self.integration * (1 - integration_decay)
        self.integration[:] += added_input.astype(int)
        state[:] = (
                state * (1 - state_decay)
                + self.integration
                + self.increment
                + noise
        )
        return state

    def do_fire(self, state):
        threshold = self.increment * self.steps_to_fire
        return state > threshold

    def is_satisfied(self, prev_assignment, integration):
        return np.logical_and(prev_assignment, np.logical_not(integration))
