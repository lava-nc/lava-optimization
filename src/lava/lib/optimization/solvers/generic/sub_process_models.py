# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

try:
    from lava.magma.core.model.nc.net import NetL2
except ImportError:
    class NetL2:
        pass

import numpy as np
from lava.lib.optimization.solvers.generic.hierarchical_processes import (
    CostConvergenceChecker,
    DiscreteVariablesProcess,
    StochasticIntegrateAndFire)
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.cost_integrator.process import CostIntegrator
from lava.proc.dense.process import Dense
from lava.magma.core.resources import Loihi2NeuroCore
import numpy as np
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.proc.lif.process import LIF, TernaryLIF
from lava.proc.scif.process import QuboScif


@implements(proc=DiscreteVariablesProcess, protocol=LoihiProtocol)
@requires(CPU)
class DiscreteVariablesModel(AbstractSubProcessModel):
    """Model for the DiscreteVariables process.

    The model composes a population of StochasticIntegrateAndFire units and
    connects them via Dense processes as to represent integer or binary
    variables.
    """

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
        step_size = proc.hyperparameters.get("step_size", 10)
        noise_amplitude = proc.hyperparameters.get("noise_amplitude", 1)
        noise_precision = proc.hyperparameters.get("noise_precision", 8)
        steps_to_fire = proc.hyperparameters.get("steps_to_fire", 10)
        init_value = proc.hyperparameters.get("init_value", np.zeros(shape))
        init_state = proc.hyperparameters.get("init_value", np.zeros(shape))
        self.s_bit = StochasticIntegrateAndFire(step_size=step_size,
                                                init_state=init_state,
                                                shape=shape,
                                                noise_amplitude=noise_amplitude,
                                                noise_precision=noise_precision,
                                                steps_to_fire=steps_to_fire,
                                                cost_diagonal=diagonal,
                                                init_value=init_value)
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
    """Model for the CostConvergence process.

    The model composes a CostIntegrator unit with incomming connections,
    in this way, downstream processes can be directly connected to the
    CostConvergence process.
    """

    def __init__(self, proc):
        # Instantiate child processes
        # The input shape is a 2D vector (shape of the weight matrix).
        shape = proc.proc_params.get("shape", (1,))
        weights = proc.proc_params.get("weights", np.ones((1, shape[0])))
        self.dense = Dense(weights=weights, num_message_bits=24)
        self.cost_integrator = CostIntegrator(shape=(1,), min_cost=0)
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


@implements(proc=StochasticIntegrateAndFire, protocol=LoihiProtocol)
@requires(Loihi2NeuroCore)
class StochasticIntegrateAndFireModelSCIF(AbstractSubProcessModel):
    """Model for the StochasticIntegrateAndFire process.

    The process is just a wrapper over the QuboScif process.
    # Todo deprecate in favour of QuboScif.
    """

    def __init__(self, proc):
        shape = proc.proc_params.get("shape", (1,))
        step_size = proc.proc_params.get("step_size", (1,))
        theta = proc.proc_params.get("steps_to_fire", (1,)) * step_size
        cost_diagonal = proc.proc_params.get("cost_diagonal", (1,))
        noise_amplitude = proc.proc_params.get("noise_amplitude", (1,))
        noise_precision = proc.proc_params.get("noise_precision", (1,))
        noise_shift = 16 - noise_precision
        self.scif = QuboScif(shape=shape,
                             step_size=step_size,
                             theta=theta,
                             cost_diag=cost_diagonal,
                             noise_amplitude=noise_amplitude,
                             noise_shift=noise_shift)
        proc.in_ports.added_input.connect(self.scif.in_ports.a_in)
        self.scif.s_wta_out.connect(proc.out_ports.messages)
        self.scif.s_sig_out.connect(proc.out_ports.local_cost)

        proc.vars.prev_assignment.alias(self.scif.vars.spk_hist)
        proc.vars.state.alias(self.scif.vars.state)
        proc.vars.cost_diagonal.alias(self.scif.vars.cost_diagonal)
        proc.vars.noise_amplitude.alias(self.scif.vars.noise_ampl)


@implements(proc=StochasticIntegrateAndFire, protocol=LoihiProtocol)
@requires(CPU)
class StochasticIntegrateAndFireModel(PyLoihiProcessModel):
    """CPU Model for the StochasticIntegrateAndFire process.

    The model computes stochastic gradient and local cost both based on
    the input received from the recurrent connectivity ot other units.
    # Todo deprecate in favour of QuboScif.
    """
    added_input: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)
    messages: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)
    local_cost: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)

    integration: np.ndarray = LavaPyType(np.ndarray, int, 32)
    step_size: np.ndarray = LavaPyType(np.ndarray, int, 32)
    state: np.ndarray = LavaPyType(np.ndarray, int, 32)
    noise_amplitude: np.ndarray = LavaPyType(np.ndarray, int, 32)
    noise_precision: np.ndarray = LavaPyType(np.ndarray, int, 32)
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
        self.local_cost.send(local_cost)

    def iterate_dynamics(self, added_input: np.ndarray, state: np.ndarray):
        integration_decay = 1
        state_decay = 0
        noise = self.noise_amplitude * np.random.randint(0, 2 * self.step_size,
                                                         self.integration.shape)
        self.integration[:] = self.integration * (1 - integration_decay)
        self.integration[:] += added_input.astype(int)
        decayed_state = state * (1 - state_decay)
        state[:] = decayed_state + self.integration + self.step_size + noise
        return state

    def do_fire(self, state):
        threshold = self.step_size * self.steps_to_fire
        return state > threshold

    def is_satisfied(self, prev_assignment, integration):
        return np.logical_and(prev_assignment, np.logical_not(integration))
